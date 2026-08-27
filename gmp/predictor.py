"""GMP — zero-config inference from a unified checkpoint.

Everything needed for inference (model config, window sizes, sampling rate,
scaler statistics, target mode) is stored inside `unified_checkpoint.ckpt`,
so construction takes only the checkpoint path. Delta-target checkpoints
(`target_mode == "delta"`) are integrated back to the absolute signal inside
`_predict_window`; callers always receive absolute-signal forecasts.

`predict(x)` does a single window forecast.

`predict_long(signal, mode)` produces a long forecast aligned to `signal[K:]`,
applying one of two stitching modes:

  seamed        stride=H, hard concat (benchmark default; broadband seams).
  overlap_add   stride=H/2, Hann cross-fade in the overlap region.

This package is self-contained (stdlib + numpy + torch only): `_models.py`
and `_blocks.py` are ports of GMP_X5's `py_models/models.py` and
`py_models/model_blocks.py` — keep them in sync when model definitions change.

`GMP` reads X5-schema checkpoints. The five legacy GMP_X3 checkpoints under
`checkpoints/legacy/` use an older schema — load anything path-driven with
`gmp.load_gmp`, which dispatches to the right class.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ._models import build_model


_MODES = ("seamed", "overlap_add")

# predict_long batches every window of a long signal into one call; cap the
# per-forward batch so hour-long signals with large input windows (e.g. the
# legacy 45 s PatchTST) don't exhaust memory.
_MAX_BATCH = 1024


class GMP:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)

        self.model_type    = ckpt["meta"]["model_type"]
        d                  = ckpt["data"]
        self.input_window  = int(d["input_window"])
        self.output_window = int(d["output_window"])
        self.sampling_rate = int(d["sampling_rate"])
        self.target_mode   = str(d["target_mode"])
        self.scaler_mean = np.asarray(d["scaler"]["mean"], dtype=np.float32)
        self.scaler_std  = np.asarray(d["scaler"]["std"],  dtype=np.float32)

        self.model = build_model(self.model_type, ckpt["model"]["config"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------ utils
    def _validate_signal(self, signal) -> np.ndarray:
        sig = np.asarray(signal, dtype=np.float32)
        if sig.ndim == 1:
            sig = sig[:, None]
        return sig

    @torch.no_grad()
    def _predict_window(self, x_kc: np.ndarray) -> np.ndarray:
        # x_kc: (K, C) or (B, K, C). Returns absolute-signal forecasts in
        # original units regardless of the checkpoint's target mode.
        single = (x_kc.ndim == 2)
        if single:
            x_kc = x_kc[None]
        x_scaled = (x_kc - self.scaler_mean) / self.scaler_std
        x_t = torch.from_numpy(x_scaled.astype(np.float32))
        out = np.concatenate([
            self.model(x_t[i:i + _MAX_BATCH].to(self.device)).cpu().numpy()
            for i in range(0, len(x_t), _MAX_BATCH)
        ], axis=0)
        if self.target_mode == "delta":
            # The model emits scaled increments Δy; integrate against the last
            # input sample in the same scaled space, then inverse-transform
            # once (mirror of py_data.datamodule.reconstruct_from_delta).
            out = x_scaled[:, -1:, :] + np.cumsum(out, axis=1)
        out = out * self.scaler_std + self.scaler_mean
        return out[0] if single else out

    # ------------------------------------------------------------------- API
    def predict(self, x):
        """Single window forecast.

        x : (K, C) or (B, K, C). Returns (H, C) or (B, H, C).
        """
        x = np.asarray(x)
        if x.shape[-2] != self.input_window:
            raise ValueError(
                f"Expected input_window={self.input_window}, got {x.shape[-2]}"
            )
        if x.ndim == 2:
            x = x.astype(np.float32)
        return self._predict_window(x)

    def predict_long(self, signal, mode: str = "seamed") -> np.ndarray:
        """Forecast a long stream aligned to `signal[K:]`.

        Returns shape `(len(signal) - K, C)`.
        """
        if mode not in _MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {_MODES}")
        sig = self._validate_signal(signal)
        T, C = sig.shape
        K, H = self.input_window, self.output_window
        if T < K + H:
            raise ValueError(f"signal too short: need >= K+H = {K + H}, got {T}")

        if mode == "seamed":
            return self._stitch_seamed(sig, K, H)
        return self._stitch_overlap_add(sig, K, H)

    # ----------------------------------------------------------- stitch impls
    def _stitch_seamed(self, sig: np.ndarray, K: int, H: int) -> np.ndarray:
        """Hard concatenation: forecasts are launched every H samples and each
        output sample comes from exactly one forecast. Adjacent forecasts can
        jump at the boundary — broadband seam noise above the signal band."""
        T, C = sig.shape
        out_len = T - K
        out = np.zeros((out_len, C), dtype=np.float32)
        positions = list(range(0, out_len - H + 1, H))
        # Batch all windows in one forward pass.
        x = np.stack([sig[p:p + K] for p in positions], axis=0)
        preds = self._predict_window(x)            # (n, H, C)
        for k, p in enumerate(positions):
            out[p:p + H] = preds[k]
        # If out_len % H != 0, fill the last partial bucket with the next pred.
        last_p = positions[-1] + H if positions else 0
        if last_p < out_len:
            x_last = sig[out_len - K:out_len][None]
            tail = self._predict_window(x_last)
            out[out_len - H:out_len] = tail[0]
        return out

    def _stitch_overlap_add(self, sig: np.ndarray, K: int, H: int) -> np.ndarray:
        """Hann cross-fade concatenation of overlapping forecasts.

        How the predictions are combined, step by step:

        1. Forecasts are launched every `stride = H // 2` samples, so each
           H-sample forecast overlaps the next one by H/2 — every output
           sample is covered by (up to) two forecasts. A final forecast is
           pinned at `out_len - H` so the stream end is always covered.
        2. Each forecast is weighted by a zero-free Hann window `win` of
           length H (`np.hanning(H + 2)[1:-1]`, so no sample gets weight 0):
           weight ramps up over the first half of the forecast and down over
           the second half.
        3. Weighted forecasts are summed into `sum_`, and the weights
           themselves into `weight`; the output is `sum_ / weight`. At every
           sample the two contributions therefore cross-fade: the *late*
           (stale, far-horizon) half of the previous forecast fades out while
           the *fresh*, near-horizon half of the next forecast fades in.

        """
        T, C = sig.shape
        out_len = T - K
        stride = max(1, H // 2)
        positions = list(range(0, out_len - H + 1, stride))
        if not positions or positions[-1] != out_len - H:
            positions.append(out_len - H)

        x = np.stack([sig[p:p + K] for p in positions], axis=0)
        preds = self._predict_window(x)            # (n, H, C)
        win = np.hanning(H + 2)[1:-1].astype(np.float32)   # zero-free Hann
        win = win[:, None]                                  # (H, 1)

        sum_   = np.zeros((out_len, C), dtype=np.float32)
        weight = np.zeros((out_len, 1), dtype=np.float32)
        for k, p in enumerate(positions):
            sum_  [p:p + H] += preds[k] * win
            weight[p:p + H] += win
        weight = np.where(weight > 0, weight, 1.0)
        return sum_ / weight
