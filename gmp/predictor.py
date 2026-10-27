"""GMP — zero-config inference from a unified checkpoint.

Everything needed for inference (model config, window sizes, sampling rate,
scaler statistics, target mode) is stored inside `unified_checkpoint.ckpt`,
so construction takes only the checkpoint path. Delta-target checkpoints
(`target_mode == "delta"`) are integrated back to the absolute signal inside
`_predict_window`; callers always receive absolute-signal forecasts.

The predictor does not filter. It expects the stream the model was trained
on, at `sampling_rate`: under protocol `zero_phase` (the default) the
ground motion band-passed 0.05-0.5 Hz with seisim's zero-phase filter (the
`g_filt` seisim hands a predictor); under `causal`, the causal chain of
GMP_X6 `py_data/protocol.py`. `protocol` and `band` are exposed for the
caller to check.

`predict(x)` does a single window forecast (or a batch of windows).

Precision: windows are scaled and fed in the model's `input_dtype` and the
forecasts come back in it — float32 for every family, float64 for a
proposed checkpoint with `linear_dtype: float64` (its frozen W then runs
on exactly the float64 input the classical baseline sees). Pass float64
streams: a float32 model rounds them itself, as it was trained. Every
forward pass runs with TF32 off (cuDNN convolutions and RNNs, CUDA
matmuls; the caller's settings are restored after), so a GPU gives the
float32 forecasts a CPU gives: PyTorch's default cuDNN TF32 moved a
ModernTCN forecast by 12 % of its peak (GMP_X8_CS docs/findings.md 1).

`predict_long(signal, mode)` produces a long forecast aligned to `signal[K:]`,
applying one of two stitching modes:

  seamed        stride=H, hard concat (benchmark default; broadband seams).
                Output sample p always comes from the forecast issued from
                signal[p0 : p0+K], p0 = p - p % H — the final partial
                bucket too (its forecast is cut, as seisim cuts the last
                actuation).
  overlap_add   stride=H/2, Hann cross-fade in the overlap region.

This package is self-contained (numpy + torch only): `_models.py` and
`_blocks.py` are ports of `py_models/models.py` and
`py_models/model_blocks.py` — keep them in sync when model definitions change.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from ._models import build_model


_MODES = ("seamed", "overlap_add")

# predict / predict_long batch every window into one call; cap the
# per-forward batch so hour-long signals at 64 Hz don't exhaust memory.
_MAX_BATCH = 1024


def default_device() -> str:
    """The device GMP takes when none is given."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@contextmanager
def _no_tf32():
    """Full float32 for one forward pass (module docstring): TF32 off for
    CUDA matmuls and cuDNN, the caller's settings restored after."""
    matmul, cudnn = torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(matmul)
        torch.backends.cudnn.allow_tf32 = cudnn


class GMP:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.device = device or default_device()
        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)

        self.model_type    = ckpt["meta"]["model_type"]
        self.variant       = ckpt["meta"].get("variant", "-")
        d                  = ckpt["data"]
        self.input_window  = int(d["input_window"])
        self.output_window = int(d["output_window"])
        self.sampling_rate = int(d["sampling_rate"])
        self.target_mode   = str(d["target_mode"])
        self.protocol      = str(d["protocol"])
        self.band          = dict(d["band"])
        self.scaler_mean = np.asarray(d["scaler"]["mean"], dtype=np.float32)
        self.scaler_std  = np.asarray(d["scaler"]["std"],  dtype=np.float32)

        self.model = build_model(self.model_type, ckpt["model"]["config"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        # Window / forecast precision (module docstring).
        self.input_dtype = (np.float64 if getattr(self.model, "input_dtype",
                                                  torch.float32) == torch.float64
                            else np.float32)

    # ------------------------------------------------------------------ utils
    def _validate_signal(self, signal) -> np.ndarray:
        sig = np.asarray(signal, dtype=self.input_dtype)
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
        dt = self.input_dtype
        x_scaled = ((x_kc.astype(dt) - self.scaler_mean.astype(dt))
                    / self.scaler_std.astype(dt)).astype(dt)
        x_t = torch.from_numpy(x_scaled)
        with _no_tf32():
            out = np.concatenate([
                self.model(x_t[i:i + _MAX_BATCH].to(self.device)).cpu().numpy()
                for i in range(0, len(x_t), _MAX_BATCH)
            ], axis=0)
        if self.target_mode == "delta":
            # The model emits scaled increments Δy; integrate against the last
            # input sample in the same scaled space, then inverse-transform
            # once (mirror of py_data.datamodule.reconstruct_from_delta).
            out = x_scaled[:, -1:, :] + np.cumsum(out, axis=1)
        out = (out * self.scaler_std.astype(dt) + self.scaler_mean.astype(dt)).astype(dt)
        return out[0] if single else out

    # ------------------------------------------------------------------- API
    def predict(self, x):
        """Single window forecast.

        x : (K, C) or (B, K, C). Returns (H, C) or (B, H, C), in the
        model's input dtype.
        """
        x = np.asarray(x, dtype=self.input_dtype)
        if x.shape[-2] != self.input_window:
            raise ValueError(
                f"Expected input_window={self.input_window}, got {x.shape[-2]}"
            )
        return self._predict_window(x)

    def predict_long(self, signal, mode: str = "seamed") -> np.ndarray:
        """Forecast a long stream aligned to `signal[K:]`.

        Returns shape `(len(signal) - K, C)`. `seamed` needs len(signal) > K
        (a stream shorter than one horizon gets the leading part of one
        forecast); `overlap_add` needs len(signal) >= K + H.
        """
        if mode not in _MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {_MODES}")
        sig = self._validate_signal(signal)
        T, C = sig.shape
        K, H = self.input_window, self.output_window
        need = K + 1 if mode == "seamed" else K + H
        if T < need:
            raise ValueError(f"signal too short for {mode}: need >= {need}, got {T}")

        if mode == "seamed":
            return self._stitch_seamed(sig, K, H)
        return self._stitch_overlap_add(sig, K, H)

    # ----------------------------------------------------------- stitch impls
    def _stitch_seamed(self, sig: np.ndarray, K: int, H: int) -> np.ndarray:
        """Hard concatenation: forecasts are launched every H samples and each
        output sample comes from exactly one forecast. Adjacent forecasts can
        jump at the boundary — broadband seam noise above the signal band.

        The forecast for out[p : p+H] reads sig[p : p+K] (the K samples
        before signal index K + p). When (T - K) % H != 0 the last bucket
        p = last is partial: it is forecast from sig[last : last+K] and only
        its leading out_len - last samples are kept."""
        T, C = sig.shape
        out_len = T - K
        out = np.zeros((out_len, C), dtype=self.input_dtype)
        positions = list(range(0, out_len, H))
        # Batch all windows in one forward pass.
        x = np.stack([sig[p:p + K] for p in positions], axis=0)
        preds = self._predict_window(x)            # (n, H, C)
        for k, p in enumerate(positions):
            n = min(H, out_len - p)
            out[p:p + n] = preds[k][:n]
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

        This removes the seam jumps of `seamed`, but the averaging of two
        imperfectly-phased forecasts shrinks amplitude in the passband —
        X4 exp4 measured markedly worse closed-loop suppression (BLS, ABLS,
        SR, MS) despite the better open-loop RMSE/RMSP. Use `seamed` for
        control-drive benchmarks; treat this mode as the smooth-signal
        reference.
        """
        T, C = sig.shape
        out_len = T - K
        stride = max(1, H // 2)
        positions = list(range(0, out_len - H + 1, stride))
        if not positions or positions[-1] != out_len - H:
            positions.append(out_len - H)

        x = np.stack([sig[p:p + K] for p in positions], axis=0)
        preds = self._predict_window(x)            # (n, H, C)
        dt = self.input_dtype
        win = np.hanning(H + 2)[1:-1].astype(dt)           # zero-free Hann
        win = win[:, None]                                  # (H, 1)

        sum_   = np.zeros((out_len, C), dtype=dt)
        weight = np.zeros((out_len, 1), dtype=dt)
        for k, p in enumerate(positions):
            sum_  [p:p + H] += preds[k] * win
            weight[p:p + H] += win
        weight = np.where(weight > 0, weight, 1.0)
        return sum_ / weight
