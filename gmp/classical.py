"""The paper's classical baselines, with the `GMP` interface (numpy only).

  Wiener       GMP_X6's closed-form (Toeplitz) Wiener filter: one (H, K) FIR
               map fitted on the pooled training split, the ridge picked per
               horizon on validation RMSP. Loaded from the fit file
               exp2_classical wrote (`find("wiener", input_sec=...)`).
  Persistence  the last input sample, held for H samples.

The professor's polynomial extrapolator is `seisim.predictor.ConstantOrder`.

Both expose `input_window`, `output_window`, `sampling_rate`, `predict` and
`predict_long` exactly as `GMP` does (the stitching is GMP's own code), so
`forecast_stream`, `SeisimGMP` and `run_closed_loop` take them unchanged.
Forecasts are float64.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .predictor import GMP


class _Stitched:
    """GMP's predict / predict_long over a subclass's `_predict_window`."""
    input_dtype = np.float64
    target_mode = "absolute"
    device = "cpu"
    _validate_signal = GMP._validate_signal
    predict = GMP.predict
    predict_long = GMP.predict_long
    _stitch_seamed = GMP._stitch_seamed
    _stitch_overlap_add = GMP._stitch_overlap_add


class Wiener(_Stitched):
    model_type = "wiener"

    def __init__(self, fit_path: str | Path, output_sec: float,
                 sampling_rate: int = 8):
        z = np.load(fit_path)
        j = np.flatnonzero(np.isclose(z["horizons"], output_sec))
        if len(j) != 1:
            raise ValueError(f"{fit_path} has horizons {list(z['horizons'])} s, "
                             f"not {output_sec:g} s")
        H = int(round(output_sec * sampling_rate))
        k = int(z["chosen"][j[0]])
        self.W = np.asarray(z["W"][k][:H], dtype=np.float64)    # (H, K)
        self.ridge = float(z["ridges"][k])
        self.sampling_rate = int(sampling_rate)
        self.input_window, self.output_window = self.W.shape[1], H

    def _predict_window(self, x_kc: np.ndarray) -> np.ndarray:
        single = (x_kc.ndim == 2)
        x = np.asarray(x_kc[None] if single else x_kc, dtype=np.float64)
        out = np.einsum("hk,bkc->bhc", self.W, x)
        return out[0] if single else out


class Persistence(_Stitched):
    model_type = "persistence"
    input_window = 1

    def __init__(self, output_sec: float, sampling_rate: int = 8):
        self.sampling_rate = int(sampling_rate)
        self.output_window = int(round(output_sec * sampling_rate))

    def _predict_window(self, x_kc: np.ndarray) -> np.ndarray:
        single = (x_kc.ndim == 2)
        x = np.asarray(x_kc[None] if single else x_kc, dtype=np.float64)
        out = np.repeat(x[:, -1:, :], self.output_window, axis=1)
        return out[0] if single else out
