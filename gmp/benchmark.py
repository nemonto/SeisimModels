"""The benchmark's forecast stream and open-loop score, numpy/scipy only.

Mirrors GMP_X6/X8 (`py_data.protocol.control_steps` / `assemble_stream`,
`py_utils.metrics.welch_kwargs` / `compute_rmsp`), so `rmsp` on a test
record reproduces the RMSP column of `checkpoints/model_index.csv`.

  forecast_stream  every predictor is first called at the control start
                   (64 s), then every H samples (seisim's control grid);
                   its H-sample forecasts, concatenated, are the seamed
                   stream aligned to input[i0:].
  rmsp             in-band (0.05-0.5 Hz) NRMSE of that stream against the
                   input it forecasts, from one Welch estimate: 600 s Hann
                   segments, 50 % overlap, constant detrend; the band is the
                   open interval, as seisim's BLS masks it. Lower is better;
                   1 = no better than predicting zero.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .preprocess import BAND

CONTROL_START_SEC = 64.0
WELCH_SEC = 600.0
WELCH_OVERLAP = 0.5


def welch_kwargs(n: int, fs: float) -> dict:
    """scipy.signal.welch settings for an n-sample stream at fs (shared by
    RMSP and the closed-loop benchmark)."""
    nperseg = int(min(n, round(WELCH_SEC * fs)))
    return {"nperseg": nperseg, "noverlap": int(nperseg * WELCH_OVERLAP),
            "window": "hann", "detrend": "constant"}


def forecast_stream(model, x, control_start_sec: float = CONTROL_START_SEC
                    ) -> np.ndarray:
    """The seamed forecast stream of `model` (GMP, Wiener or Persistence)
    on the 1-D input `x`, aligned to x[i0:], i0 = control start in samples."""
    x = np.asarray(x, dtype=np.float64).ravel()
    i0, K = int(round(control_start_sec * model.sampling_rate)), model.input_window
    if K > i0:
        raise ValueError(f"input window {K} exceeds the control start {i0}")
    return model.predict_long(x[i0 - K:], mode="seamed")[:, 0].astype(np.float64)


def rmsp(pred: np.ndarray, target: np.ndarray, fs: float) -> float:
    """Band-integrated ASD ratio of error to target (one channel)."""
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(target, dtype=np.float64).ravel()
    kw = welch_kwargs(len(t), fs)
    f, P_e = welch(t - p, fs=fs, **kw)
    _, P_t = welch(t, fs=fs, **kw)
    m = (f > BAND["low"]) & (f < BAND["high"])
    return float(np.sqrt(np.trapezoid(P_e[m], f[m]))
                 / np.sqrt(np.trapezoid(P_t[m], f[m])))
