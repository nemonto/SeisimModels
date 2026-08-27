"""Turn raw ground motion into what a checkpoint expects.

`GMP` does no filtering, decimation or resampling — its `sampling_rate` is
metadata. Every model here was trained on zero-phase Butterworth bandpassed
0.06-0.6 Hz data decimated to the checkpoint's own rate (8 Hz for GMP_X5
models, 16 Hz for the legacy GMP_X3 ones), and the scaler baked into the
checkpoint assumes exactly that. Feed it anything else and the forecast
degrades silently.

`bandpass`, `decimate_to` and `bandpass_decimate` are lifted verbatim from
GMP_X5 `py_data/datamodule.py` so training and inference share one recipe.
`recipe()` and `prepare()` read the parameters out of the checkpoint, so a
caller never hardcodes 0.06 / 0.6 / 4 / 512 / 8.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt, resample_poly


def bandpass(arr: np.ndarray, fs_in: int, low: float, high: float,
             order: int) -> np.ndarray:
    """Zero-phase Butterworth bandpass at fs_in."""
    sos = butter(order, [low, high], btype="bandpass", fs=fs_in, output="sos")
    return sosfiltfilt(sos, arr, axis=0).astype(np.float64)


def decimate_to(arr: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Resample to fs_out via polyphase FIR (safe for any integer down-factor)."""
    if fs_out == fs_in:
        return arr.astype(np.float32, copy=False)
    factor = int(round(fs_in / fs_out))
    return resample_poly(arr, up=1, down=factor, axis=0).astype(np.float32)


def bandpass_decimate(arr: np.ndarray, fs_in: int, fs_out: int,
                      low: float, high: float, order: int) -> np.ndarray:
    return decimate_to(bandpass(arr, fs_in, low, high, order), fs_in, fs_out)


def recipe(checkpoint_path: str | Path) -> dict:
    """The preprocessing a checkpoint was trained with.

    Returns `{fs_in, fs_out, low, high, order}`. `fs_in` is the raw rate the
    training data came from (512 Hz); pass your own `fs_in` to `prepare` when
    your recording is at a different rate.
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu",
                      weights_only=False)
    d = ckpt["data"]
    filt = d["filter_cfg"] if "filter_cfg" in d else d["filter"]
    return {
        "fs_in":  int(d["original_sampling_rate"]),
        "fs_out": int(d["sampling_rate"]),
        "low":    float(filt["low"]),
        "high":   float(filt["high"]),
        "order":  int(filt["order"]),
    }


def prepare(raw, checkpoint_path: str | Path, fs_in: int | None = None
            ) -> np.ndarray:
    """Bandpass + decimate `raw` for one checkpoint.

    raw    : (T,) or (T, C) at `fs_in` samples/s.
    fs_in  : rate of `raw`; defaults to the checkpoint's training rate (512 Hz).
             `fs_in / fs_out` must be a whole number.

    Returns (T', C) float32, ready for `GMP.predict_long`.
    """
    r = recipe(checkpoint_path)
    fs_in = int(fs_in if fs_in is not None else r["fs_in"])
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    factor = fs_in / r["fs_out"]
    if abs(factor - round(factor)) > 1e-9:
        raise ValueError(
            f"fs_in={fs_in} is not an integer multiple of the model's "
            f"{r['fs_out']} Hz — resample first."
        )
    return bandpass_decimate(arr, fs_in, r["fs_out"],
                             r["low"], r["high"], r["order"])
