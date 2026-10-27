"""Turn a raw ground-motion record into what the models and seisim expect.

The shipped models were trained on GMP_X6's signal path (mirrored here from
GMP_X6/X8 `py_data/protocol.py`; keep the two in sync):

  ground   raw -> `scipy.signal.decimate` in q=8 steps (zero-phase IIR, the
           call the seisim notebooks use) -> mean removed. This is the
           ground motion you hand to `Seisim.simulate`.
  input    seisim's prediction filter applied to the ground: 4th-order
           Butterworth band-pass 0.05-0.5 Hz, zero-phase (`sosfiltfilt`).
           Bit-identical to the `g_filt` that `Seisim.simulate` computes
           internally and passes to its predictor, so a model in seisim's
           loop sees exactly what it was trained on.

`GMP` does no filtering, decimation or resampling: feed `predict_long` the
`input`, at the model's `sampling_rate`. Every checkpoint records its band,
rates and protocol (`recipe`); all shipped ones share this path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.signal
import torch

# seisim's prediction band — inline in `seisim.seisim.Seisim.simulate`
# (N = 4, lowcut = 0.05, highcut = 0.5).
BAND = {"order": 4, "low": 0.05, "high": 0.5}

# Decimation factor per scipy.signal.decimate call (512 -> 8 Hz in two
# q=8 steps; the notebooks' 64 -> 8 Hz is one).
Q_STEP = 8


def _q_steps(fs_raw: int, fs: int) -> list[int]:
    if fs_raw % fs:
        raise ValueError(f"{fs_raw} Hz -> {fs} Hz is not an integer decimation")
    factor, steps = fs_raw // fs, []
    while factor > 1:
        q = Q_STEP if factor % Q_STEP == 0 else factor
        steps.append(q)
        factor //= q
    return steps


def decimate(raw: np.ndarray, fs_raw: int, fs: int,
             zero_phase: bool = True) -> np.ndarray:
    """`scipy.signal.decimate` (IIR anti-aliasing) in q=8 steps, float64."""
    y = np.asarray(raw, dtype=np.float64)
    for q in _q_steps(fs_raw, fs):
        y = scipy.signal.decimate(y, q, zero_phase=zero_phase)
    return y


def ground(raw: np.ndarray, fs_raw: int, fs: int) -> np.ndarray:
    """Ground motion passed to seisim: decimated raw data, mean removed."""
    g = decimate(raw, fs_raw, fs, zero_phase=True)
    return g - np.mean(g)


def band_sos(fs: float) -> np.ndarray:
    return scipy.signal.butter(BAND["order"], (BAND["low"], BAND["high"]),
                               btype="bandpass", fs=fs, output="sos")


def zero_phase_band(g: np.ndarray, fs: float) -> np.ndarray:
    """seisim's `g_filt`: the zero-phase band-pass of the ground."""
    return scipy.signal.sosfiltfilt(band_sos(fs), g)


def prepare(raw, fs_in: int = 512, fs: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """`(ground, input)` of a 1-D raw record at `fs_in` Hz, both 1-D float64
    at `fs` Hz: `ground` for `Seisim.simulate`, `input` (seisim's `g_filt`)
    for `GMP.predict_long`. `fs_in / fs` must be a whole number."""
    g = ground(np.asarray(raw, dtype=np.float64).ravel(), int(fs_in), int(fs))
    return g, zero_phase_band(g, fs)


def recipe(checkpoint_path: str | Path) -> dict:
    """The signal path a checkpoint was trained on: `{fs_in, fs, protocol,
    low, high, order}`."""
    d = torch.load(str(checkpoint_path), map_location="cpu",
                   weights_only=False)["data"]
    return {"fs_in": int(d["original_sampling_rate"]), "fs": int(d["sampling_rate"]),
            "protocol": str(d["protocol"]),
            **{k: d["band"][k] for k in ("low", "high", "order")}}
