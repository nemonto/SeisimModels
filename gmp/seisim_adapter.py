"""Drive the seisim closed-loop simulator with a GMP (or Wiener, Persistence)
forecast — the GMP_X6/X8 benchmark setting.

This module imports `seisim` and `control`, which are **not** dependencies of
this package — seisim is unpackaged, so it is only importable when its own
directory is on `sys.path` (normally because that is the notebook's working
directory). Import this module only from code that already has seisim
available; `gmp/__init__.py` deliberately does not.

The benchmark (GMP_X8_CS `py_exp2/adapters.py`):
  * seisim untouched: `Seisim.simulate` band-passes the ground itself and
    hands its predictor the history of `g_filt` — the models' training
    input — so you pass the *ground* (`prepare(raw)[0]`), not the band.
  * control starts at 64 s for every predictor: n_history = 64 s, so no
    actuation before it; the model reads the last `input_window` samples of
    the history seisim passes.
  * control horizon = prediction horizon (seamed replay).
  * scored on the controlled interval [64 s, end) with the shared Welch
    settings (`gmp.benchmark.welch_kwargs`).
On a test record this reproduces the BLS column of model_index.csv.
"""
from __future__ import annotations

import numpy as np

from .benchmark import CONTROL_START_SEC, welch_kwargs

# The seisim notebooks' plant: 1.25 Hz resonance, Q = 1.11.
WN = 7.85398163397
Q  = 1.11111111


class SeisimGMP:
    """Adapter: a predictor inside seisim's MPCFF controller.

    Satisfies seisim's predictor contract — an `n_history` attribute plus a
    `predict(y)` method; `MPCFF.get_actuation` slices `y[-n_history:]` and
    applies `-forecast[:n_control]`. The plant must run at the model's own
    sampling rate.
    """
    def __init__(self, model, control_start_sec: float = CONTROL_START_SEC):
        self.model = model
        self.n_history = int(round(control_start_sec * model.sampling_rate))
        self.n_predict = int(model.output_window)
        if model.input_window > self.n_history:
            raise ValueError(f"input window {model.input_window} exceeds the "
                             f"control start {self.n_history}")

    def run(self, data_in):
        return self.predict(data_in)

    def predict(self, y):
        x = np.asarray(y, dtype=np.float64)[-self.model.input_window:].reshape(-1, 1)
        return np.asarray(self.model.predict(x), dtype=np.float64).reshape(-1)


def make_plant(wn: float = WN, q: float = Q):
    """The benchmark plant as a `control` transfer function."""
    import control
    s = control.tf("s")
    return wn ** 2 / (s ** 2 + wn / q * s + wn ** 2)


def run_closed_loop(model, ground, control_start_sec: float = CONTROL_START_SEC,
                    tf_sys=None) -> dict:
    """Simulate feedforward isolation driven by `model` and score it.

    ground : 1-D ground motion at `model.sampling_rate`, mean removed
             (`gmp.prepare(raw)[0]`), in the training data's units.

    Returns seisim's benchmark metrics on [control start, end): BLS, ABLS,
    SR, MS, rms_ground, rms_platform.
    """
    from seisim.benchmark  import Benchmark
    from seisim.controller import MPCFF
    from seisim.seisim     import Seisim
    from seisim.system     import System

    ground = np.asarray(ground, dtype=np.float64).ravel()
    fs = float(model.sampling_rate)
    t = np.arange(len(ground)) / fs
    pred = SeisimGMP(model, control_start_sec)
    ctrl = MPCFF(model=None, n_predict=pred.n_predict,
                 n_control=pred.n_predict, predictor=pred)
    sim = Seisim(system=System(tf_sys if tf_sys is not None else make_plant(), 1.0 / fs),
                 controller=ctrl)
    motion = sim.simulate(t, ground)
    i0 = pred.n_history
    g, u = ground[i0:], motion[i0:]
    b = Benchmark(t[i0:], g, u).run(**welch_kwargs(len(g), fs))
    return {"BLS":  b["band_limited_suppression"],
            "ABLS": b["auto_band_limited_suppression"],
            "SR":   b["suppression_ratio"],
            "MS":   b["maximum_suppression"],
            "rms_ground":   b["rms_ground"],
            "rms_platform": b["rms_platform"]}
