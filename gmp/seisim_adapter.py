"""Drive the seisim closed-loop simulator with a GMP forecast.

This module imports `seisim` and `control`, which are **not** dependencies of
this package — seisim is unpackaged, so it is only importable when its own
directory is on `sys.path` (normally because that is the notebook's working
directory). Import this module only from code that already has seisim
available; `gmp/__init__.py` deliberately does not.

    cd H:\\seisim-master
    python H:\\SeisimModels\\examples\\seisim_closed_loop.py --help

The adapter, the plant and the wiring are the ones GMP_X5's exp2 benchmark
used (`py_exp2/exp2_eval.py`), so numbers produced here are comparable with
the BLS/ABLS/SR/MS columns in `checkpoints/model_index.csv`.
"""
from __future__ import annotations

import numpy as np

from .predictor import GMP

# Plant of the X4/X5 benchmark: 1.25 Hz resonance, Q = 1.11 — the same one
# seisim's own test notebooks build.
WN = 7.85398163397
Q  = 1.11111111


class SeisimGMP:
    """Adapter: a `GMP` inside seisim's closed-loop controller.

    Satisfies seisim's predictor contract — an `n_history` attribute plus a
    `predict(y)` method; `MPCFF.get_actuation` slices `y[-n_history:]` and
    applies `-forecast[:n_control]`. The plant must run at the model's own
    sampling rate and the control horizon equals the prediction horizon —
    seamed stitching, i.e. benchmark mode.
    """
    def __init__(self, gmp: GMP):
        self.gmp = gmp
        self.n_history = int(gmp.input_window)
        self.n_predict = int(gmp.output_window)

    def run(self, data_in):
        return self.predict(data_in)

    def predict(self, y):
        x = np.asarray(y, dtype=np.float32)[-self.n_history:].reshape(-1, 1)
        return self.gmp.predict(x).reshape(-1).astype(np.float64)


def make_plant(wn: float = WN, q: float = Q):
    """The benchmark plant as a `control` transfer function."""
    import control
    s = control.tf("s")
    return wn ** 2 / (s ** 2 + wn / q * s + wn ** 2)


def run_closed_loop(gmp: GMP, ground, tf_sys=None) -> dict:
    """Simulate feedforward isolation driven by `gmp` and score it.

    ground : 1-D ground motion already at `gmp.sampling_rate` (use
             `gmp.prepare`), in the same units/scale as the training data.

    Returns the seisim benchmark metrics (BLS, ABLS, SR, MS, rms_ground,
    rms_platform).
    """
    from seisim.benchmark  import Benchmark
    from seisim.controller import MPCFF
    from seisim.seisim     import Seisim
    from seisim.system     import System

    ground = np.asarray(ground, dtype=np.float64).ravel()
    dt = 1.0 / float(gmp.sampling_rate)
    t = np.arange(len(ground)) * dt

    pred = SeisimGMP(gmp)
    ctrl = MPCFF(model=None, n_predict=pred.n_predict,
                 n_control=pred.n_predict, predictor=pred)
    sim = Seisim(system=System(tf_sys if tf_sys is not None else make_plant(), dt),
                 controller=ctrl)
    platform = sim.simulate(t, ground)
    bench = Benchmark(t, ground, platform).run()
    return {"BLS":  bench["band_limited_suppression"],
            "ABLS": bench["auto_band_limited_suppression"],
            "SR":   bench["suppression_ratio"],
            "MS":   bench["maximum_suppression"],
            "rms_ground":   bench["rms_ground"],
            "rms_platform": bench["rms_platform"]}
