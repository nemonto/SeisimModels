"""Verify every shipped model loads, was trained on the package's signal path
(`gmp.preprocess`), predicts a single window correctly, produces NaN-free
length-matched outputs in both stitching modes, and stitches `seamed`
forecasts at the right timestamps (each output sample against a direct
`predict` of its own window, including a partial final bucket and a stream
shorter than one horizon). The checks mirror GMP_X8_CS
`predictor/test_predictor.py`. Covers the 60 checkpoints and the Wiener /
Persistence baselines at every horizon. Expect 0 failures.

Run::
    python tests/test_predictor.py
    conda run -n seisim-models python tests/test_predictor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmp import (BAND, GMP, HORIZONS, WIENER_INPUTS, Persistence, Wiener,   # noqa: E402
                 find, list_checkpoints, recipe)
from gmp.paths import CHECKPOINTS   # noqa: E402


def _seamed_reference(model, sig: np.ndarray) -> np.ndarray:
    """Index-by-index reference for `seamed`: output sample p comes from the
    forecast of sig[p0 : p0+K], p0 = p - p % H."""
    K, H = model.input_window, model.output_window
    out_len = len(sig) - K
    ref = np.empty((out_len, sig.shape[1]), dtype=model.input_dtype)
    for p0 in range(0, out_len, H):
        n = min(H, out_len - p0)
        ref[p0:p0 + n] = model.predict(sig[p0:p0 + K])[:n]
    return ref


def _check(model) -> int:
    K = model.input_window; H = model.output_window
    rng = np.random.default_rng(0)
    failures = 0

    x = rng.standard_normal((K, 1))
    y = model.predict(x)
    assert y.shape == (H, 1), f"single-window shape: {y.shape}"
    assert y.dtype == model.input_dtype, f"forecast dtype {y.dtype}"
    assert np.isfinite(y).all(), "NaNs/Infs in single-window prediction"

    T = K + H * 7 + 3       # not divisible by H to exercise tail handling
    sig = rng.standard_normal((T, 1))
    for mode in ("seamed", "overlap_add"):
        out = model.predict_long(sig, mode=mode)
        ok = out.shape == (T - K, 1) and bool(np.isfinite(out).all())
        failures += not ok
        print(f"    {mode:<12s} shape={out.shape}  {'ok' if ok else 'FAIL'}")

    for name, T_ in (("partial tail", K + H * 7 + 3),
                     ("divisible", K + H * 7),
                     ("T - K < H", K + max(1, H // 2))):
        s = rng.standard_normal((T_, 1))
        out = model.predict_long(s, mode="seamed")
        ref = _seamed_reference(model, s)
        err = float(np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-12))
        ok = out.shape == ref.shape and err < 1e-4
        failures += not ok
        print(f"    seamed alignment ({name:<12s}) rel err={err:.1e}  {'ok' if ok else 'FAIL'}")
    return failures


def main() -> int:
    ckpts = list_checkpoints()
    if not ckpts:
        print(f"NO CHECKS PERFORMED: no checkpoints under {CHECKPOINTS}")
        return 2
    fails, n = 0, 0
    for cp in ckpts:
        print(f"\n{cp.relative_to(CHECKPOINTS).as_posix()}")
        r = recipe(cp)
        band_ok = (r["protocol"] == "zero_phase" and r["fs"] == 8
                   and all(r[k] == BAND[k] for k in BAND))
        fails += not band_ok
        print(f"    recipe {r}  {'ok' if band_ok else 'FAIL'}")
        m = GMP(cp)
        print(f"    model_type={m.model_type}  K={m.input_window}  H={m.output_window}")
        fails += _check(m); n += 1
    for h in HORIZONS:
        for k in WIENER_INPUTS:
            m = Wiener(find("wiener", input_sec=k), h)
            print(f"\nwiener {k:g} s, {h:g} s  K={m.input_window}  H={m.output_window}  "
                  f"ridge={m.ridge:g}")
            fails += _check(m); n += 1
        m = Persistence(h)
        print(f"\npersistence {h:g} s  H={m.output_window}")
        fails += _check(m); n += 1
    print(f"\n{n} models, {fails} failures")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
