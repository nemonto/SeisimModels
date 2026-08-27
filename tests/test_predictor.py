"""Verify every shipped checkpoint loads, predicts a single window correctly,
and produces NaN-free length-matched outputs in both stitching modes.

Legacy (GMP_X3) checkpoints use an older schema, so loading goes through
`gmp.load_gmp`, which dispatches on it. Expect 107 checkpoints, 0 failures.

Run::
    python tests/test_predictor.py
    conda run -n seisim-models python tests/test_predictor.py
    uv run python tests/test_predictor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmp import GMP, load_gmp          # noqa: E402
from gmp.paths import CHECKPOINTS      # noqa: E402


def _check(model: GMP) -> int:
    K = model.input_window; H = model.output_window
    rng = np.random.default_rng(0)

    # single-window
    x = rng.standard_normal((K, 1)).astype(np.float32)
    y = model.predict(x)
    assert y.shape == (H, 1), f"single-window shape: {y.shape}"
    assert np.isfinite(y).all(), "NaNs/Infs in single-window prediction"

    # long stream
    T = K + H * 7 + 3       # not divisible by H to exercise tail handling
    sig = rng.standard_normal((T, 1)).astype(np.float32)
    expected_len = T - K
    failures = 0
    for mode in ("seamed", "overlap_add"):
        out = model.predict_long(sig, mode=mode)
        ok_shape = out.shape == (expected_len, 1)
        ok_finite = bool(np.isfinite(out).all())
        status = "ok" if ok_shape and ok_finite else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"    {mode:<12s} shape={out.shape}  finite={ok_finite}  {status}")
    return failures


def main() -> int:
    ckpts = sorted(CHECKPOINTS.rglob("unified_checkpoint.ckpt"))
    if not ckpts:
        print(f"No checkpoints found under {CHECKPOINTS} — broken install?")
        return 1
    fails = 0
    for cp in ckpts:
        rel = cp.relative_to(CHECKPOINTS)
        print(f"\n{rel.as_posix()}")
        m = load_gmp(str(cp))
        print(f"  model_type={m.model_type}  K={m.input_window}  "
              f"H={m.output_window}  fs={m.sampling_rate}  "
              f"target={m.target_mode}")
        fails += _check(m)
    print(f"\n{len(ckpts)} checkpoints, {fails} failures")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
