"""Load a checkpoint and forecast — the 30-second tour. Needs no data.

Run::
    python examples/quickstart.py
    python examples/quickstart.py --model patch-tst --sub-type winter --output-sec 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmp import find, load_gmp        # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gru", help="dlinear|gru|lstm|moderntcn|patch-tst")
    p.add_argument("--sub-type", default="all", help="summer|winter|eql|eqh|all|legacy")
    p.add_argument("--output-sec", type=float, default=0.5, choices=[0.5, 1.0])
    p.add_argument("--target", default="absolute", choices=["absolute", "delta"])
    p.add_argument("--device", default=None, help='"cpu" to force CPU')
    args = p.parse_args()

    ckpt = find(args.model, sub_type=args.sub_type,
                output_sec=args.output_sec, target=args.target)
    print(f"checkpoint : {ckpt.relative_to(Path(__file__).resolve().parents[1])}")

    gmp = load_gmp(ckpt, device=args.device)
    K, H = gmp.input_window, gmp.output_window
    print(f"model_type : {gmp.model_type}")
    print(f"window     : {K} in / {H} out samples "
          f"= {K / gmp.sampling_rate:g} s / {H / gmp.sampling_rate:g} s "
          f"at {gmp.sampling_rate} Hz")
    print(f"target_mode: {gmp.target_mode}  (output is always absolute)")
    print(f"device     : {gmp.device}")

    rng = np.random.default_rng(0)

    # One window in, one horizon out.
    x = rng.standard_normal((K, 1)).astype(np.float32)
    y = gmp.predict(x)
    print(f"\npredict     : {x.shape} -> {y.shape}")

    # A batch of windows.
    xb = rng.standard_normal((4, K, 1)).astype(np.float32)
    print(f"predict (B) : {xb.shape} -> {gmp.predict(xb).shape}")

    # A long stream: the forecast lines up with signal[K:].
    sig = rng.standard_normal((K + 200 * H, 1)).astype(np.float32)
    for mode in ("seamed", "overlap_add"):
        out = gmp.predict_long(sig, mode=mode)
        print(f"predict_long ({mode:<11s}): {sig.shape} -> {out.shape} "
              f"(aligned to signal[{K}:])")

    print("\nReal data must be bandpassed 0.06-0.6 Hz and decimated to "
          f"{gmp.sampling_rate} Hz first — see gmp.prepare and "
          "examples/preprocess_demo.py.")


if __name__ == "__main__":
    main()
