"""Load a model and forecast — the 30-second tour. Needs no data.

Run::
    python examples/quickstart.py
    python examples/quickstart.py --model patch-tst --output-sec 1 --seed 2
    python examples/quickstart.py --model wiener --output-sec 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmp import FAMILIES, GMP, HORIZONS, Persistence, Wiener, find   # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gru",
                   choices=list(FAMILIES) + ["wiener", "persistence"])
    p.add_argument("--output-sec", type=float, default=2.0, choices=list(HORIZONS))
    p.add_argument("--seed", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--wiener-input-sec", type=float, default=20, choices=[5, 20])
    p.add_argument("--device", default=None, help='"cpu" to force CPU')
    args = p.parse_args()

    if args.model == "wiener":
        model = Wiener(find("wiener", input_sec=args.wiener_input_sec), args.output_sec)
    elif args.model == "persistence":
        model = Persistence(args.output_sec)
    else:
        ckpt = find(args.model, output_sec=args.output_sec, seed=args.seed)
        print(f"checkpoint : {ckpt.relative_to(Path(__file__).resolve().parents[1]).as_posix()}")
        model = GMP(ckpt, device=args.device)
    K, H, fs = model.input_window, model.output_window, model.sampling_rate
    print(f"model_type : {model.model_type}")
    print(f"window     : {K} in / {H} out samples = {K / fs:g} s / {H / fs:g} s at {fs} Hz")
    print(f"device     : {model.device}")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((K, 1))
    print(f"\npredict     : {x.shape} -> {model.predict(x).shape}")
    xb = rng.standard_normal((4, K, 1))
    print(f"predict (B) : {xb.shape} -> {model.predict(xb).shape}")
    sig = rng.standard_normal(K + 200 * H)
    for mode in ("seamed", "overlap_add"):
        out = model.predict_long(sig, mode=mode)
        print(f"predict_long ({mode:<11s}): {sig.shape} -> {out.shape} "
              f"(aligned to signal[{K}:])")

    print("\nReal data: decimate to 8 Hz and band-pass 0.05-0.5 Hz first — "
          "`gmp.prepare` and examples/preprocess_demo.py.")


if __name__ == "__main__":
    main()
