"""Drive the seisim closed-loop simulator with a GMP forecast.

Run it from the seisim checkout, so `import seisim` resolves:

    cd H:\\seisim-master
    python H:\\SeisimModels\\examples\\seisim_closed_loop.py \\
        --data test_data/itmx_z_1441947618_64hz.txt --model gru --sub-type all

Prints the benchmark metrics: BLS / ABLS / SR / MS plus the ground and
platform RMS. Same plant, controller and stitching as the GMP_X5 exp2
benchmark, so the numbers are comparable with `checkpoints/model_index.csv`
(not equal — that record and these units differ).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmp import find, load_gmp, prepare                    # noqa: E402
from gmp.seisim_adapter import make_plant, run_closed_loop  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path,
                   help="two-row ASCII ground motion (time row, amplitude row)")
    p.add_argument("--fs-in", type=float, default=None)
    p.add_argument("--model", default="gru")
    p.add_argument("--sub-type", default="all")
    p.add_argument("--output-sec", type=float, default=0.5, choices=[0.5, 1.0])
    p.add_argument("--target", default="absolute", choices=["absolute", "delta"])
    p.add_argument("--ckpt", type=Path, default=None,
                   help="explicit checkpoint path (overrides the selectors)")
    args = p.parse_args()

    ckpt = args.ckpt if args.ckpt else find(
        args.model, sub_type=args.sub_type,
        output_sec=args.output_sec, target=args.target)
    gmp = load_gmp(ckpt)
    print(f"model  : {gmp.model_type}  K={gmp.input_window} H={gmp.output_window} "
          f"fs={gmp.sampling_rate} Hz  target={gmp.target_mode}")

    t, raw = np.loadtxt(args.data)
    fs_in = int(args.fs_in if args.fs_in else round(1.0 / (t[1] - t[0])))
    ground = prepare(raw, ckpt, fs_in=fs_in).ravel()
    print(f"ground : {len(ground)} samples at {gmp.sampling_rate} Hz "
          f"(from {fs_in} Hz), rms = {np.sqrt((ground ** 2).mean()):.4g}")

    metrics = run_closed_loop(gmp, ground, tf_sys=make_plant())
    print()
    for k, v in metrics.items():
        print(f"  {k:<13s} {v:.6g}")


if __name__ == "__main__":
    main()
