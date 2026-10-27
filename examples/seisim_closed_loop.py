"""Drive the seisim closed-loop simulator with a GMP forecast.

Run it from the seisim checkout, so `import seisim` resolves:

    cd G:\\seisim-master
    python G:\\SeisimModels\\examples\\seisim_closed_loop.py \\
        --data test_data/itmx_z_1441947618_64hz.txt --model gru --output-sec 2

Prints seisim's benchmark metrics on the controlled interval (from 64 s):
BLS / ABLS / SR / MS plus the ground and platform RMS. Same plant,
controller, control start and scoring as the GMP_X6/X8 benchmark, so on a
test record the numbers are the BLS column of `checkpoints/model_index.csv`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmp import FAMILIES, GMP, Persistence, Wiener, find, prepare   # noqa: E402
from gmp.seisim_adapter import run_closed_loop                       # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path,
                   help="two-row ASCII ground motion (time row, amplitude row)")
    p.add_argument("--fs-in", type=float, default=None)
    p.add_argument("--model", default="gru",
                   choices=list(FAMILIES) + ["wiener", "persistence"])
    p.add_argument("--output-sec", type=float, default=2.0, choices=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--seed", type=int, default=0, choices=[0, 1, 2])
    args = p.parse_args()

    if args.model == "wiener":
        model = Wiener(find("wiener", input_sec=20), args.output_sec)
    elif args.model == "persistence":
        model = Persistence(args.output_sec)
    else:
        model = GMP(find(args.model, output_sec=args.output_sec, seed=args.seed))
    print(f"model  : {model.model_type}  K={model.input_window} H={model.output_window} "
          f"fs={model.sampling_rate} Hz")

    t, raw = np.loadtxt(args.data)
    fs_in = int(args.fs_in if args.fs_in else round(1.0 / (t[1] - t[0])))
    ground, _ = prepare(raw, fs_in=fs_in)
    print(f"ground : {len(ground)} samples at 8 Hz (from {fs_in} Hz), "
          f"rms = {np.sqrt((ground ** 2).mean()):.4g}")

    metrics = run_closed_loop(model, ground)
    print()
    for k, v in metrics.items():
        print(f"  {k:<13s} {v:.6g}")


if __name__ == "__main__":
    main()
