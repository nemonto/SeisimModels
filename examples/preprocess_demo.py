"""Raw recording -> model input -> forecast -> RMSP, on a real record.

Uses seisim's own 1 h test file (64 Hz, two ASCII rows: time, amplitude):

    python examples/preprocess_demo.py --data G:/seisim-master/test_data/itmx_z_1441947618_64hz.txt

Any 1-D record works — pass its rate with --fs-in. The models were trained
on LIGO ground velocity in nm/s (see README, "Units and scale"); a record in
other units gives a poor forecast without raising anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmp import (CONTROL_START_SEC, GMP, Persistence, Wiener, find,   # noqa: E402
                 forecast_stream, prepare, recipe, rmsp)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path,
                   help="two-row ASCII file (time row, amplitude row)")
    p.add_argument("--fs-in", type=float, default=None,
                   help="sample rate of --data (default: inferred from the time row)")
    p.add_argument("--output-sec", type=float, default=2.0, choices=[0.5, 1.0, 2.0, 4.0])
    args = p.parse_args()

    t, raw = np.loadtxt(args.data)
    fs_in = int(args.fs_in if args.fs_in else round(1.0 / (t[1] - t[0])))
    print(f"raw      : {len(raw)} samples at {fs_in} Hz = {len(raw) / fs_in:g} s, "
          f"rms = {np.sqrt((raw ** 2).mean()):.4g}")
    print(f"recipe   : {recipe(find('gru', output_sec=args.output_sec))}")

    ground, x = prepare(raw, fs_in=fs_in)
    print(f"prepared : ground and input, {len(x)} samples at 8 Hz, "
          f"input rms = {np.sqrt((x ** 2).mean()):.4g}")

    i0 = int(CONTROL_START_SEC * 8)
    models = {fam: GMP(find(fam, output_sec=args.output_sec))
              for fam in ("gru", "lstm", "moderntcn", "patch-tst", "dlinear")}
    models["wiener-20s"] = Wiener(find("wiener", input_sec=20), args.output_sec)
    models["persistence"] = Persistence(args.output_sec)
    print(f"\nRMSP at {args.output_sec:g} s ahead, scored from {CONTROL_START_SEC:g} s "
          "(lower is better; 1 = no better than predicting zero):")
    for name, m in models.items():
        print(f"  {name:<12s} {rmsp(forecast_stream(m, x), x[i0:], 8):.4g}")


if __name__ == "__main__":
    main()
