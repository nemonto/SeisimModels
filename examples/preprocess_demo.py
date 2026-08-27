"""Raw recording -> model input -> forecast, on a real record.

Uses seisim's own 1 h test file (64 Hz, two ASCII rows: time, amplitude):

    python examples/preprocess_demo.py --data H:/seisim-master/test_data/itmx_z_1441947618_64hz.txt

Any (T,) or (T, C) array works — pass its rate with --fs-in. The bandpass and
the decimation factor come from the checkpoint itself, not from this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmp import find, load_gmp, prepare, recipe        # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path,
                   help="two-row ASCII file (time row, amplitude row)")
    p.add_argument("--fs-in", type=float, default=None,
                   help="sample rate of --data (default: inferred from the time row)")
    p.add_argument("--model", default="gru")
    p.add_argument("--sub-type", default="all")
    p.add_argument("--output-sec", type=float, default=0.5, choices=[0.5, 1.0])
    args = p.parse_args()

    t, raw = np.loadtxt(args.data)
    fs_in = args.fs_in if args.fs_in else round(1.0 / (t[1] - t[0]))
    print(f"raw       : {raw.shape[0]} samples at {fs_in:g} Hz "
          f"= {raw.shape[0] / fs_in:g} s,  rms = {np.sqrt((raw ** 2).mean()):.4g}")

    ckpt = find(args.model, sub_type=args.sub_type, output_sec=args.output_sec)
    print(f"recipe    : {recipe(ckpt)}")

    sig = prepare(raw, ckpt, fs_in=int(fs_in))
    gmp = load_gmp(ckpt)
    print(f"prepared  : {sig.shape} at {gmp.sampling_rate} Hz,  "
          f"rms = {np.sqrt((sig ** 2).mean()):.4g}")

    pred = gmp.predict_long(sig, mode="seamed")
    target = sig[gmp.input_window:]
    rmse = float(np.sqrt(((pred - target) ** 2).mean()))
    rms_t = float(np.sqrt((target ** 2).mean()))
    print(f"forecast  : {pred.shape}, aligned to prepared[{gmp.input_window}:]")
    print(f"open-loop : RMSE = {rmse:.4g}   (target RMS = {rms_t:.4g}, "
          f"ratio = {rmse / rms_t:.3f})")
    print("\nA ratio near or above 1 usually means the record is scaled "
          "differently from the training data — the checkpoint's scaler is "
          "fixed, so units matter (see README, 'Units and scale').")


if __name__ == "__main__":
    main()
