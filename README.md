# SeisimModels

Pretrained ground-motion predictors for seismic-isolation feedforward control,
packaged for inference: the 60 deep-model checkpoints of the GMP_X6/X8
benchmark (the CS paper), its classical baselines, and an adapter that drives
the [seisim](../seisim-master) closed-loop simulator exactly as the benchmark
does.

No training code, no datasets, no hyperparameter search — that lives in
GMP_X6 / GMP_X8_CS. This is the hand-off: copy the folder, install an env,
predict. The earlier GMP_X5 / X3 release is git tag `v1.0-x5`.

```python
from gmp import GMP, find, prepare

ground, x = prepare(raw_512hz)                  # 512 -> 8 Hz; x = seisim's g_filt
model = GMP(find("gru", output_sec=2))          # 2 s ahead, seed 0
pred  = model.predict_long(x)                   # (T-K, 1), aligned to x[K:]
```

---

## What's in the box

| | count | |
|---|---|---|
| Deep models | 60 | 5 families x 4 horizons (0.5, 1, 2, 4 s) x 3 training seeds |
| Wiener filter | 2 fit files | 5 s and 20 s input, every horizon in each file |
| Persistence | code | last sample held |
| `checkpoints/model_index.csv` | 416 rows | the benchmark's test scores of every method (below) |

Families and their input length (picked by the X6 hyperparameter search):
`dlinear`, `gru`, `lstm` 2.5 s; `moderntcn`, `patch-tst` 10 s. Every model is
single-channel, 8 Hz, trained on the pooled data of all four conditions
(summer, winter, earthquake that broke lock, earthquake that held lock),
absolute target. ~64 MB of checkpoints.

```
SeisimModels/
  gmp/                       the package  (import gmp)
    predictor.py             the GMP class — load, predict, stitch
    _models.py  _blocks.py   architectures
    classical.py             Wiener, Persistence (same interface as GMP)
    preprocess.py            the signal path: decimate, band (seisim's g_filt)
    benchmark.py             the benchmark's forecast stream and RMSP
    paths.py                 checkpoint discovery, find()
    seisim_adapter.py        SeisimGMP, make_plant, run_closed_loop
  checkpoints/               60 x unified_checkpoint.ckpt, classical/, model_index.csv
  examples/                  quickstart, preprocess_demo, seisim_closed_loop
  tests/test_predictor.py    checks every shipped model
```

`gmp/predictor.py`, `_models.py`, `_blocks.py`, the checkpoints and the index
are written by GMP_X8_CS `py_exp2/export_seisim_models.py`; don't edit them
here.

---

## Install

Python >= 3.10 (3.11 verified). Pick one route. All three end with the project
installed **editable** — the checkpoints are repo data resolved relative to the
package, so a non-editable install leaves them behind (override with
`SEISIM_MODELS_ROOT` if you must).

### conda

```bash
cd G:/SeisimModels
conda env create -f environment.yml
conda activate seisim-models
```

### uv

```bash
cd G:/SeisimModels
uv sync --extra seisim
uv run python examples/quickstart.py
```

### plain pip

```bash
cd G:/SeisimModels
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[seisim,index]"
```

### About torch

CPU is plenty — the largest checkpoint is a few MB and `predict_long` batches
windows internally. CUDA wheels are **not** on PyPI: for a GPU box install
torch first from `--index-url https://download.pytorch.org/whl/cu129`, then
`pip install -e .`. `GMP` picks CUDA automatically when it is available;
pass `device="cpu"` to override. Inference always runs in full float32 with
TF32 off (restored afterwards), so CPU and GPU give the same forecasts: the
cuDNN TF32 default can move a forecast by several percent.

### Check the install

```bash
python tests/test_predictor.py        # every checkpoint + Wiener/Persistence at every horizon
```

The last line should read `72 models, 0 failures`.

---

## Quickstart

```python
from gmp import GMP, Wiener, Persistence, find

model = GMP(find("gru", output_sec=2, seed=0))
# -> checkpoints/gru/w2.5x2_s0/unified_checkpoint.ckpt
model.model_type      # 'gru'
model.input_window    # 20 samples = 2.5 s at 8 Hz   (K)
model.output_window   # 16 samples = 2 s at 8 Hz     (H)
model.sampling_rate   # 8

y  = model.predict(x)                        # (K, C) -> (H, C);  (B, K, C) -> (B, H, C)
yl = model.predict_long(sig, mode="seamed")  # (T,) or (T, C) -> (T-K, C), aligned to sig[K:]

wiener = Wiener(find("wiener", input_sec=20), output_sec=2)   # same interface
persist = Persistence(output_sec=2)
```

Everything needed for inference lives inside the checkpoint, so construction
takes only a path. `predict` needs the trailing axis pair `(K, C)`;
`predict_long` promotes `(T,)` to `(T, 1)`, and element *i* of its output is
the forecast of `signal[K+i]`.

### Stitching modes

| mode | stride | what it does |
|---|---|---|
| `seamed` (default) | H | hard concatenation; each output sample comes from exactly one forecast — what seisim does. Broadband seam noise above the signal band. |
| `overlap_add` | H/2 | Hann cross-fade between overlapping forecasts. Smoother, but averaging two imperfectly-phased forecasts shrinks passband amplitude. |

Use `seamed` for anything that drives control.

---

## Preprocessing: what the models expect

**`GMP` does no filtering, decimation or resampling.** Every model was trained
on the signal seisim hands its predictor:

1. **ground** — the raw 512 Hz velocity decimated to 8 Hz with
   `scipy.signal.decimate` in q = 8 steps (zero-phase IIR, as the seisim
   notebooks do), mean removed. This is what you pass to `Seisim.simulate`.
2. **input** — seisim's prediction band applied to the ground: 4th-order
   Butterworth 0.05-0.5 Hz, zero-phase (`sosfiltfilt`). Bit-identical to the
   `g_filt` that `Seisim.simulate` computes internally.

```python
from gmp import prepare, recipe

ground, x = prepare(raw)                 # raw at 512 Hz
ground, x = prepare(raw64, fs_in=64)     # a 64 Hz record instead
recipe(find("gru"))  # {'fs_in': 512, 'fs': 8, 'protocol': 'zero_phase', 'low': 0.05, 'high': 0.5, 'order': 4}
```

`fs_in / 8` must be a whole number. Give it at least a few minutes of data
(`sosfiltfilt` padding). The input is the **zero-phase** (noncausal) band, as
in seisim: an offline setting. A real-time causal filter would give every
predictor a harder input.

### Units and scale

The scaler is baked into each checkpoint. Your data must be ground velocity
in **nm/s**, like the training data, or the forecast degrades **without
raising anything**. The `rms_ground` column of `model_index.csv` is the
reference (per condition, at 8 Hz after decimation).

---

## Choosing a model

```
checkpoints/<family>/w<input_sec>x<output_sec>_s<seed>/unified_checkpoint.ckpt
checkpoints/classical/wiener_w<input_sec>.npz
```

```python
find("patch-tst", output_sec=1, seed=2)
find("wiener", input_sec=5)
```

### Results (the benchmark's test set)

Macro RMSP (lower is better; per condition the mean over its test files,
then the mean over the four conditions), mean over the 3 seeds, in seisim's
closed loop with control from 64 s:

| Method | 0.5 s | 1 s | 2 s | 4 s |
|---|---|---|---|---|
| GRU | 0.00118 | 0.0119 | 0.144 | 0.434 |
| LSTM | 0.00200 | 0.0150 | 0.156 | 0.456 |
| PatchTST | 0.00926 | 0.0531 | 0.237 | 0.490 |
| ModernTCN | 0.00910 | 0.0523 | 0.239 | 0.476 |
| DLinear | 0.0241 | 0.139 | 0.487 | 0.818 |
| Wiener (20 s) | 0.0138 | 0.0740 | 0.341 | 0.650 |
| Wiener (5 s) | 0.0146 | 0.0799 | 0.392 | 0.720 |
| ConstantOrder (o7, seisim) | 0.00149 | 0.0749 | 4.41 | 390 |
| Persistence | 0.463 | 0.779 | 1.30 | 1.23 |
| No actuation | 1 | 1 | 1 | 1 |

These are exploratory: the test files informed the models' development.

### The model index

`checkpoints/model_index.csv` — one row per (method, condition): the 60
checkpoints, Wiener 5/20 s, persistence, ConstantOrder o2-o7, no actuation
(`zero`) and the filtered-target reference (`oracle`), at every horizon:

`model_type, variant, window, seed, testset_id, RMSE, RMSP, BLS, ABLS, SR,
MS, rms_ground, rms_platform, n_segments, path`

`window` is `"<input s>:<output s>"`; `seed` is -1 for deterministic methods;
`RMSE`/`RMSP` are open-loop, `BLS`/`ABLS`/`SR`/`MS` closed-loop suppression
(higher is better), all on the controlled interval; `path` is the shipped
file, relative to this project (`-` when nothing is shipped).

```python
import pandas as pd
from gmp import INDEX_CSV, GMP, resolve

idx = pd.read_csv(INDEX_CSV)
win = idx[(idx.testset_id == "winter") & idx.window.str.endswith(":2")]
print(win.sort_values("BLS", ascending=False).head(5)[["model_type", "seed", "RMSP", "BLS", "path"]])
```

### Reproducing a score

`gmp.benchmark` replays the benchmark's grid: the first forecast at the
control start (64 s), then every H samples.

```python
from gmp import CONTROL_START_SEC, forecast_stream, rmsp
stream = forecast_stream(model, x)                # aligned to x[512:]
rmsp(stream, x[int(CONTROL_START_SEC * 8):], 8)
```

---

## Using it with seisim

seisim's predictor contract is small: an `n_history` attribute and a
`predict(y)` method. `MPCFF.get_actuation` slices `seismic[-n_history:]`, calls
the predictor, and applies `-forecast[:n_control]`. `gmp.seisim_adapter`
implements it the way the benchmark does: `n_history` = 64 s (no actuation
before the control start), the model reads the last `input_window` samples,
control horizon = prediction horizon.

**Pass seisim the ground, not the band.** `Seisim.simulate` band-passes the
ground itself and hands the predictor `g_filt`, the models' training input.

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "SeisimModels"))   # if not installed

import numpy as np
from gmp import GMP, find, prepare
from gmp.seisim_adapter import run_closed_loop

model = GMP(find("gru", output_sec=2))
t64, g64 = np.loadtxt("test_data/itmx_z_1441947618_64hz.txt")
ground, _ = prepare(g64, fs_in=64)
print(run_closed_loop(model, ground))     # BLS, ABLS, SR, MS on [64 s, end)
```

Or build it yourself with `SeisimGMP(model)`, `MPCFF(model=None,
n_predict=H, n_control=H, predictor=...)` and `System(make_plant(), 1/8)`.

Things that will bite you:

- **The plant must run at the model's rate**: `dt = 1/8` s. seisim's own
  notebooks run at 64 Hz; decimate the ground (`prepare`) rather than
  changing `dt`.
- **`test_seisim.ipynb` rebinds the module name** in its last cell
  (`seisim = seisim.seisim.Seisim(...)`), after which `seisim.predictor` no
  longer resolves. Do your imports before that cell, or use
  `from seisim.seisim import Seisim`.
- **`seisim.predictor` is not this package.** It is seisim's own polynomial
  extrapolator (`ConstantOrder`, the paper's third baseline). This package
  is top-level `gmp` so the two never collide.

`make_plant()` is the benchmark plant — 1.25 Hz resonance, Q = 1.11 — the same
transfer function seisim's own test notebooks build.

Runnable version: `examples/seisim_closed_loop.py`, run with cwd =
`G:\seisim-master` so `import seisim` resolves.
