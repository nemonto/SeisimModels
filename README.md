# SeisimModels

Pretrained ground-motion predictors for seismic-isolation feedforward control,
packaged for inference. 107 checkpoints exported from the GMP_X5 exp2
benchmark, plus the self-contained `gmp` package that loads and runs them, plus
an adapter that drives the [seisim](../seisim-master) closed-loop simulator.

No training code, no datasets, no hyperparameter search — that all lives in
`GMP_X5`. This is the hand-off: copy the folder, install an env, predict.

```python
from gmp import GMP, find, prepare

ckpt  = find("gru", sub_type="all", output_sec=0.5, target="absolute")
model = GMP(ckpt)
sig   = prepare(raw_512hz, ckpt)          # bandpass 0.06-0.6 Hz + 512 -> 8 Hz
pred  = model.predict_long(sig)           # (T-K, 1), aligned to sig[K:]
```

---

## What's in the box

| | count | |
|---|---|---|
| GMP_X5 models | 100 | 5 families x 5 training conditions x 2 horizons x 2 target modes |
| Legacy GMP_X3 models | 5 | LSTM, GRU, DLinear, PatchTST, CNN-LSTM — 16 Hz, 0.5 s horizon |
| Persistence baselines | 2 | one per horizon; the floor any model must beat |

Families: `dlinear`, `gru`, `lstm`, `moderntcn`, `patch-tst`. Everything is
single-channel, trained on 0.06-0.6 Hz bandpassed vertical ground motion at
8 Hz. ~51 MB of checkpoints in total.

```
SeisimModels/
  gmp/                       the package  (import gmp)
    predictor.py             the GMP class — load, predict, stitch
    _models.py  _blocks.py   architectures (ports of GMP_X5 py_models/)
    legacy.py                GMP_X3 schema + architectures, load_gmp()
    preprocess.py            bandpass / decimate, straight from the training code
    paths.py                 checkpoint discovery, find()
    seisim_adapter.py        SeisimGMP, make_plant, run_closed_loop
  checkpoints/               107 x unified_checkpoint.ckpt + model_index.csv
  examples/                  quickstart, preprocess_demo, seisim_closed_loop
  tests/test_predictor.py    smoke test over every shipped checkpoint
```

---

## Install

Python >= 3.10 (3.11 verified). Pick one route. All three end with the project
installed **editable** — the checkpoints are repo data resolved relative to the
package, so a non-editable install leaves them behind (override with
`SEISIM_MODELS_ROOT` if you must).

### conda

```bash
cd H:/SeisimModels
conda env create -f environment.yml
conda activate seisim-models
```

### uv

```bash
cd H:/SeisimModels
uv sync --extra seisim
uv run python examples/quickstart.py
```

### plain pip

```bash
cd H:/SeisimModels
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[seisim,index]"
```

### About torch

CPU is plenty — the biggest checkpoint is 1.9 MB and `predict_long` batches
windows internally. CUDA wheels are **not** on PyPI: for a GPU box install
torch first from `--index-url https://download.pytorch.org/whl/cu129`, then
`pip install -e .`. `GMP` picks CUDA automatically when it is available;
pass `device="cpu"` to override.

### Check the install

```bash
python tests/test_predictor.py        # walks all 107 checkpoints
```

Last line should read `107 checkpoints, 0 failures`. A few minutes on CPU.

---

## Quickstart

```python
from gmp import GMP, load_gmp, find

ckpt = find("gru", sub_type="all", output_sec=0.5, target="absolute")
# -> checkpoints/gru/all/w5x0.5_absolute/unified_checkpoint.ckpt
gmp = GMP(ckpt)                    # or load_gmp(ckpt) — see "Legacy models"

gmp.model_type      # 'gru'
gmp.input_window    # 40 samples  = 5 s at 8 Hz      (K)
gmp.output_window   # 4 samples   = 0.5 s at 8 Hz    (H)
gmp.sampling_rate   # 8
gmp.target_mode     # 'absolute'

y  = gmp.predict(x)                        # (K, C) -> (H, C);  (B, K, C) -> (B, H, C)
yl = gmp.predict_long(sig, mode="seamed")  # (T, C), T >= K+H -> (T-K, C)
```

Everything needed for inference lives inside the checkpoint, so construction
takes only a path — no config, no model class, no window arguments.

- `predict` needs the trailing axis pair `(K, C)` and **rejects bare 1-D**.
- `predict_long` promotes `(T,)` to `(T, 1)` and returns a forecast aligned to
  `signal[K:]` — element *i* of the output is the model's forecast of
  `signal[K+i]`.
- **Delta checkpoints are integrated internally.** `target_mode` describes how
  the model was trained; the return value is always an absolute-signal forecast
  in the original units. To a caller, `absolute` and `delta` differ only in
  accuracy.

### Stitching modes

| mode | stride | what it does |
|---|---|---|
| `seamed` (default) | H | hard concatenation; each output sample comes from exactly one forecast. Broadband seam noise above the signal band. |
| `overlap_add` | H/2 | Hann cross-fade between overlapping forecasts. Smoother, but averaging two imperfectly-phased forecasts shrinks passband amplitude. |

Use `seamed` for anything that drives control. GMP_X4 measured `overlap_add`
scoring *better* open-loop (RMSE) and markedly *worse* closed-loop (BLS, ABLS,
SR, MS).

---

## Preprocessing: what the models expect

**`GMP` does no filtering, decimation or resampling.** Its `sampling_rate` is
metadata. Every model was trained on ground motion that was:

1. zero-phase Butterworth bandpassed **0.06-0.6 Hz**, 4th order (`sosfiltfilt`), then
2. decimated by polyphase FIR to **8 Hz** (16 Hz for the legacy models) from a
   512 Hz raw recording.

Each checkpoint carries its own recipe, so you never hardcode those numbers:

```python
from gmp import recipe, prepare

recipe(ckpt)                 # {'fs_in': 512, 'fs_out': 8, 'low': 0.06, 'high': 0.6, 'order': 4}
sig = prepare(raw, ckpt)                 # raw is 512 Hz
sig = prepare(raw64, ckpt, fs_in=64)     # a 64 Hz record instead
```

`prepare` returns `(T, 1)` float32 at the model's rate. `fs_in / fs_out` must
be a whole number; resample first if it isn't. Very short records will trip
`sosfiltfilt`'s padding requirement (roughly `3*(2*order+1)` samples of margin)
— give it at least a few minutes of data.

The lower-level `bandpass`, `decimate_to` and `bandpass_decimate` are exported
too; they are lifted verbatim from the GMP_X5 training code so there is one
recipe, not two.

### Units and scale

The z-score scaler is baked into each checkpoint. Your data must be in the
same physical units and on the same scale as the training data, or the forecast
degrades **without raising anything**. The `rms_ground` column of
`checkpoints/model_index.csv` is the reference: roughly 180 for the summer
split, up to ~4800 for winter. If your record's RMS is orders of magnitude off
that, rescale it before predicting.

---

## Choosing a checkpoint

```
checkpoints/<family>/<sub_type>/w<input_sec>x<output_sec>_<target>/unified_checkpoint.ckpt
checkpoints/naive/w5x<output_sec>_absolute/unified_checkpoint.ckpt
checkpoints/legacy/<family>/unified_checkpoint.ckpt
```

| axis | values | meaning |
|---|---|---|
| family | `dlinear` `gru` `lstm` `moderntcn` `patch-tst` | architecture |
| sub_type | `summer` `winter` `eql` `eqh` `all` | **training** data: Aug 2024 days / Jan 2024 days / earthquakes that broke lock / earthquakes that held lock / the union of all four |
| window | `w5x0.5` `w5x1` | 5 s of input, 0.5 s or 1 s of forecast |
| target | `absolute` `delta` | training target; output is absolute either way |

`find()` builds these paths for you:

```python
find("patch-tst", sub_type="winter", output_sec=1.0, target="delta")
find("naive", output_sec=0.5)                   # persistence baseline
find("cnn-lstm", sub_type="legacy")             # GMP_X3
```

### The model index

`checkpoints/model_index.csv` — 428 rows, one per (checkpoint, test
condition), straight from the exp2 benchmark:

`model_type, sub_type, window, target, testset_id, RMSE, RMSP, BLS, ABLS, SR,
MS, rms_ground, rms_platform, fs, n_segments, ckpt_path`

`RMSE`/`RMSP` are open-loop; `BLS`/`ABLS`/`SR`/`MS` are closed-loop
suppression (higher is better). `ckpt_path` is POSIX-relative to this project.

```python
import pandas as pd
from gmp import INDEX_CSV, ROOT, load_gmp

idx = pd.read_csv(INDEX_CSV)
best = (idx[idx.testset_id == "winter"]
        .sort_values("BLS", ascending=False)
        .head(5)[["model_type", "sub_type", "window", "target", "BLS", "ckpt_path"]])
print(best)

gmp = load_gmp(ROOT / best.iloc[0].ckpt_path)
```

### Legacy models

The five checkpoints under `checkpoints/legacy/` come from GMP_X3: 16 Hz,
0.5 s horizon, absolute only, and an older checkpoint schema that bare `GMP`
cannot read. Use `load_gmp`, which sniffs the schema and returns `GMP` or
`LegacyGMP` — identical interface either way.

```python
from gmp import load_gmp
gmp = load_gmp("checkpoints/legacy/patch-tst/unified_checkpoint.ckpt")
gmp.sampling_rate    # 16
gmp.input_window     # 720 = 45 s
```

`load_gmp` works for every shipped checkpoint, so prefer it whenever the path
is data rather than a literal.

---

## Using it with seisim

seisim's predictor contract is small: an `n_history` attribute and a
`predict(y)` method. `MPCFF.get_actuation` slices `seismic[-n_history:]`, calls
the predictor, and applies `-forecast[:n_control]`. `gmp.seisim_adapter`
implements it.

Two ways to reach this package from a seisim notebook (its kernel starts in
`H:\seisim-master`):

```python
# A — installed (preferred): nothing to do, just import
from gmp import find, load_gmp, prepare

# B — not installed: SeisimModels is a sibling of seisim-master
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "SeisimModels"))
from gmp import find, load_gmp, prepare
```

Then swap `ConstantOrder` for the GMP adapter — this is the existing
`test_seisim.ipynb` cell with three lines changed:

```python
import numpy as np
from seisim.benchmark  import Benchmark
from seisim.controller import MPCFF
from seisim.seisim     import Seisim
from seisim.system     import System
from gmp import find, load_gmp, prepare
from gmp.seisim_adapter import SeisimGMP, make_plant

ckpt = find("gru", sub_type="all", output_sec=0.5, target="absolute")
gmp  = load_gmp(ckpt)

t64, g64 = np.loadtxt("test_data/itmx_z_1441947618_64hz.txt")
ground = prepare(g64, ckpt, fs_in=64).ravel()       # 64 Hz -> 8 Hz, 0.06-0.6 Hz
dt = 1.0 / gmp.sampling_rate
t  = np.arange(len(ground)) * dt

pred = SeisimGMP(gmp)
ctrl = MPCFF(model=None, n_predict=pred.n_predict,
             n_control=pred.n_predict, predictor=pred)
sim  = Seisim(system=System(make_plant(), dt), controller=ctrl)
platform = sim.simulate(t, ground)
print(Benchmark(t, ground, platform).run())
```

Or in one call: `gmp.seisim_adapter.run_closed_loop(gmp, ground)`.

Three things that will bite you:

- **The plant must run at the model's rate.** `dt = 1/gmp.sampling_rate`, i.e.
  0.125 s for X5 models and 0.0625 s for legacy. seisim's own notebooks run at
  64 Hz; resample the ground motion (that is what `prepare` is for) rather
  than changing `dt`.
- **`test_seisim.ipynb` rebinds the module name** in its last cell
  (`seisim = seisim.seisim.Seisim(...)`), after which `seisim.predictor` no
  longer resolves. Do your imports before that cell, or use
  `from seisim.seisim import Seisim`.
- **`seisim.predictor` is not this package.** It is seisim's own polynomial
  extrapolator (`ConstantOrder`). This package is top-level `gmp` precisely so
  the two never collide.

`make_plant()` is the benchmark plant — 1.25 Hz resonance, Q = 1.11 — the same
transfer function seisim's own test notebooks build.

Runnable version: `examples/seisim_closed_loop.py`, run with cwd =
`H:\seisim-master` so `import seisim` resolves.
