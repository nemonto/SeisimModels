"""Where the shipped models live, and how to name one.

The project root is derived once from this file's location, so nothing in the
package (or in a notebook that imports it) ever needs an absolute path. Set
`SEISIM_MODELS_ROOT` to override — needed only if the package was installed
non-editable and the `checkpoints/` tree stayed behind.

Layout (written by GMP_X8_CS `py_exp2/export_seisim_models.py`):
    checkpoints/<family>/w<input_sec>x<output_sec>_s<seed>/unified_checkpoint.ckpt
    checkpoints/classical/wiener_w<input_sec>.npz
    checkpoints/model_index.csv
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT        = Path(os.environ.get("SEISIM_MODELS_ROOT",
                                  Path(__file__).resolve().parents[1]))
CHECKPOINTS = ROOT / "checkpoints"
INDEX_CSV   = CHECKPOINTS / "model_index.csv"

FAMILIES      = ("dlinear", "gru", "lstm", "moderntcn", "patch-tst")
HORIZONS      = (0.5, 1.0, 2.0, 4.0)      # forecast horizons, s
SEEDS         = (0, 1, 2)
WIENER_INPUTS = (5, 20)                   # Wiener input lengths, s

_MISSING = (
    "{path} does not exist.\n"
    "Checkpoints are repo data resolved relative to the package "
    f"(root: {ROOT}).\n"
    "If you installed non-editably, either reinstall with `pip install -e .` "
    "or point SEISIM_MODELS_ROOT at the project directory."
)


def resolve(rel: str | Path) -> Path:
    """Absolute path for a project-relative POSIX path (an index `path`)."""
    p = Path(rel)
    p = p if p.is_absolute() else ROOT / p
    if not p.exists():
        raise FileNotFoundError(_MISSING.format(path=p))
    return p


def find(model_type: str, output_sec: float = 2.0, seed: int = 0,
         input_sec: float = 20) -> Path:
    """Path of one shipped model file.

    model_type : dlinear | gru | lstm | moderntcn | patch-tst  (a checkpoint
                 for `GMP`), or wiener (a fit file for `Wiener`, every
                 horizon in one file)
    output_sec : 0.5 | 1 | 2 | 4 (forecast horizon, s; deep models)
    seed       : 0 | 1 | 2 (training seed; deep models)
    input_sec  : 5 | 20 (Wiener only; each deep family has one input length)
    """
    mt = model_type.lower()
    if mt == "wiener":
        return resolve(f"checkpoints/classical/wiener_w{input_sec:g}.npz")
    if mt not in FAMILIES:
        raise ValueError(f"model_type must be one of {FAMILIES + ('wiener',)}; "
                         f"got {model_type!r}")
    hits = sorted((CHECKPOINTS / mt).glob(f"w*x{output_sec:g}_s{seed}/unified_checkpoint.ckpt"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} {mt} checkpoints for {output_sec:g} s, "
                                f"seed {seed} under {CHECKPOINTS / mt}")
    return hits[0]


def list_checkpoints(root: str | Path | None = None) -> list[Path]:
    """Every shipped `unified_checkpoint.ckpt`, sorted."""
    base = Path(root) if root is not None else CHECKPOINTS
    return sorted(base.rglob("unified_checkpoint.ckpt"))
