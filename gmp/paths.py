"""Where the shipped checkpoints live, and how to name one.

The project root is derived once from this file's location, so nothing in the
package (or in a notebook that imports it) ever needs an absolute path. Set
`SEISIM_MODELS_ROOT` to override — needed only if the package was installed
non-editable and the `checkpoints/` tree stayed behind.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT        = Path(os.environ.get("SEISIM_MODELS_ROOT",
                                  Path(__file__).resolve().parents[1]))
CHECKPOINTS = ROOT / "checkpoints"
INDEX_CSV   = CHECKPOINTS / "model_index.csv"

FAMILIES   = ("dlinear", "gru", "lstm", "moderntcn", "patch-tst")
SUB_TYPES  = ("summer", "winter", "eql", "eqh", "all")
TARGETS    = ("absolute", "delta")
LEGACY_FAMILIES = ("cnn-lstm", "dlinear", "gru", "lstm", "patch-tst")

_MISSING = (
    "{path} does not exist.\n"
    "Checkpoints are repo data resolved relative to the package "
    f"(root: {ROOT}).\n"
    "If you installed non-editably, either reinstall with `pip install -e .` "
    "or point SEISIM_MODELS_ROOT at the project directory."
)


def resolve(rel: str | Path) -> Path:
    """Absolute path for a project-relative POSIX path (an index `ckpt_path`)."""
    p = Path(rel)
    p = p if p.is_absolute() else ROOT / p
    if not p.exists():
        raise FileNotFoundError(_MISSING.format(path=p))
    return p


def _win(input_sec: float, output_sec: float) -> str:
    return f"w{input_sec:g}x{output_sec:g}"


def find(model_type: str, sub_type: str = "all", output_sec: float = 0.5,
         target: str = "absolute", input_sec: float = 5) -> Path:
    """Path of one shipped checkpoint.

    model_type : dlinear | gru | lstm | moderntcn | patch-tst | naive | cnn-lstm
    sub_type   : the training condition — summer | winter | eql | eqh | all,
                 or "legacy" for a GMP_X3 checkpoint ("-" / "naive" for the
                 persistence baseline).
    output_sec : 0.5 or 1 (forecast horizon in seconds)
    target     : absolute | delta (training-side only; both return absolute)
    input_sec  : 5 for every X5 checkpoint shipped here.

    Legacy and persistence checkpoints ignore `output_sec`/`target`/`input_sec`
    where they don't apply.
    """
    mt = model_type.lower()
    st = str(sub_type).lower()
    if st == "legacy":
        return resolve(f"checkpoints/legacy/{mt}/unified_checkpoint.ckpt")
    if mt == "naive" or st in {"-", "naive"}:
        return resolve(f"checkpoints/naive/{_win(input_sec, output_sec)}"
                       f"_absolute/unified_checkpoint.ckpt")
    return resolve(f"checkpoints/{mt}/{st}/"
                   f"{_win(input_sec, output_sec)}_{target}/"
                   "unified_checkpoint.ckpt")


def list_checkpoints(root: str | Path | None = None) -> list[Path]:
    """Every shipped `unified_checkpoint.ckpt`, sorted."""
    base = Path(root) if root is not None else CHECKPOINTS
    return sorted(base.rglob("unified_checkpoint.ckpt"))
