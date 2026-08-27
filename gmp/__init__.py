"""Pretrained ground-motion predictors, ready for inference.

    from gmp import GMP, find, prepare
    ckpt = find("gru", sub_type="all", output_sec=0.5, target="absolute")
    model = GMP(ckpt)
    forecast = model.predict_long(prepare(raw_512hz, ckpt))

`load_gmp(path)` is the safe entry point when the path may be one of the
legacy GMP_X3 checkpoints under `checkpoints/legacy/`.

`gmp.seisim_adapter` is deliberately not imported here — it needs `seisim` and
`control` on the path.
"""
from .predictor  import GMP
from .legacy     import LegacyGMP, is_legacy_checkpoint, load_gmp
from .paths      import (CHECKPOINTS, FAMILIES, INDEX_CSV, LEGACY_FAMILIES,
                         ROOT, SUB_TYPES, TARGETS, find, list_checkpoints,
                         resolve)
from .preprocess import (bandpass, bandpass_decimate, decimate_to, prepare,
                         recipe)

__all__ = [
    "GMP", "LegacyGMP", "load_gmp", "is_legacy_checkpoint",
    "find", "resolve", "list_checkpoints",
    "ROOT", "CHECKPOINTS", "INDEX_CSV",
    "FAMILIES", "SUB_TYPES", "TARGETS", "LEGACY_FAMILIES",
    "prepare", "recipe", "bandpass", "decimate_to", "bandpass_decimate",
]

__version__ = "1.0.0"
