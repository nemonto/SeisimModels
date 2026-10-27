"""Pretrained ground-motion predictors (GMP_X6/X8), ready for inference.

    from gmp import GMP, find, prepare
    ground, x = prepare(raw_512hz)              # seisim's ground, the model input
    model = GMP(find("gru", output_sec=2))
    forecast = model.predict_long(x)            # aligned to x[K:]

`Wiener` and `Persistence` are the paper's classical baselines with the same
interface. `gmp.benchmark` has the benchmark's forecast stream and RMSP;
`gmp.seisim_adapter` is deliberately not imported here — it needs `seisim`
and `control` on the path.
"""
from .predictor  import GMP, default_device
from .classical  import Persistence, Wiener
from .benchmark  import CONTROL_START_SEC, forecast_stream, rmsp
from .paths      import (CHECKPOINTS, FAMILIES, HORIZONS, INDEX_CSV, ROOT,
                         SEEDS, WIENER_INPUTS, find, list_checkpoints, resolve)
from .preprocess import BAND, ground, prepare, recipe, zero_phase_band

__all__ = [
    "GMP", "Wiener", "Persistence", "default_device",
    "find", "resolve", "list_checkpoints",
    "ROOT", "CHECKPOINTS", "INDEX_CSV",
    "FAMILIES", "HORIZONS", "SEEDS", "WIENER_INPUTS",
    "prepare", "ground", "zero_phase_band", "recipe", "BAND",
    "forecast_stream", "rmsp", "CONTROL_START_SEC",
]

__version__ = "2.0.0"
