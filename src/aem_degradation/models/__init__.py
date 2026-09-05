from .base import ModelFamily
from .gaussian_process import GaussianProcessModel
from .gradient_boosting import GradientBoostingModel
from .mlp import MLPModel
from .pls import PLSModel
from .random_forest import RandomForestModel
from .svr import SVRModel

#: name -> class, for CLI dispatch (see scripts/run_pipeline.py)
REGISTRY: dict[str, type[ModelFamily]] = {
    "pls": PLSModel,
    "rf": RandomForestModel,
    "gb": GradientBoostingModel,
    "svr": SVRModel,
    "gp": GaussianProcessModel,
    "mlp": MLPModel,
}

__all__ = [
    "ModelFamily",
    "PLSModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "SVRModel",
    "GaussianProcessModel",
    "MLPModel",
    "REGISTRY",
]
