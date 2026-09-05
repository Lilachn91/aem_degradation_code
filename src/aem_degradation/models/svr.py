"""Support Vector Regression, RBF kernel -- extends the max-margin SVM
formulation to a continuous target via the epsilon-insensitive loss
(Drucker et al. 1997). Scale-sensitive family: log1p(Time(h)) alongside
raw Time(h)."""
from sklearn.svm import SVR as SklearnSVR

from .base import ModelFamily


class SVRModel(ModelFamily):
    name = "svr"
    log_time = True

    def build_pipeline(self, dev_df):
        return self._preprocessed_pipeline(dev_df, SklearnSVR(kernel="rbf"))

    @property
    def param_grid(self):
        return {
            "model__C": [1, 10, 100, 300],
            "model__epsilon": [1.0, 5.0, 10.0],
            "model__gamma": ["scale", 0.01, 0.001],
        }
