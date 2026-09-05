"""Gradient-boosted trees -- the "boosting" half of the tree-ensemble
comparison alongside Random Forest's bagging. Raw Time(h) only, no
log-time feature, same reasoning as Random Forest."""
from sklearn.ensemble import GradientBoostingRegressor

from .base import ModelFamily

RANDOM_STATE = 42


class GradientBoostingModel(ModelFamily):
    name = "gb"
    log_time = False

    def build_pipeline(self, dev_df):
        return self._preprocessed_pipeline(
            dev_df, GradientBoostingRegressor(random_state=RANDOM_STATE)
        )

    @property
    def param_grid(self):
        return {
            "model__n_estimators": [200, 500],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 3],
        }
