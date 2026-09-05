"""Random Forest -- tree ensemble, raw Time(h) only (invariant to monotone
transforms of the input, so no log-time feature is needed)."""
from sklearn.ensemble import RandomForestRegressor

from .base import ModelFamily

RANDOM_STATE = 42


class RandomForestModel(ModelFamily):
    name = "rf"
    log_time = False

    def build_pipeline(self, dev_df):
        return self._preprocessed_pipeline(
            dev_df, RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        )

    @property
    def param_grid(self):
        return {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 3],
        }
