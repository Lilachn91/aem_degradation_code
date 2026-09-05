"""
Shallow MLP -- the course's required neural-network baseline, deliberately
not a deep architecture. Regularized via L2 weight decay (alpha) + early
stopping rather than depth (sklearn's MLPRegressor exposes both; dropout
is not available in sklearn). Optimizer: Adam (sklearn default), loss:
squared error. Scale-sensitive family: log1p(Time(h)) alongside raw.

The alpha grid runs well past a lightly-regularized range because an
earlier pass showed the un/lightly-regularized MLP overfits this
feature-rich, small dataset.

Hidden activation is grid-searched too (relu vs. tanh vs. logistic
[sigmoid]) rather than left at the sklearn default, so the choice is
backed by the same inner-CV evidence as every other hyperparameter here.
"""
from sklearn.neural_network import MLPRegressor

from .base import ModelFamily

RANDOM_STATE = 42


class MLPModel(ModelFamily):
    name = "mlp"
    log_time = True

    def build_pipeline(self, dev_df):
        model = MLPRegressor(
            random_state=RANDOM_STATE, max_iter=3000, early_stopping=True, n_iter_no_change=20
        )
        return self._preprocessed_pipeline(dev_df, model)

    @property
    def param_grid(self):
        return {
            "model__hidden_layer_sizes": [(32,), (64,), (32, 16)],
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 30.0],
            "model__activation": ["relu", "tanh", "logistic"],
        }
