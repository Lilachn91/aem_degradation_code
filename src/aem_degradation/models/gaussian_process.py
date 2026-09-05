"""Gaussian Process regression -- chosen in part because it returns
predictive uncertainty alongside the point estimate, relevant for
eventually screening the original study's unlabeled candidate pool.

Kernel: ConstantKernel * RBF + WhiteKernel (squared-exponential; the White
term absorbs measurement noise). Length scale and signal/noise variances
are optimized internally by marginal likelihood on each fit; the grid
tunes only the additional alpha jitter. Scale-sensitive family: log1p
(Time(h)) alongside raw.
"""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from .base import ModelFamily

RANDOM_STATE = 42


class GaussianProcessModel(ModelFamily):
    name = "gp"
    log_time = True
    # GP fits are O(n^3); capping n_jobs keeps 3 configs x 10 folds from
    # oversubscribing memory (each fit holds a ~600x600 kernel matrix).
    default_n_jobs = 2

    def _kernel(self):
        return (
            ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e4))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))
        )

    def build_pipeline(self, dev_df):
        model = GaussianProcessRegressor(
            kernel=self._kernel(), normalize_y=True, n_restarts_optimizer=1, random_state=RANDOM_STATE
        )
        return self._preprocessed_pipeline(dev_df, model)

    @property
    def param_grid(self):
        return {"model__alpha": [1e-2, 1e-1, 1.0]}
