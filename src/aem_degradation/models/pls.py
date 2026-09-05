"""
PLS regression -- the standard chemometric approach for collinear,
high-dimensional tabular descriptors. The number of latent components is
selected by the same inner-CV grid search every other family uses (see
protocol.ExperimentProtocol), over n_components in {2,4,6,8,10,15,20,25,30}.
"""
from sklearn.cross_decomposition import PLSRegression

from .base import ModelFamily


class PLSModel(ModelFamily):
    name = "pls"
    log_time = True  # scale-sensitive family: log1p(Time(h)) alongside raw

    def build_pipeline(self, dev_df):
        return self._preprocessed_pipeline(dev_df, PLSRegression())

    @property
    def param_grid(self):
        return {"model__n_components": [2, 4, 6, 8, 10, 15, 20, 25, 30]}
