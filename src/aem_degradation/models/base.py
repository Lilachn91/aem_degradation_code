"""Abstract base class every model family implements."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from sklearn.pipeline import Pipeline

from ..preprocessing import Preprocessor
from ..protocol import ExperimentProtocol


class ModelFamily(ABC):
    """One regression family under the shared evaluation protocol.

    Subclasses declare `name`, whether they need the log1p(Time(h)) feature
    (`log_time`), how to build their (unfit) pipeline, and their
    hyperparameter grid. `run()` then follows the identical grid-search ->
    pooled-OOF -> locked-test sequence for every family.
    """

    name: str
    log_time: bool = False
    # Per-family override for GridSearchCV's n_jobs (e.g. GP fits are O(n^3)
    # and hold a large kernel matrix per fit, so it caps parallelism lower).
    default_n_jobs: int = -1

    @abstractmethod
    def build_pipeline(self, dev_df: pd.DataFrame) -> Pipeline:
        """Return an unfit sklearn Pipeline: [("prep", ...), ("model", ...)]."""

    @property
    @abstractmethod
    def param_grid(self) -> dict:
        ...

    def _preprocessed_pipeline(self, dev_df: pd.DataFrame, estimator) -> Pipeline:
        return Pipeline([("prep", Preprocessor().build(dev_df)), ("model", estimator)])

    def run(self, protocol: ExperimentProtocol, n_jobs: int | None = None):
        dev_df, test_df = protocol.load_data(log_time=self.log_time)
        pipeline = self.build_pipeline(dev_df)
        n_jobs = self.default_n_jobs if n_jobs is None else n_jobs
        return protocol.run(self.name, pipeline, self.param_grid, dev_df, test_df, n_jobs=n_jobs)
