"""
Smoke tests for the ModelFamily subclasses -- checks pipeline construction
and hyperparameter-grid wiring on a tiny synthetic DataFrame. Never fits
against, or even imports, any real CSV under data/.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from aem_degradation.models import REGISTRY, GaussianProcessModel


def _tiny_dev_df() -> pd.DataFrame:
    n = 16
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "SMILES": [f"MOL_{i % 5}" for i in range(n)],
        "Degradation(%)": rng.uniform(0, 100, n),
        "random_split_diagnostic": ["train"] * n,
        "test_holdout": ["dev"] * n,
        "cv_fold": [i % 4 for i in range(n)],
        "is_halflife_proxy": [0] * n,
        "Time(h)": rng.uniform(1, 500, n),
        "Concentration": rng.uniform(0.5, 2.0, n),
        "Temperature": rng.uniform(20, 90, n),
        "feature_a": rng.randn(n),
    })
    df["Time_log1p"] = np.log1p(df["Time(h)"])
    return df


@pytest.mark.parametrize("name", list(REGISTRY.keys()))
def test_build_pipeline_has_prep_and_model_steps(name):
    dev_df = _tiny_dev_df()
    family = REGISTRY[name]()
    pipe = family.build_pipeline(dev_df)
    assert isinstance(pipe, Pipeline)
    assert [s[0] for s in pipe.steps] == ["prep", "model"]


@pytest.mark.parametrize("name", list(REGISTRY.keys()))
def test_param_grid_keys_target_the_model_step(name):
    family = REGISTRY[name]()
    grid = family.param_grid
    assert grid, f"{name} must declare a non-empty param_grid"
    assert all(k.startswith("model__") for k in grid), \
        f"{name} param_grid keys must target the 'model' pipeline step"


def test_gaussian_process_overrides_default_n_jobs_down_from_base():
    from aem_degradation.models.base import ModelFamily
    assert ModelFamily.default_n_jobs == -1
    assert GaussianProcessModel.default_n_jobs == 2, \
        "GP fits are O(n^3); the family must cap GridSearchCV parallelism"
