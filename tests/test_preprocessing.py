"""
Smoke tests for Preprocessor on a synthetic DataFrame -- checks that
non-feature columns are excluded and that the built ColumnTransformer is
usable, without ever touching real data.
"""
import numpy as np
import pandas as pd

from aem_degradation.preprocessing import Preprocessor


def _tiny_split_df() -> pd.DataFrame:
    n = 12
    return pd.DataFrame({
        "SMILES": [f"MOL_{i % 4}" for i in range(n)],
        "Degradation(%)": np.linspace(0, 100, n),
        "random_split_diagnostic": ["train"] * n,
        "test_holdout": ["dev"] * n,
        "cv_fold": [i % 3 for i in range(n)],
        "is_halflife_proxy": [0] * n,
        "feature_a": np.arange(n, dtype=float),
        "feature_b": np.random.RandomState(0).randn(n),
    })


def test_feature_columns_excludes_non_feature_cols():
    df = _tiny_split_df()
    prep = Preprocessor()
    feats = prep.feature_columns(df)
    assert set(feats) == {"feature_a", "feature_b"}
    for col in prep.NON_FEATURE_COLS:
        assert col not in feats


def test_build_returns_fittable_column_transformer():
    df = _tiny_split_df()
    prep = Preprocessor()
    ct = prep.build(df)
    X = ct.fit_transform(df)
    assert X.shape[0] == len(df)
    # scaled numeric columns -> zero mean, unit variance (up to float tolerance)
    scaled = X[:, : len(prep.feature_columns(df))]
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-8)
