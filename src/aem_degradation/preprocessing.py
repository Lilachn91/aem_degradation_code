"""
Feature scaling. Object-oriented port of the original preprocessing.py --
same column groups, same fit-inside-each-fold discipline, unchanged.

This deliberately does NOT expose a way to write a pre-scaled CSV. Per the
project's methodology: feature selection, PCA/PLS, scaling, and
hyperparameter tuning happen only inside cross-validation (no fitting on
the full dataset first). Fitting a StandardScaler on the whole dev set (or
worse, the whole 896-row table including the locked test holdout) and
persisting the result would leak each fold's/test's own mean and variance
into the features every other fold trains on.

Preprocessor.build(df) returns an UNFIT sklearn ColumnTransformer. The
protocol module wraps it in a Pipeline with each model
(Pipeline([("prep", Preprocessor().build(df)), ("model", ...)])) and calls
.fit() separately inside every CV fold (and once more on the full dev set
for the final locked-test evaluation) -- so the scaler only ever sees
training rows for whichever split it's currently being used in.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

TARGET_COL = "Degradation(%)"


class Preprocessor:
    """Builds the (unfit) scaling ColumnTransformer for a given dataframe."""

    # Identifiers, target, split-bookkeeping columns, and is_halflife_proxy.
    # Never enter the feature matrix. is_halflife_proxy is metadata about how
    # a row was produced (a literature half-life re-encoded as a fake
    # single-timepoint row) -- not a chemistry or experimental-condition
    # input a new molecule could ever supply -- so it must not be a predictor.
    NON_FEATURE_COLS = [
        "SMILES",
        TARGET_COL,
        "random_split_diagnostic",
        "test_holdout",
        "cv_fold",
        "is_halflife_proxy",
    ]

    # Extension point for any real binary predictor added later (a
    # passthrough entry here would skip scaling, appropriate for a true 0/1
    # indicator feature). Empty for the current feature set.
    BINARY_COLS: list[str] = []

    def feature_columns(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in self.NON_FEATURE_COLS and c not in self.BINARY_COLS]

    def build(self, df: pd.DataFrame) -> ColumnTransformer:
        numeric_cols = self.feature_columns(df)
        return ColumnTransformer(
            transformers=[
                ("scale", StandardScaler(), numeric_cols),
                ("passthrough_binary", "passthrough", self.BINARY_COLS),
            ]
        )


if __name__ == "__main__":
    from .paths import SPLITS_CSV

    df = pd.read_csv(SPLITS_CSV)
    prep = Preprocessor()
    num_cols = prep.feature_columns(df)
    print(
        f"{len(num_cols)} columns to scale, {len(prep.BINARY_COLS)} binary passthrough, "
        f"{len(prep.NON_FEATURE_COLS)} excluded (identifier/target/split bookkeeping)"
    )
    prep.build(df)
    print("built ColumnTransformer (unfit -- fit inside each CV fold, not here)")
