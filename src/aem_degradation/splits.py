"""
Splitting strategy. Object-oriented port of the original build_splits.py --
same three split columns, same algorithm, same defaults (N_SPLITS=10,
TEST_SIZE=0.2, RANDOM_STATE=42), unchanged.

Builds the split assignments on top of database_prepared.csv and appends
them as columns, so every downstream model comparison reads the exact same
splits:

(a) random_split_diagnostic -- plain row-level 80/20 train/test. Rows are
    NOT grouped by molecule, so the same molecule's repeated measurements
    can land in both train and test. NOT part of the modeling protocol --
    it exists only to demonstrate, with real numbers, why row-level random
    splitting is the wrong thing to trust here.

(b) test_holdout -- the real protocol, two nested layers grouped by SMILES
    so no molecule's chemistry ever appears on both sides of any split:
      - dev / test: one grouped 80/20 split (GroupShuffleSplit). The test
        molecules are locked away and touched exactly once, at the very
        end, for the final reported number.
      - cv_fold: within the dev 80% only, 10-fold GroupKFold, used for
        model comparison and lightweight hyperparameter tuning (fixed/
        lightly-tuned hyperparameters, not nested CV). Test rows get
        cv_fold = -1 (never used for tuning or fold-based scoring).

Why cross-validate at all, rather than a single train/validation split:
with only 120 development molecules, one split would leave any performance
estimate hostage to which molecules happened to land on the validation
side. Cross-validating instead (the cv_fold column) means every
development molecule is held out exactly once, and the reported dev score
is pooled over all of them rather than over one arbitrary subset. (Paper
Methodology, "Data Integrity and Validation Strategy" -- trimmed out of
the paper text to this docstring since it's a generic CV-design point,
not a project-specific finding.)

(c) family-holdout (train on some chemical families, test on a held-out
    one) is intentionally not built here -- future work.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

from .paths import PREPARED_CSV, SPLITS_CSV


class SplitBuilder:
    def __init__(
        self,
        source_csv: Path = PREPARED_CSV,
        output_csv: Path = SPLITS_CSV,
        n_splits: int = 10,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.source_csv = Path(source_csv)
        self.output_csv = Path(output_csv)
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.diagnostics: dict = {}

    def add_random_diagnostic_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """(a) Row-level 80/20 split, diagnostic only -- quantifies leakage."""
        train_idx, test_idx = train_test_split(
            df.index, test_size=self.test_size, random_state=self.random_state
        )
        df["random_split_diagnostic"] = "train"
        df.loc[test_idx, "random_split_diagnostic"] = "test"

        leaked_mols = set(df.loc[train_idx, "SMILES"]) & set(df.loc[test_idx, "SMILES"])
        n_leaked_rows = df["SMILES"].isin(leaked_mols).sum()
        self.diagnostics["random_split"] = {
            "n_train_rows": int(len(train_idx)),
            "n_test_rows": int(len(test_idx)),
            "n_molecules_leaked": int(len(leaked_mols)),
            "n_rows_affected": int(n_leaked_rows),
            "pct_rows_affected": round(100 * n_leaked_rows / len(df), 1),
        }
        return df

    def add_grouped_test_holdout(self, df: pd.DataFrame) -> pd.DataFrame:
        """(b) Locked test holdout, grouped by SMILES -- the real protocol."""
        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        dev_i, test_i = next(gss.split(df, groups=df["SMILES"]))
        df["test_holdout"] = "dev"
        df.loc[df.index[test_i], "test_holdout"] = "test"

        dev_mols = set(df.loc[df.index[dev_i], "SMILES"])
        test_mols = set(df.loc[df.index[test_i], "SMILES"])
        assert not (dev_mols & test_mols), "a molecule leaked across the dev/test holdout"
        self.diagnostics["test_holdout"] = {
            "n_dev_molecules": len(dev_mols),
            "n_test_molecules": len(test_mols),
            "n_dev_rows": int((df["test_holdout"] == "dev").sum()),
            "n_test_rows": int((df["test_holdout"] == "test").sum()),
        }
        return df

    def add_dev_cv_folds(self, df: pd.DataFrame) -> pd.DataFrame:
        """GroupKFold by SMILES, within the dev portion only."""
        dev_df = df.loc[df["test_holdout"] == "dev"]
        gkf = GroupKFold(n_splits=self.n_splits)
        df["cv_fold"] = -1
        for fold, (_, fold_test_i) in enumerate(gkf.split(dev_df, groups=dev_df["SMILES"])):
            df.loc[dev_df.index[fold_test_i], "cv_fold"] = fold

        fold_span = df[df["cv_fold"] >= 0].groupby("SMILES")["cv_fold"].nunique()
        assert (fold_span == 1).all(), "GroupKFold leaked a molecule across folds"
        return df

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all three split steps in order and return the annotated df."""
        df = self.add_random_diagnostic_split(df)
        df = self.add_grouped_test_holdout(df)
        df = self.add_dev_cv_folds(df)
        return df

    def run(self) -> pd.DataFrame:
        df = pd.read_csv(self.source_csv)
        df = self.build(df)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False)
        self._print_summary(df)
        return df

    def _print_summary(self, df: pd.DataFrame) -> None:
        print(f"wrote {self.output_csv}")
        print(f"{len(df)} rows x {len(df.columns)} columns, {df['SMILES'].nunique()} molecules")
        print()
        print("=== (a) random split -- diagnostic only, NOT used for modeling ===")
        rs = self.diagnostics["random_split"]
        print(f"train: {rs['n_train_rows']} rows, test: {rs['n_test_rows']} rows")
        print(f"molecules with rows in BOTH train and test (leakage): "
              f"{rs['n_molecules_leaked']} / {df['SMILES'].nunique()}")
        print(f"rows affected by that leakage: {rs['n_rows_affected']} / {len(df)} "
              f"({rs['pct_rows_affected']}%)")
        print()
        print("=== (b) locked test_holdout, grouped by SMILES -- the real protocol ===")
        th = self.diagnostics["test_holdout"]
        print(f"dev: {th['n_dev_molecules']} molecules / {th['n_dev_rows']} rows")
        print(f"test (locked, touch once at the end): {th['n_test_molecules']} molecules / "
              f"{th['n_test_rows']} rows")
        print("confirmed: zero molecule overlap between dev and test")
        print()
        print(f"=== cv_fold: {self.n_splits}-fold GroupKFold within dev only "
              f"(tuning + model comparison) ===")
        print(df.loc[df["cv_fold"] >= 0, "cv_fold"].value_counts().sort_index())
        print("molecules per fold:")
        print(df[df["cv_fold"] >= 0].groupby("cv_fold")["SMILES"].nunique())
        print("confirmed: every dev molecule stays in exactly one fold; test rows are cv_fold = -1")


if __name__ == "__main__":
    SplitBuilder().run()
