"""
Smoke tests for SplitBuilder on a synthetic, molecule-grouped DataFrame.
Never touches data/database_prepared.csv -- purely checks that grouping is
respected and every split column gets built the way splits.py documents.
"""
import numpy as np
import pandas as pd

from aem_degradation.splits import SplitBuilder


def _tiny_prepared_df(n_molecules=40, rows_per_molecule=2) -> pd.DataFrame:
    rows = []
    for m in range(n_molecules):
        for r in range(rows_per_molecule):
            rows.append({"SMILES": f"MOL_{m}", "Degradation(%)": float((m + r) % 100),
                        "Time(h)": r + 1, "feature_a": m * 0.1})
    return pd.DataFrame(rows)


def test_random_split_diagnostic_marks_every_row():
    df = _tiny_prepared_df()
    sb = SplitBuilder(n_splits=10, test_size=0.2, random_state=42)
    out = sb.add_random_diagnostic_split(df.copy())
    assert set(out["random_split_diagnostic"]) <= {"train", "test"}
    assert (out["random_split_diagnostic"] == "test").sum() > 0


def test_grouped_test_holdout_has_zero_molecule_overlap():
    df = _tiny_prepared_df()
    sb = SplitBuilder(n_splits=10, test_size=0.2, random_state=42)
    out = sb.add_grouped_test_holdout(df.copy())
    dev_mols = set(out.loc[out["test_holdout"] == "dev", "SMILES"])
    test_mols = set(out.loc[out["test_holdout"] == "test", "SMILES"])
    assert dev_mols and test_mols
    assert dev_mols.isdisjoint(test_mols), "a molecule leaked across dev/test"


def test_cv_folds_keep_each_dev_molecule_in_exactly_one_fold():
    df = _tiny_prepared_df()
    sb = SplitBuilder(n_splits=10, test_size=0.2, random_state=42)
    out = sb.build(df.copy())

    test_rows = out[out["test_holdout"] == "test"]
    assert (test_rows["cv_fold"] == -1).all(), "test rows must never get a real cv_fold"

    dev_rows = out[out["test_holdout"] == "dev"]
    assert (dev_rows["cv_fold"] >= 0).all()
    fold_span = dev_rows.groupby("SMILES")["cv_fold"].nunique()
    assert (fold_span == 1).all(), "GroupKFold must not split one molecule across folds"
    assert dev_rows["cv_fold"].nunique() == sb.n_splits


def test_run_writes_output_and_reproduces_with_same_seed(tmp_path):
    src = tmp_path / "prepared.csv"
    out1 = tmp_path / "splits1.csv"
    out2 = tmp_path / "splits2.csv"
    _tiny_prepared_df().to_csv(src, index=False)

    df1 = SplitBuilder(source_csv=src, output_csv=out1, n_splits=10, random_state=42).run()
    df2 = SplitBuilder(source_csv=src, output_csv=out2, n_splits=10, random_state=42).run()

    assert np.array_equal(df1["test_holdout"].values, df2["test_holdout"].values)
    assert np.array_equal(df1["cv_fold"].values, df2["cv_fold"].values)
