"""
Smoke tests for RowLevelDiagnostic and KFoldSensitivity on synthetic,
molecule-grouped data. Never touches the real data/results.

Note: KFoldSensitivity.WINNER_PARAMS is deliberately hardcoded to the real
project's Random Forest winner (see results/rf.json), not re-derived from
whatever a given run's grid search picks -- that is the point of the
diagnostic (fix hyperparameters, vary only the fold structure). So on a
synthetic dataset (where grid search naturally lands on different
hyperparameters) "reproduces_published" need not be True; these tests only
check structural correctness, not that number.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from aem_degradation.diagnostics import KFoldSensitivity, RowLevelDiagnostic
from aem_degradation.models.random_forest import RandomForestModel
from aem_degradation.protocol import ExperimentProtocol
from aem_degradation.splits import SplitBuilder


def _synth_splits_csv(tmp_path, n_mol=60, rows_per_mol=3, seed=1) -> Path:
    rng = np.random.RandomState(seed)
    rows = [
        {
            "SMILES": f"MOL_{m}", "Degradation(%)": float(rng.uniform(0, 100)),
            "Time(h)": rng.uniform(1, 500), "Concentration": rng.uniform(0.5, 2.0),
            "Temperature": rng.uniform(20, 90), "feature_a": rng.randn(),
        }
        for m in range(n_mol) for _ in range(rows_per_mol)
    ]
    df = SplitBuilder(n_splits=10, test_size=0.2, random_state=42).build(pd.DataFrame(rows))
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    return path


def test_row_level_diagnostic_writes_leakage_and_inflation(tmp_path):
    splits_csv = _synth_splits_csv(tmp_path)
    results_dir = tmp_path / "results"
    protocol = ExperimentProtocol(data_csv=splits_csv, results_dir=results_dir, n_splits=10)
    RandomForestModel().run(protocol)  # rf.json is required as the "grouped" reference

    out = RowLevelDiagnostic(data_csv=splits_csv, results_dir=results_dir).run()

    assert (results_dir / "rf_rowlevel_diagnostic.json").exists()
    assert out["leakage"]["n_molecules_both_sides"] > 0, \
        "the naive row-level split should leak molecules across train/test on this data"
    assert "inflation" in out and "cv_r2_delta" in out["inflation"]


def test_kfold_sensitivity_covers_declared_k_values(tmp_path):
    splits_csv = _synth_splits_csv(tmp_path)
    results_dir = tmp_path / "results"
    protocol = ExperimentProtocol(data_csv=splits_csv, results_dir=results_dir, n_splits=10)
    RandomForestModel().run(protocol)

    kfs = KFoldSensitivity(protocol=protocol, results_dir=results_dir)
    out = kfs.run()

    assert (results_dir / "kfold_sensitivity.json").exists()
    for k in kfs.K_VALUES:
        assert str(k) in out["by_k"]
        assert out["by_k"][str(k)]["n_partitions"] == kfs.N_SEEDS
    assert "LOMO" in out["by_k"]
    assert isinstance(out["reproduces_published"], bool)
