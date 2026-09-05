"""
Smoke tests for SHAPExplainer, ZouBenchmark, and FigureGenerator on
synthetic, molecule-grouped data. Never touches the real data/results.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from aem_degradation.benchmarking import ZouBenchmark
from aem_degradation.figures import FigureGenerator
from aem_degradation.interpretability import SHAPExplainer
from aem_degradation.models.random_forest import RandomForestModel
from aem_degradation.protocol import ExperimentProtocol
from aem_degradation.splits import SplitBuilder


def _synth_splits_csv(tmp_path, n_mol=60, rows_per_mol=3, seed=2) -> Path:
    rng = np.random.RandomState(seed)
    rows = [
        {
            "SMILES": f"MOL_{m}", "Degradation(%)": float(rng.uniform(0, 100)),
            "Time(h)": rng.uniform(1, 500), "Concentration": rng.uniform(0.5, 2.0),
            "Temperature": rng.uniform(20, 90), "feature_a": rng.randn(), "feature_b": rng.randn(),
        }
        for m in range(n_mol) for _ in range(rows_per_mol)
    ]
    df = SplitBuilder(n_splits=10, test_size=0.2, random_state=42).build(pd.DataFrame(rows))
    path = tmp_path / "splits.csv"
    df.to_csv(path, index=False)
    return path


def _fit_rf(tmp_path):
    splits_csv = _synth_splits_csv(tmp_path)
    results_dir = tmp_path / "results"
    protocol = ExperimentProtocol(data_csv=splits_csv, results_dir=results_dir, n_splits=10)
    dev_df, test_df = protocol.load_data(log_time=False)
    RandomForestModel().run(protocol)
    return results_dir, dev_df, test_df


def test_shap_explainer_runs_all_three_parts_and_axioms_pass(tmp_path):
    results_dir, dev_df, test_df = _fit_rf(tmp_path)
    out = SHAPExplainer("rf", n_splits=10, results_dir=results_dir).run(dev_df, test_df)

    assert (results_dir / "rf_final_shap_importance.csv").exists()
    assert (results_dir / "rf_shap_by_fold.csv").exists()
    assert (results_dir / "rf_interpretability.json").exists()
    assert -1.0 <= out["mean_pairwise_spearman"] <= 1.0
    assert out["axioms"]["consistency_pass"] is True
    assert out["axioms"]["missingness_max_abs"] < 1e-9
    assert out["axioms"]["additivity_max_abs_dev"] < 1e-6


def test_zou_benchmark_compares_against_top15_and_reports_condition_share(tmp_path):
    results_dir, _dev_df, _test_df = _fit_rf(tmp_path)
    dev_df_2, test_df_2 = ExperimentProtocol(
        data_csv=results_dir.parent / "splits.csv", results_dir=results_dir
    ).load_data(log_time=False)
    SHAPExplainer("rf", n_splits=10, results_dir=results_dir).final_shap(dev_df_2, test_df_2)

    out = ZouBenchmark(results_dir=results_dir, top_n=5).compare("rf")
    assert len(out["our_top"]) == 5
    assert 0.0 <= out["condition_feature_share_of_shap"] <= 1.0
    assert set(out["sub_overlaps"].keys()) == {5, 10}


def test_figure_generator_writes_all_three_pdfs(tmp_path):
    results_dir, dev_df, test_df = _fit_rf(tmp_path)
    SHAPExplainer("rf", n_splits=10, results_dir=results_dir).run(dev_df, test_df)

    figs = FigureGenerator(results_dir=results_dir).run("rf")
    assert len(figs) == 3
    assert all(p.exists() and p.suffix == ".pdf" for p in figs)
