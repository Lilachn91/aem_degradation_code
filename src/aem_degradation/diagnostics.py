"""
Two diagnostics that are NOT part of the reported modeling protocol -- they
exist to quantify how load-bearing two protocol choices are. Object-oriented
port of model_rf_rowlevel_diagnostic.py and kfold_sensitivity.py; neither
touches the locked test set or overwrites the locked splits.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline

from .paths import RESULTS_DIR, SPLITS_CSV
from .preprocessing import Preprocessor, TARGET_COL
from .protocol import ExperimentProtocol

RANDOM_STATE = 42


class RowLevelDiagnostic:
    """The same Random Forest winner, run under a naive row-level split
    instead of the molecule-grouped split, to quantify how much apparent
    performance row-level splitting manufactures on this dataset.

    A controlled A/B against results/rf.json: identical feature set,
    ColumnTransformer/StandardScaler discipline, param_grid, GridSearchCV
    scoring, and pooled-out-of-fold scoring. The ONLY thing that changes is
    the split policy (test_holdout/GroupKFold vs. random_split_diagnostic/
    KFold(shuffle=True)).
    """

    SPLIT_COL = "random_split_diagnostic"
    N_SPLITS = 10  # same as protocol.py
    PARAM_GRID = {
        "model__n_estimators": [200, 500],
        "model__max_depth": [None, 10],
        "model__min_samples_leaf": [1, 3],
    }

    def __init__(self, data_csv: Path = SPLITS_CSV, results_dir: Path = RESULTS_DIR):
        self.data_csv = Path(data_csv)
        self.results_dir = Path(results_dir)

    def _leakage_stats(self, df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        train_mols, test_mols = set(train_df["SMILES"]), set(test_df["SMILES"])
        leaked_mols = train_mols & test_mols
        n_leaked_rows = int(df["SMILES"].isin(leaked_mols).sum())
        return {
            "n_rows_total": int(len(df)),
            "n_molecules_total": int(df["SMILES"].nunique()),
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "n_test_molecules": int(len(test_mols)),
            "n_molecules_both_sides": int(len(leaked_mols)),
            "pct_molecules_both_sides": round(100 * len(leaked_mols) / df["SMILES"].nunique(), 1),
            "n_rows_of_leaked_molecules": n_leaked_rows,
            "pct_rows_of_leaked_molecules": round(100 * n_leaked_rows / len(df), 1),
            "pct_test_molecules_seen_in_train": round(100 * len(leaked_mols) / len(test_mols), 1),
        }

    def run(self) -> dict:
        df = pd.read_csv(self.data_csv)
        train_df = df[df[self.SPLIT_COL] == "train"].reset_index(drop=True)
        test_df = df[df[self.SPLIT_COL] == "test"].reset_index(drop=True)
        leakage = self._leakage_stats(df, train_df, test_df)
        print("=== leakage in the naive row-level split ===")
        for k, v in leakage.items():
            print(f"  {k}: {v}")
        print()

        base_pipeline = Pipeline([
            ("prep", Preprocessor().build(train_df)),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        ])
        y_train = train_df[TARGET_COL].values
        y_test = test_df[TARGET_COL].values

        kf = KFold(n_splits=self.N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        custom_cv = list(kf.split(train_df))

        import time
        t0 = time.time()
        grid = GridSearchCV(base_pipeline, self.PARAM_GRID, cv=custom_cv,
                            scoring="neg_root_mean_squared_error", n_jobs=-1)
        grid.fit(train_df, y_train)
        t_tune = time.time() - t0
        print(f"[rf_rowlevel] best params: {grid.best_params_}")
        print(f"[rf_rowlevel] best CV RMSE (mean of {self.N_SPLITS} fold scores): "
              f"{-grid.best_score_:.2f}  (tuning took {t_tune:.0f}s)")

        oof_pred = np.full(len(train_df), np.nan)
        for train_i, test_i in custom_cv:
            pipe = clone(base_pipeline).set_params(**grid.best_params_)
            pipe.fit(train_df.iloc[train_i], y_train[train_i])
            oof_pred[test_i] = np.asarray(pipe.predict(train_df.iloc[test_i])).ravel()
        assert not np.isnan(oof_pred).any()
        r2 = r2_score(y_train, oof_pred)
        rmse = mean_squared_error(y_train, oof_pred) ** 0.5
        mae = mean_absolute_error(y_train, oof_pred)
        print(f"[rf_rowlevel] pooled row-level CV ({len(train_df)} rows): "
              f"R2 = {r2:.3f}  RMSE = {rmse:.2f}  MAE = {mae:.2f}")

        final_pred = np.asarray(grid.best_estimator_.predict(test_df)).ravel()
        r2_t = r2_score(y_test, final_pred)
        rmse_t = mean_squared_error(y_test, final_pred) ** 0.5
        mae_t = mean_absolute_error(y_test, final_pred)
        print(f"[rf_rowlevel] row-level HELD-OUT TEST ({len(test_df)} rows): "
              f"R2 = {r2_t:.3f}  RMSE = {rmse_t:.2f}  MAE = {mae_t:.2f}")

        with open(self.results_dir / "rf.json") as f:
            grouped = json.load(f)

        # control arm: the GROUPED winner's exact hyperparameters, run under
        # the row-level split, isolating the split policy as the only variable
        grouped_params = {"model__" + k.split("model__")[-1]: v
                          for k, v in grouped["best_params"].items()}
        ctrl_oof = np.full(len(train_df), np.nan)
        for train_i, test_i in custom_cv:
            pipe = clone(base_pipeline).set_params(**grouped_params)
            pipe.fit(train_df.iloc[train_i], y_train[train_i])
            ctrl_oof[test_i] = np.asarray(pipe.predict(train_df.iloc[test_i])).ravel()
        assert not np.isnan(ctrl_oof).any()
        ctrl_r2 = r2_score(y_train, ctrl_oof)
        ctrl_rmse = mean_squared_error(y_train, ctrl_oof) ** 0.5
        ctrl_mae = mean_absolute_error(y_train, ctrl_oof)

        ctrl_final = clone(base_pipeline).set_params(**grouped_params)
        ctrl_final.fit(train_df, y_train)
        ctrl_pred = np.asarray(ctrl_final.predict(test_df)).ravel()
        ctrl_r2_t = r2_score(y_test, ctrl_pred)
        ctrl_rmse_t = mean_squared_error(y_test, ctrl_pred) ** 0.5
        ctrl_mae_t = mean_absolute_error(y_test, ctrl_pred)
        print(f"[rf_rowlevel/fixed-hparams] pooled row-level CV: "
              f"R2 = {ctrl_r2:.3f}  RMSE = {ctrl_rmse:.2f}  MAE = {ctrl_mae:.2f}")
        print(f"[rf_rowlevel/fixed-hparams] row-level HELD-OUT TEST: "
              f"R2 = {ctrl_r2_t:.3f}  RMSE = {ctrl_rmse_t:.2f}  MAE = {ctrl_mae_t:.2f}")

        out = {
            "name": "rf_rowlevel_diagnostic",
            "note": ("Diagnostic only. Same RF pipeline/grid as the RandomForestModel family; "
                     "only the split policy differs (row-level instead of grouped by SMILES)."),
            "split_policy": {
                "split_column": self.SPLIT_COL,
                "inner_cv": f"KFold(n_splits={self.N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
            },
            "leakage": leakage,
            "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                            for k, v in grid.best_params_.items()},
            "grid_best_cv_rmse": float(-grid.best_score_),
            "row_level_cv": {"r2": float(r2), "rmse": float(rmse), "mae": float(mae)},
            "row_level_test": {"r2": float(r2_t), "rmse": float(rmse_t), "mae": float(mae_t)},
            "row_level_fixed_grouped_hparams": {
                "params": grouped["best_params"],
                "cv": {"r2": float(ctrl_r2), "rmse": float(ctrl_rmse), "mae": float(ctrl_mae)},
                "test": {"r2": float(ctrl_r2_t), "rmse": float(ctrl_rmse_t), "mae": float(ctrl_mae_t)},
            },
            "grouped_reference": {
                "best_params": grouped["best_params"],
                "dev_cv": grouped["dev_cv"],
                "locked_test": grouped["locked_test"],
            },
            "inflation": {
                "cv_r2_delta": float(r2 - grouped["dev_cv"]["r2"]),
                "test_r2_delta": float(r2_t - grouped["locked_test"]["r2"]),
                "cv_rmse_delta": float(rmse - grouped["dev_cv"]["rmse"]),
                "test_rmse_delta": float(rmse_t - grouped["locked_test"]["rmse"]),
            },
            "tuning_seconds": round(t_tune, 1),
        }
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_dir / "rf_rowlevel_diagnostic.json", "w") as f:
            json.dump(out, f, indent=2)
        np.savez(self.results_dir / "rf_rowlevel_diagnostic_preds.npz",
                 y_train=y_train, oof_pred=oof_pred, y_test=y_test, test_pred=final_pred)
        return out


class KFoldSensitivity:
    """Sensitivity of the pooled dev-CV score to the number of GroupKFold
    folds and to which of several equally-valid partitions at that fold
    count is drawn. RF hyperparameters are held fixed at the grouped
    winner's values throughout; only the partitioning varies. Diagnostic
    only -- the locked test set is never touched.

    Justifies the paper's choice of k=10: repeating the pooled dev-CV
    score across k in {3, 5, 10, 15, 20} and leave-one-molecule-out
    (K_VALUES below), with hyperparameters fixed, confirms k=10 already
    sits close to the leave-one-out ceiling -- i.e. k=10 was not an
    arbitrary pick. This detail lives here rather than in main.tex to
    keep the paper's Data Integrity subsection short; see
    results/kfold_sensitivity.json for the numbers."""

    N_SEEDS = 10
    K_VALUES = [3, 5, 10, 15, 20]
    # the grouped winner's hyperparameters, from results/rf.json -- fixed on
    # purpose: the question is the score's sensitivity to the partition, not
    # to re-tuning
    WINNER_PARAMS = {
        "model__max_depth": 10,
        "model__min_samples_leaf": 1,
        "model__n_estimators": 200,
    }

    def __init__(self, protocol: ExperimentProtocol | None = None, results_dir: Path = RESULTS_DIR):
        self.protocol = protocol or ExperimentProtocol()
        self.results_dir = Path(results_dir)

    def score_partition(self, dev_df, y_dev, base_pipeline, splitter) -> dict:
        """Mirror ExperimentProtocol's manual OOF loop for an arbitrary splitter."""
        groups = dev_df["SMILES"].values
        oof = np.full(len(dev_df), np.nan)
        for train_i, test_i in splitter.split(dev_df, groups=groups):
            pipe = clone(base_pipeline).set_params(**self.WINNER_PARAMS)
            pipe.fit(dev_df.iloc[train_i], y_dev[train_i])
            oof[test_i] = np.asarray(pipe.predict(dev_df.iloc[test_i])).ravel()
        assert not np.isnan(oof).any(), "a dev row never landed in a test fold"
        return {
            "r2": float(r2_score(y_dev, oof)),
            "rmse": float(mean_squared_error(y_dev, oof) ** 0.5),
            "mae": float(mean_absolute_error(y_dev, oof)),
        }

    def score_saved_folds(self, dev_df, y_dev, base_pipeline) -> dict:
        """Exactly ExperimentProtocol's loop over the saved cv_fold column
        (the published run)."""
        folds = dev_df["cv_fold"].values
        oof = np.full(len(dev_df), np.nan)
        for f in np.unique(folds):
            train_i = np.where(folds != f)[0]
            test_i = np.where(folds == f)[0]
            pipe = clone(base_pipeline).set_params(**self.WINNER_PARAMS)
            pipe.fit(dev_df.iloc[train_i], y_dev[train_i])
            oof[test_i] = np.asarray(pipe.predict(dev_df.iloc[test_i])).ravel()
        assert not np.isnan(oof).any()
        return {
            "r2": float(r2_score(y_dev, oof)),
            "rmse": float(mean_squared_error(y_dev, oof) ** 0.5),
            "mae": float(mean_absolute_error(y_dev, oof)),
        }

    def run(self) -> dict:
        dev_df, _test_df = self.protocol.load_data(log_time=False)  # tree model -> raw Time(h)
        y_dev = dev_df[TARGET_COL].values
        n_mols = dev_df["SMILES"].nunique()

        base_pipeline = Pipeline([
            ("prep", Preprocessor().build(dev_df)),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        ])

        with open(self.results_dir / "rf.json") as f:
            published = json.load(f)["dev_cv"]
        repro = self.score_saved_folds(dev_df, y_dev, base_pipeline)
        delta = abs(repro["r2"] - published["r2"])
        print(f"saved cv_fold column -> R2={repro['r2']:.4f}  "
              f"(published {published['r2']:.4f}, |delta|={delta:.2e}) "
              f"{'REPRODUCED' if delta < 1e-9 else 'MISMATCH'}")
        print()

        results = {}
        for k in self.K_VALUES:
            runs = [
                self.score_partition(dev_df, y_dev, base_pipeline,
                                     GroupKFold(n_splits=k, shuffle=True, random_state=s))
                for s in range(self.N_SEEDS)
            ]
            r2s = np.array([r["r2"] for r in runs])
            results[str(k)] = {
                "n_splits": k, "n_partitions": self.N_SEEDS,
                "r2_mean": float(r2s.mean()), "r2_sd": float(r2s.std(ddof=1)),
                "r2_min": float(r2s.min()), "r2_max": float(r2s.max()),
                "rmse_mean": float(np.mean([r["rmse"] for r in runs])),
                "mae_mean": float(np.mean([r["mae"] for r in runs])),
                "r2_all": r2s.tolist(),
            }
            m = results[str(k)]
            print(f"k={k:>4}  R2 = {m['r2_mean']:.4f} +/- {m['r2_sd']:.4f}  "
                  f"[{m['r2_min']:.4f}, {m['r2_max']:.4f}]  RMSE={m['rmse_mean']:.2f}")

        lomo = self.score_partition(dev_df, y_dev, base_pipeline, GroupKFold(n_splits=n_mols))
        results["LOMO"] = {"n_splits": int(n_mols), "n_partitions": 1,
                           "r2_mean": lomo["r2"], "r2_sd": 0.0,
                           "r2_min": lomo["r2"], "r2_max": lomo["r2"],
                           "rmse_mean": lomo["rmse"], "mae_mean": lomo["mae"],
                           "r2_all": [lomo["r2"]]}
        print(f"k=LOMO  R2 = {lomo['r2']:.4f} (deterministic)  RMSE={lomo['rmse']:.2f}")

        means = [v["r2_mean"] for v in results.values()]
        within = max(v["r2_sd"] for v in results.values())
        out = {
            "note": ("Diagnostic only. RF hyperparameters fixed at the grouped "
                     "winner's; only the fold count and the partition vary. Dev "
                     "set only; locked test never touched."),
            "fixed_params": self.WINNER_PARAMS,
            "n_dev_molecules": int(n_mols), "n_dev_rows": int(len(dev_df)),
            "n_partitions_per_k": self.N_SEEDS,
            "published_k10_dev_cv": published,
            "saved_fold_column_reproduction": repro,
            "reproduces_published": bool(delta < 1e-9),
            "by_k": results,
            "spread_of_k_means": {"min": min(means), "max": max(means),
                                  "range": max(means) - min(means)},
            "max_within_k_sd": within,
        }
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_dir / "kfold_sensitivity.json", "w") as f:
            json.dump(out, f, indent=2)
        print()
        print(f"spread of k means: {min(means):.4f}..{max(means):.4f} "
              f"(range {max(means)-min(means):.4f})")
        print(f"largest within-k sd across partitions: {within:.4f}")
        print("wrote results/kfold_sensitivity.json")
        return out
