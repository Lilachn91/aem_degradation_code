"""
Shared step-4 protocol runner: every model family goes through the
IDENTICAL sequence (object-oriented port of the original protocol.py; the
sequence itself is unchanged):

  1. GridSearchCV with cv = the 10 saved GroupKFold dev folds (cv_fold in
     database_with_splits.csv) -> ONE fixed hyperparameter set per model
     family ("option 2" lightweight tuning, not nested CV).
  2. Manual 10-fold loop with those fixed hyperparameters -> pooled
     out-of-fold predictions over all 120 dev molecules -> one honest
     dev-CV R2/RMSE/MAE per family.
  3. GridSearchCV's best_estimator_ (already refit on the full dev set)
     evaluated exactly once on the locked 30-molecule test_holdout.

Per the paper's Methodology: for scale-sensitive families (PLS/SVR/GP/MLP),
log1p(Time(h)) is supplied as an ADDITIONAL feature alongside the raw
Time(h) value (target untouched). Tree ensembles get raw Time(h) only.

Each run writes results/<name>.json (metrics + best params) and
results/<name>_preds.npz (OOF + locked-test predictions) so the
figure/table step can consume every family uniformly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from .paths import RESULTS_DIR, SPLITS_CSV
from .preprocessing import TARGET_COL


class ExperimentProtocol:
    """Runs the shared grid-search -> pooled-OOF -> locked-test sequence for
    one model family and persists its results."""

    def __init__(self, data_csv: Path = SPLITS_CSV, results_dir: Path = RESULTS_DIR,
                 n_splits: int = 10):
        self.data_csv = Path(data_csv)
        self.results_dir = Path(results_dir)
        self.n_splits = n_splits

    def load_data(self, log_time: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(self.data_csv)
        if log_time:
            # extra input feature alongside raw Time(h); Degradation(%) untouched
            df["Time_log1p"] = np.log1p(df["Time(h)"])
        dev_df = df[df["test_holdout"] == "dev"].reset_index(drop=True)
        test_df = df[df["test_holdout"] == "test"].reset_index(drop=True)
        return dev_df, test_df

    def _custom_cv(self, dev_df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            (np.where(dev_df["cv_fold"] != f)[0], np.where(dev_df["cv_fold"] == f)[0])
            for f in range(self.n_splits)
        ]

    def run(self, name: str, base_pipeline, param_grid: dict,
            dev_df: pd.DataFrame, test_df: pd.DataFrame, n_jobs: int = -1) -> tuple[GridSearchCV, dict]:
        y_dev = dev_df[TARGET_COL].values
        y_test = test_df[TARGET_COL].values
        custom_cv = self._custom_cv(dev_df)

        # ---- step 1: inner CV grid search fixes one hyperparameter set ----
        t0 = time.time()
        grid = GridSearchCV(base_pipeline, param_grid, cv=custom_cv,
                            scoring="neg_root_mean_squared_error", n_jobs=n_jobs)
        grid.fit(dev_df, y_dev)
        t_tune = time.time() - t0
        print(f"[{name}] best params: {grid.best_params_}")
        print(f"[{name}] best CV RMSE (mean of {self.n_splits} fold scores): "
              f"{-grid.best_score_:.2f}  (tuning took {t_tune:.0f}s)")

        # ---- step 2: pooled out-of-fold predictions, hyperparameters FIXED ----
        oof_pred = np.full(len(dev_df), np.nan)
        for fold in range(self.n_splits):
            train_i = np.where(dev_df["cv_fold"] != fold)[0]
            test_i = np.where(dev_df["cv_fold"] == fold)[0]
            pipe = clone(base_pipeline).set_params(**grid.best_params_)
            pipe.fit(dev_df.iloc[train_i], y_dev[train_i])
            oof_pred[test_i] = np.asarray(pipe.predict(dev_df.iloc[test_i])).ravel()
        assert not np.isnan(oof_pred).any()

        r2 = r2_score(y_dev, oof_pred)
        rmse = mean_squared_error(y_dev, oof_pred) ** 0.5
        mae = mean_absolute_error(y_dev, oof_pred)
        n_dev_mol = dev_df["SMILES"].nunique()
        print(f"[{name}] pooled dev-CV ({n_dev_mol} molecules): R2 = {r2:.3f}  RMSE = {rmse:.2f}  MAE = {mae:.2f}")

        # ---- step 3: locked test, touched exactly once ----
        final_pred = np.asarray(grid.best_estimator_.predict(test_df)).ravel()
        r2_t = r2_score(y_test, final_pred)
        rmse_t = mean_squared_error(y_test, final_pred) ** 0.5
        mae_t = mean_absolute_error(y_test, final_pred)
        n_test_mol = test_df["SMILES"].nunique()
        print(f"[{name}] LOCKED TEST ({n_test_mol} molecules):    R2 = {r2_t:.3f}  RMSE = {rmse_t:.2f}  MAE = {mae_t:.2f}")

        self.results_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "name": name,
            "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                            for k, v in grid.best_params_.items()},
            "grid_best_cv_rmse": float(-grid.best_score_),
            "dev_cv": {"r2": float(r2), "rmse": float(rmse), "mae": float(mae)},
            "locked_test": {"r2": float(r2_t), "rmse": float(rmse_t), "mae": float(mae_t)},
            "tuning_seconds": round(t_tune, 1),
        }
        with open(self.results_dir / f"{name}.json", "w") as f:
            json.dump(out, f, indent=2)
        np.savez(self.results_dir / f"{name}_preds.npz",
                 y_dev=y_dev, oof_pred=oof_pred, y_test=y_test, test_pred=final_pred)
        return grid, out
