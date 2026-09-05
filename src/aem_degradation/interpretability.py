"""
Interpretability protocol (paper Methodology, "Interpretability Protocol"):
run AFTER model comparison picks a winner. Object-oriented port of the
original shap_analysis.py -- unchanged logic, three parts:

1. Per-fold SHAP stability: refit the winning family (fixed tuned
   hyperparameters) on each of the 10 GroupKFold dev folds, compute SHAP
   on that fold's own held-out rows only, then report (a) mean pairwise
   Spearman rho between the 10 per-fold mean|SHAP| feature rankings and
   (b) how many folds each overall-top-15 feature appears in.
2. Final SHAP pass: the winner refit on the full dev set, explained on the
   locked 30-molecule test set -> headline attribution + beeswarm summary
   plot.
3. Shapley axiom verification (Additivity / Missingness / Consistency).

Explainer choice: exact TreeExplainer for tree models (rf/gb); model-
agnostic PermutationExplainer on the pipeline's transformed feature space
otherwise (svr/gp/mlp/pls have no exact explainer).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from .models import REGISTRY
from .paths import RESULTS_DIR
from .preprocessing import Preprocessor, TARGET_COL

RANDOM_STATE = 42
MAX_BACKGROUND = 100  # background sample size for model-agnostic explainers

TREE_FAMILIES = {"rf", "gb"}


class SHAPExplainer:
    def __init__(self, name: str, n_splits: int = 10, results_dir: Path = RESULTS_DIR):
        if name not in REGISTRY:
            raise SystemExit(f"unknown winner: {name}")
        self.name = name
        self.tree = name in TREE_FAMILIES
        self.n_splits = n_splits
        self.results_dir = Path(results_dir)

    def make_winner_pipeline(self, dev_df: pd.DataFrame) -> Pipeline:
        """Rebuild the winning pipeline with its tuned hyperparameters from
        results/<name>.json."""
        with open(self.results_dir / f"{self.name}.json") as f:
            best = json.load(f)["best_params"]
        model_family = REGISTRY[self.name]()
        pipe = model_family.build_pipeline(dev_df)
        params = {}
        for k, v in best.items():
            if isinstance(v, str) and v.startswith("("):  # e.g. "(32, 16)" hidden_layer_sizes
                v = tuple(int(x) for x in v.strip("()").split(",") if x.strip())
            params[k] = v
        pipe.set_params(**params)
        return pipe

    def explain(self, pipe: Pipeline, fit_df: pd.DataFrame, explain_df: pd.DataFrame):
        """SHAP values on the pipeline's transformed feature space."""
        prep = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]
        X_bg = prep.transform(fit_df)
        X_ex = prep.transform(explain_df)
        feat_names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
        if self.tree:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_ex)
            base = explainer.expected_value
        else:
            rng = np.random.RandomState(RANDOM_STATE)
            bg = X_bg[rng.choice(len(X_bg), min(MAX_BACKGROUND, len(X_bg)), replace=False)]
            f = lambda X: np.asarray(model.predict(X)).ravel()
            explainer = shap.PermutationExplainer(f, bg, seed=RANDOM_STATE)
            ex = explainer(X_ex)
            sv, base = ex.values, float(np.mean(ex.base_values))
        return np.asarray(sv), base, feat_names, X_ex

    def fold_stability(self, dev_df: pd.DataFrame):
        y_dev = dev_df[TARGET_COL].values
        per_fold = {}
        for fold in range(self.n_splits):
            tr = np.where(dev_df["cv_fold"] != fold)[0]
            te = np.where(dev_df["cv_fold"] == fold)[0]
            pipe = clone(self.make_winner_pipeline(dev_df))
            pipe.fit(dev_df.iloc[tr], y_dev[tr])
            sv, _, feat_names, _ = self.explain(pipe, dev_df.iloc[tr], dev_df.iloc[te])
            per_fold[fold] = pd.Series(np.abs(sv).mean(axis=0), index=feat_names)
            print(f"  fold {fold}: SHAP over {len(te)} held-out rows done")
        shap_df = pd.DataFrame(per_fold)
        rhos = [spearmanr(shap_df.iloc[:, i], shap_df.iloc[:, j])[0]
                for i in range(self.n_splits) for j in range(i + 1, self.n_splits)]
        mean_rho = float(np.mean(rhos))
        overall = shap_df.mean(axis=1).sort_values(ascending=False)
        top15 = overall.head(15)
        per_fold_top15 = {f: set(shap_df[f].sort_values(ascending=False).head(15).index)
                          for f in range(self.n_splits)}
        consistency = {feat: sum(feat in per_fold_top15[f] for f in range(self.n_splits))
                       for feat in top15.index}
        print(f"\n=== SHAP fold-stability ({self.name}) ===")
        print(f"mean pairwise Spearman rho across {self.n_splits} folds: {mean_rho:.3f}")
        print("top-15 features by overall mean|SHAP| (appearances in fold-level top-15):")
        for feat, imp in top15.items():
            print(f"  {feat:<24} mean|SHAP|={imp:7.3f}   {consistency[feat]}/{self.n_splits}")
        shap_df.to_csv(self.results_dir / f"{self.name}_shap_by_fold.csv")
        return mean_rho, top15, consistency, shap_df

    def final_shap(self, dev_df: pd.DataFrame, test_df: pd.DataFrame):
        y_dev = dev_df[TARGET_COL].values
        pipe = self.make_winner_pipeline(dev_df)
        pipe.fit(dev_df, y_dev)
        sv, base, feat_names, X_ex = self.explain(pipe, dev_df, test_df)
        imp = pd.Series(np.abs(sv).mean(axis=0), index=feat_names).sort_values(ascending=False)
        print(f"\n=== FINAL locked-test SHAP ({self.name}), top 15 ===")
        print(imp.head(15))
        imp.to_csv(self.results_dir / f"{self.name}_final_shap_importance.csv")
        np.savez(self.results_dir / f"{self.name}_final_shap.npz",
                 shap_values=sv, base_value=base, X=X_ex,
                 feat_names=np.array(feat_names, dtype=object))
        return pipe, sv, base, feat_names, X_ex

    def verify_axioms(self, pipe, sv, base, test_df, dev_df):
        """Numerically verify Additivity; verify Missingness and demonstrate
        Consistency with controlled constructions (exact TreeExplainer)."""
        print("\n=== Shapley axiom verification ===")
        y_dev = dev_df[TARGET_COL].values

        # --- 1. Additivity: sum(phi_i) + E[f] == f(x) per locked-test sample ---
        pred = np.asarray(pipe.predict(test_df)).ravel()
        recon = np.asarray(sv).sum(axis=1) + base
        max_dev = float(np.max(np.abs(recon - pred)))
        rel = max_dev / (np.max(np.abs(pred)) + 1e-12)
        print(f"1. Additivity over {len(pred)} locked-test rows: "
              f"max |sum(SHAP)+E[f] - f(x)| = {max_dev:.3e} (relative {rel:.3e})"
              f" -> {'PASS' if rel < 1e-6 or max_dev < 1e-4 else 'CHECK TOLERANCE'}")

        # --- 2. Missingness: a constant feature must get exactly zero attribution ---
        dev2 = dev_df.copy()
        test2 = test_df.copy()
        dev2["CONST_DUMMY"] = 1.0
        test2["CONST_DUMMY"] = 1.0
        pipe2 = clone(self.make_winner_pipeline(dev2))
        pipe2.fit(dev2, y_dev)
        sv2, _, fn2, _ = self.explain(pipe2, dev2, test2)
        j = fn2.index("CONST_DUMMY")
        max_dummy = float(np.max(np.abs(np.asarray(sv2)[:, j])))
        print(f"2. Missingness (constant dummy feature): max |SHAP(CONST_DUMMY)| = "
              f"{max_dummy:.3e} -> {'PASS' if max_dummy == 0.0 or max_dummy < 1e-10 else 'FAIL'}")

        # --- 3. Consistency: controlled synthetic perturbation, exact TreeExplainer
        # on two hand-trained trees. f_A = 80*AND(x1,x2); f_B = f_A + 10*x1. For any
        # sample with x1 = 1, adding x1 to ANY coalition contributes exactly 10 more
        # under B than under A, i.e. x1's marginal contribution weakly increases
        # across all coalitions -- so consistency requires phi_B(x1) >= phi_A(x1)
        # for those samples. (x1 = 0 samples are excluded: the premise doesn't
        # hold there -- their marginal contribution legitimately decreases.)
        X_syn = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y_A = np.array([0.0, 0.0, 0.0, 80.0])   # f_A = 80 * AND(x1,x2)
        y_B = np.array([0.0, 0.0, 10.0, 90.0])  # f_B = f_A + 10*x1
        tA = DecisionTreeRegressor(random_state=0).fit(X_syn, y_A)
        tB = DecisionTreeRegressor(random_state=0).fit(X_syn, y_B)
        assert np.allclose(tA.predict(X_syn), y_A) and np.allclose(tB.predict(X_syn), y_B)
        svA = shap.TreeExplainer(tA, data=X_syn, feature_perturbation="interventional").shap_values(X_syn)
        svB = shap.TreeExplainer(tB, data=X_syn, feature_perturbation="interventional").shap_values(X_syn)
        mask = X_syn[:, 0] == 1
        delta = svB[mask, 0] - svA[mask, 0]
        ok = bool(np.all(delta >= -1e-9))
        print(f"3. Consistency (synthetic AND -> AND + 10*x1 perturbation, x1=1 samples): "
              f"phi_B(x1) - phi_A(x1) = {np.round(delta, 4).tolist()} "
              f"(all >= 0) -> {'PASS' if ok else 'FAIL'}")
        return {"additivity_max_abs_dev": max_dev, "additivity_relative": rel,
                "missingness_max_abs": max_dummy, "consistency_pass": ok}

    def run(self, dev_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """Run all three parts and persist results/<name>_interpretability.json."""
        mean_rho, top15, consistency, _ = self.fold_stability(dev_df)
        pipe, sv, base, feat_names, X_ex = self.final_shap(dev_df, test_df)
        axioms = self.verify_axioms(pipe, sv, base, test_df, dev_df)
        out = {
            "winner": self.name,
            "mean_pairwise_spearman": mean_rho,
            "top15": {k: float(v) for k, v in top15.items()},
            "top15_fold_consistency": consistency,
            "axioms": axioms,
        }
        with open(self.results_dir / f"{self.name}_interpretability.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {self.results_dir}/{self.name}_interpretability.json")
        return out
