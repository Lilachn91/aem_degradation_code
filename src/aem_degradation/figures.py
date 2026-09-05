"""
Paper figures for the Results section, consuming results/ artifacts.
Object-oriented port of make_figures.py -- unchanged logic/styling:

  1. fig_parity.pdf         -- winner: pooled dev-CV OOF + locked-test parity.
  2. fig_shap_summary.pdf   -- winner: SHAP beeswarm on the locked test set.
  3. fig_shap_stability.pdf -- winner: mean|SHAP| of overall-top-15 features
                                across the 10 GroupKFold dev folds (heatmap).

Sized for an IEEE two-column page (~3.45 in column width).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import RESULTS_DIR

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7})


class FigureGenerator:
    def __init__(self, results_dir: Path = RESULTS_DIR, outdir: Path | None = None):
        self.results_dir = Path(results_dir)
        self.outdir = Path(outdir) if outdir is not None else self.results_dir
        self.outdir.mkdir(parents=True, exist_ok=True)

    def parity(self, winner: str) -> Path:
        d = np.load(self.results_dir / f"{winner}_preds.npz")
        fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.9), sharex=True, sharey=True)
        for ax, (yt, yp, title) in zip(axes, [
                (d["y_dev"], d["oof_pred"], "dev CV (out-of-fold)"),
                (d["y_test"], d["test_pred"], "locked test")]):
            ax.scatter(yt, yp, s=6, alpha=0.45, edgecolors="none", color="#1f77b4")
            lim = [-5, 105]
            ax.plot(lim, lim, "k--", lw=0.7)
            ax.set_xlim(lim)
            ax.set_ylim(min(lim[0], float(yp.min()) - 5), max(lim[1], float(yp.max()) + 5))
            ax.set_title(title, fontsize=8)
        axes[0].set_ylabel("predicted (%)")
        fig.supxlabel("measured Degradation (%)", fontsize=8)
        fig.tight_layout()
        out = self.outdir / "fig_parity.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.name}")
        return out

    def shap_summary(self, winner: str) -> Path:
        import shap
        f = np.load(self.results_dir / f"{winner}_final_shap.npz", allow_pickle=True)
        sv, X, feat_names = f["shap_values"], f["X"], list(f["feat_names"])
        fig = plt.figure(figsize=(3.45, 3.2))
        shap.summary_plot(sv, X, feature_names=feat_names, max_display=12,
                          show=False, plot_size=None)
        plt.gcf().set_size_inches(3.45, 3.2)
        plt.tight_layout()
        out = self.outdir / "fig_shap_summary.pdf"
        plt.savefig(out, bbox_inches="tight")
        plt.close("all")
        print(f"wrote {out.name}")
        return out

    def shap_stability_heatmap(self, winner: str) -> Path:
        shap_df = pd.read_csv(self.results_dir / f"{winner}_shap_by_fold.csv", index_col=0)
        top15 = shap_df.mean(axis=1).sort_values(ascending=False).head(15).index
        M = shap_df.loc[top15]
        fig, ax = plt.subplots(figsize=(3.45, 2.6))
        im = ax.imshow(M.values, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(top15)), top15)
        ax.set_xticks(range(M.shape[1]), [str(i) for i in range(M.shape[1])])
        ax.set_xlabel("GroupKFold dev fold")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("mean |SHAP| (Degradation %)", fontsize=7)
        fig.tight_layout()
        out = self.outdir / "fig_shap_stability.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.name}")
        return out

    def run(self, winner: str = "rf") -> list[Path]:
        return [self.parity(winner), self.shap_summary(winner), self.shap_stability_heatmap(winner)]
