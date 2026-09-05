"""
Compare the winner's final locked-test SHAP importance ranking against the
original study's XGBoost-regression descriptor ranking (Zou et al. 2023,
Fig. 3a top-20). Object-oriented port of compare_zou.py / zou_top20_ranking.py.

Apples-to-apples handled explicitly: Zou ranked MOLECULAR DESCRIPTORS only
(their formulation has no experimental-condition inputs), so condition
features (Time/Concentration/solvent/Time_log1p) are excluded on our side
before comparing. Pbf_mean/NPR1_mean/NPR2_mean are our per-molecule-averaged
versions of Zou's Pbf/NPR1/NPR2 columns and are mapped back for the overlap
count.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import RESULTS_DIR

#: Zou et al. 2023 (Angew. Chem. Int. Ed. 62, e202300388) Figure 3a:
#: XGBoost-regression feature-importance ranking of their molecular
#: descriptors (top 20 shown in the figure, with printed importances).
#: Transcribed directly from the published figure.
ZOU_TOP20 = [
    ("C_count", 0.2533),
    ("N_count", 0.1856),
    ("O_count", 0.1644),
    ("MaxEStateIndex", 0.0551),
    ("MinEStateIndex", 0.0430),
    ("qed", 0.0340),
    ("MolWt", 0.0220),
    ("HeavyAtomMolWt", 0.0178),
    ("MaxPartialCharge", 0.0145),
    ("MaxAbsPartialCharge", 0.0117),
    ("MinAbsPartialCharge", 0.0100),
    ("FpDensityMorgan2", 0.0084),
    ("BCUT2D_MWLOW", 0.0084),
    ("BCUT2D_CHGHI", 0.0081),
    ("BCUT2D_CHGLO", 0.0077),
    ("BCUT2D_MRLOW", 0.0076),
    ("BertzCT", 0.0071),
    ("Chi0", 0.0069),
    ("Chi0v", 0.0067),
    ("Chi3v", 0.0063),
]

CONDITION_FEATURES = {"Time(h)", "Time_log1p", "Concentration", "Temperature",
                      "Solvent(MeOD)", "Solvent(D2O)"}
RENAME = {"Pbf_mean": "Pbf", "NPR1_mean": "NPR1", "NPR2_mean": "NPR2"}


class ZouBenchmark:
    def __init__(self, results_dir: Path = RESULTS_DIR, top_n: int = 15):
        self.results_dir = Path(results_dir)
        self.top_n = top_n
        self.zou_top = [n for n, _ in ZOU_TOP20]

    def compare(self, winner: str) -> dict:
        imp = pd.read_csv(self.results_dir / f"{winner}_final_shap_importance.csv",
                          index_col=0).iloc[:, 0]
        mol_imp = imp[~imp.index.isin(CONDITION_FEATURES)].rename(index=RENAME)
        ours_top = list(mol_imp.sort_values(ascending=False).head(self.top_n).index)
        overlap = [n for n in ours_top if n in self.zou_top]
        condition_share = imp[imp.index.isin(CONDITION_FEATURES)].sum() / imp.sum()

        print(f"our top-{self.top_n} molecular descriptors (condition features excluded): {ours_top}")
        print(f"Zou Fig.3a top-20: {self.zou_top}")
        print(f"overlap ({len(overlap)}/{self.top_n}): {overlap}")
        print()
        print(f"share of |SHAP| carried by experimental-condition features "
              f"(Time/Temperature/Concentration/solvent): {condition_share:.1%}")
        sub_overlaps = {}
        for k in (5, 10):
            ov = [f for f in ours_top[:k] if f in self.zou_top]
            sub_overlaps[k] = ov
            print(f"top-{k} overlap with Zou top-20: {len(ov)}/{k} {ov}")
        return {
            "winner": winner,
            "our_top": ours_top,
            "zou_top20": self.zou_top,
            "overlap": overlap,
            "n_overlap": len(overlap),
            "sub_overlaps": sub_overlaps,
            "condition_feature_share_of_shap": float(condition_share),
        }
