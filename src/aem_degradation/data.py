"""
Data cleaning: raw Zou et al. 2023 export -> canonical 896-row measurement
table. Object-oriented port of the original prepare_step1.py; the cleaning
DECISIONS below are unchanged from that script (see project history) --
only the code shape changed, not what it computes.

Decisions encoded here:
- Keep all 896 measurement rows (not collapsed to 150 molecules) -- Time(h)/
  Temperature/Concentration/Solvent(MeOD)/Solvent(D2O) are real input features.
- Target: Degradation(%).
- Drop RDKit_Mol_Class (junk per-row object repr, not a real label).
- Drop precomputed PC1-PC28 / Clusters_5 / Clusters_10 (verified to vary
  row-to-row for identical SMILES -- contaminated by non-deterministic
  3D-conformer descriptors and leaked-in condition columns).
- Drop raw Pbf / NPR1 / NPR2 (non-deterministic per-row 3D conformer
  descriptors), replace with per-molecule means (Pbf_mean / NPR1_mean /
  NPR2_mean).
- Keep CSP3 (deterministic, purely 2D/topological).
- Keep ALERTS (structural-alert count; verified perfectly stable per molecule).
- Drop Unnamed: 0 / Unnamed: 194 / Unnamed: 195 (CSV-export artifacts; the
  row-serial column is dropped as a *feature* specifically because a tree
  model could learn "row number range -> proxy flag" as leakage from file
  structure rather than chemistry).
- Add is_halflife_proxy: flags the 106 rows where Degradation(%) == 50.0
  exactly (literature-reported half-lives re-encoded as a synthetic
  (t_half, 50%) row rather than a real measurement). Kept in the table
  (dropping would remove 52/150 molecules entirely) but flagged so it can
  be excluded/weighted downstream.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import PREPARED_CSV, RAW_CSV

TARGET_COL = "Degradation(%)"


class DataCleaner:
    """Builds the canonical cleaned measurement table from the raw export."""

    CONFORMER_COLS = ["Pbf", "NPR1", "NPR2"]

    DROP_COLS = (
        ["RDKit_Mol_Class"]
        + CONFORMER_COLS
        + [f"PC{i}" for i in range(1, 29)]
        + ["Clusters_5", "Clusters_10", "Unnamed: 0", "Unnamed: 194", "Unnamed: 195"]
    )

    def __init__(self, source_csv: Path = RAW_CSV, output_csv: Path = PREPARED_CSV):
        self.source_csv = Path(source_csv)
        self.output_csv = Path(output_csv)

    def load_raw(self) -> pd.DataFrame:
        df = pd.read_csv(self.source_csv)
        df.columns = [c.strip() for c in df.columns]
        return df

    def check_degradation_range(self, df: pd.DataFrame) -> None:
        """Warn (do not raise) on Degradation(%) values outside [0, 100]."""
        deg_check = pd.to_numeric(df[TARGET_COL], errors="coerce")
        bad_deg = df[(deg_check < 0) | (deg_check > 100)]
        if len(bad_deg):
            print(f"WARNING: {len(bad_deg)} row(s) with {TARGET_COL} outside [0, 100]:")
            for i, row in bad_deg.iterrows():
                print(
                    f"  row {i}: SMILES={row['SMILES']} Concentration={row['Concentration']} "
                    f"Temperature={row['Temperature']} Time(h)={row['Time(h)']} "
                    f"{TARGET_COL}={row[TARGET_COL]}"
                )

    def add_per_molecule_conformer_means(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace noisy per-row 3D-conformer descriptors with per-molecule means."""
        mol_means = df.groupby("SMILES")[self.CONFORMER_COLS].transform("mean")
        for col in self.CONFORMER_COLS:
            df[f"{col}_mean"] = mol_means[col]
        return df

    def flag_halflife_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        deg = pd.to_numeric(df[TARGET_COL], errors="coerce")
        df["is_halflife_proxy"] = (deg == 50.0).astype(int)
        return df

    def clean(self) -> pd.DataFrame:
        """Run the full cleaning pipeline and return the prepared DataFrame
        (does not write to disk -- use .run() for that)."""
        df = self.load_raw()
        self.check_degradation_range(df)
        df = self.add_per_molecule_conformer_means(df)
        df = self.flag_halflife_proxy(df)
        df = df.drop(columns=self.DROP_COLS)
        return df

    def run(self) -> pd.DataFrame:
        """Clean and write database_prepared.csv, mirroring prepare_step1.py's
        printed summary."""
        df = self.clean()
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_csv, index=False)
        print(f"wrote {self.output_csv}")
        print(f"{len(df)} rows x {len(df.columns)} columns")
        print(f"unique molecules: {df['SMILES'].nunique()}")
        print(f"is_halflife_proxy: {df['is_halflife_proxy'].sum()} rows flagged")
        return df


if __name__ == "__main__":
    DataCleaner().run()
