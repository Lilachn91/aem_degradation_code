"""
Smoke tests for DataCleaner, built entirely on tiny synthetic DataFrames.
NEVER loads data/database_marked.csv or any other real CSV -- these tests
check the class wiring (which columns get dropped/added, shapes), not any
real result.
"""
import pandas as pd

from aem_degradation.data import DataCleaner


def _tiny_raw_df() -> pd.DataFrame:
    n = 6
    df = pd.DataFrame({
        "SMILES": ["CCO", "CCO", "CCO", "CCN", "CCN", "CCN"],
        "Degradation(%)": [10.0, 20.0, 50.0, 5.0, 50.0, 30.0],
        "Concentration": [1.0] * n,
        "Temperature": [60.0] * n,
        "Time(h)": [1, 2, 3, 1, 2, 3],
        "Solvent(MeOD)": [0, 0, 0, 1, 1, 1],
        "Solvent(D2O)": [1, 1, 1, 0, 0, 0],
        "CSP3": [0.5] * n,
        "ALERTS": [0] * n,
        "Pbf": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],       # varies row-to-row (noisy 3D conformer)
        "NPR1": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
        "NPR2": [0.9, 0.9, 0.9, 0.8, 0.8, 0.8],
        "RDKit_Mol_Class": ["<mol object>"] * n,
        "Clusters_5": [1, 1, 1, 2, 2, 2],
        "Clusters_10": [1, 1, 1, 2, 2, 2],
        "Unnamed: 0": range(n),
        "Unnamed: 194": [None] * n,
        "Unnamed: 195": [None] * n,
    })
    for i in range(1, 29):
        df[f"PC{i}"] = 0.0
    return df


def test_clean_drops_junk_columns_and_keeps_row_count(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    out_csv = tmp_path / "prepared.csv"
    _tiny_raw_df().to_csv(raw_csv, index=False)

    cleaner = DataCleaner(source_csv=raw_csv, output_csv=out_csv)
    df = cleaner.clean()

    assert len(df) == 6, "clean() must not drop or collapse rows"
    for col in ["RDKit_Mol_Class", "Clusters_5", "Clusters_10", "Pbf", "NPR1", "NPR2",
               "Unnamed: 0", "Unnamed: 194", "Unnamed: 195", "PC1", "PC28"]:
        assert col not in df.columns, f"{col} should have been dropped"
    for col in ["Pbf_mean", "NPR1_mean", "NPR2_mean", "CSP3", "ALERTS", "is_halflife_proxy"]:
        assert col in df.columns, f"{col} should be present after cleaning"


def test_conformer_mean_is_per_molecule_not_per_row():
    df = _tiny_raw_df()
    cleaner = DataCleaner()
    out = cleaner.add_per_molecule_conformer_means(df.copy())
    cco_mean = out.loc[out["SMILES"] == "CCO", "Pbf_mean"]
    assert cco_mean.nunique() == 1, "Pbf_mean must be constant within a molecule"
    assert abs(cco_mean.iloc[0] - (0.1 + 0.2 + 0.3) / 3) < 1e-9


def test_halflife_proxy_flags_exact_50_percent_rows():
    df = _tiny_raw_df()
    cleaner = DataCleaner()
    out = cleaner.flag_halflife_proxy(df.copy())
    assert out["is_halflife_proxy"].sum() == 2  # the two Degradation(%)==50.0 rows
    assert list(out.loc[out["Degradation(%)"] == 50.0, "is_halflife_proxy"]) == [1, 1]


def test_run_writes_output_csv(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    out_csv = tmp_path / "prepared.csv"
    _tiny_raw_df().to_csv(raw_csv, index=False)
    DataCleaner(source_csv=raw_csv, output_csv=out_csv).run()
    assert out_csv.exists()
    assert len(pd.read_csv(out_csv)) == 6
