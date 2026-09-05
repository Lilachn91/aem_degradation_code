"""
Central path resolution so every module works from a repo checkout on any
machine (no absolute paths tied to a particular home directory).

Layout assumed:
    <repo_root>/
        data/       raw + derived CSVs (checked in, see README)
        results/    JSON/NPZ artifacts + figures (checked in as reference)
        src/aem_degradation/paths.py   <- this file
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_CSV = DATA_DIR / "database_marked.csv"
RAW_CSV_ORIGINAL = DATA_DIR / "database_marked_original.csv"
PREPARED_CSV = DATA_DIR / "database_prepared.csv"
SPLITS_CSV = DATA_DIR / "database_with_splits.csv"
