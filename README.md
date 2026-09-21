# AEM Cationic Degradation — ML Pipeline

Predicting anion-exchange-membrane (AEM) cationic functional-group
degradation (%) from RDKit molecular descriptors plus experimental
conditions (time, temperature, concentration, solvent), on the dataset of
Zou et al. 2023 (*Angew. Chem. Int. Ed.* 62, e202300388).

Final project codebase for *Machine Learning for Chemical Engineering*
(Track 1, Technion, Dr. Barak Or), accompanying the 4-page project paper.

## What this is

An object-oriented Python package (`src/aem_degradation/`) implementing the
full pipeline described in the paper's Methodology: data cleaning →
molecule-grouped splitting → per-family hyperparameter search + pooled
out-of-fold evaluation + one locked-test evaluation → SHAP interpretability
→ comparison against Zou et al.'s published feature ranking → two
robustness diagnostics → the three paper figures. It is a direct,
class-based port of an earlier set of flat analysis scripts; the underlying
algorithms, hyperparameter grids, random seeds, and column decisions are
unchanged from what actually produced the numbers reported in the paper —
only the code's shape changed.

**`results/` ships the actual JSON/NPZ/CSV/PDF artifacts of the run that
produced every number in the paper.** Re-running the pipeline (see below)
regenerates these files in place; it does not fabricate anything the
checked-in files don't already contain.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or conda, etc.
pip install -r requirements.txt
```

Tested against Python 3.11 with the pinned versions in `requirements.txt`.
No RDKit or Excel dependency is required — the molecular descriptors are
already computed in the shipped CSVs (`data/`); this package only consumes
them.

## Running the pipeline

Everything goes through one entry point, `scripts/run_pipeline.py`, from the
repo root:

```bash
# 1. raw export -> cleaned 896-row table (data/database_prepared.csv)
python scripts/run_pipeline.py clean

# 2. build the three split columns on top of the cleaned table
#    (data/database_with_splits.csv)
python scripts/run_pipeline.py split

# 3. train one or more model families: grid search (10 saved GroupKFold
#    dev folds) -> pooled out-of-fold dev-CV score -> one locked-test score
python scripts/run_pipeline.py train                  # all six families
python scripts/run_pipeline.py train --models rf gb   # a subset

# 4. interpretability protocol for the winning family (default: rf) --
#    per-fold SHAP stability, final SHAP on the locked test set, and three
#    Shapley axiom checks (additivity / missingness / consistency)
python scripts/run_pipeline.py interpret --winner rf

# 5. compare the winner's SHAP ranking against Zou et al.'s published
#    Fig. 3a top-20 XGBoost feature-importance ranking
python scripts/run_pipeline.py benchmark --winner rf

# 6. two protocol-adjacent robustness diagnostics (never touch the locked
#    test set, never overwrite the winner's real result files):
#      - row-level vs. molecule-grouped split: quantifies how much of the
#        apparent score a naive (non-grouped) split would manufacture
#      - k-fold sensitivity: how much the pooled dev-CV score moves with
#        the number of GroupKFold folds and with which partition is drawn
#        at a given fold count, holding the winner's hyperparameters fixed
python scripts/run_pipeline.py diagnostics

# 7. regenerate the three paper figures (fig_parity, fig_shap_summary,
#    fig_shap_stability) from whatever is currently in results/
python scripts/run_pipeline.py figures --winner rf

# ...or all of the above, in order, for the default winner:
python scripts/run_pipeline.py all
```

Each stage reads/writes exactly the files the paper's Methodology
describes (see `src/aem_degradation/paths.py`): `data/database_prepared.csv`,
`data/database_with_splits.csv`, and one `results/<family>.json` +
`results/<family>_preds.npz` per model family, plus the SHAP/benchmark/
diagnostic/figure artifacts layered on top of those.

## Reproducibility note (read before treating a re-run as a discrepancy)

`data/database_with_splits.csv` already contains the exact `cv_fold` and
`test_holdout` assignments used to produce every number in the paper, and
`scripts/run_pipeline.py train` reads that saved column rather than
recomputing it — so re-running `train` on the shipped data reproduces the
paper's numbers exactly.

`scripts/run_pipeline.py split`, however, calls `GroupKFold` itself to
*build* that column from scratch. `GroupKFold`'s tie-breaking behavior for
equal-sized groups is not guaranteed stable across scikit-learn versions.
This package is pinned to `scikit-learn==1.9.0` (also the version the
published run used), and under that version `split` regenerates the
checked-in `database_with_splits.csv` byte-for-byte. Under a materially
different scikit-learn version, a fresh `split` run can assign a small
number of molecules to different folds than the shipped file, which can
shift the pooled dev-CV score by roughly ±0.05 R² — the locked-test split
(`GroupShuffleSplit`) and the reported locked-test numbers are not affected
by this, only the dev-CV fold assignment is. We are stating this plainly
rather than hiding it: if you need the paper's exact numbers, use the
shipped `data/database_with_splits.csv` (the default `train` behavior)
rather than regenerating it under a different environment.

## Tests (optional)

`tests/` has smoke tests for every class's wiring (`DataCleaner`,
`SplitBuilder`, `Preprocessor`, all six `ModelFamily` subclasses,
`RowLevelDiagnostic`, `KFoldSensitivity`, `SHAPExplainer`, `ZouBenchmark`,
`FigureGenerator`). They build tiny synthetic DataFrames in-memory and never
read anything under `data/` — they check that the code runs and wires
together correctly, not that any particular number comes out a certain way.

```bash
pip install pytest
pytest tests/
```

## Package layout

```
src/aem_degradation/
    paths.py            # central, portable path resolution
    data.py             # DataCleaner        -- raw export -> cleaned table
    preprocessing.py    # Preprocessor       -- per-fold StandardScaler, never fit on the full table
    splits.py           # SplitBuilder       -- the three split columns
    protocol.py         # ExperimentProtocol -- shared grid-search -> OOF -> locked-test sequence
    models/
        base.py             # ModelFamily (abstract base every family implements)
        pls.py              # PLSModel
        random_forest.py    # RandomForestModel
        gradient_boosting.py# GradientBoostingModel
        svr.py              # SVRModel
        gaussian_process.py # GaussianProcessModel
        mlp.py              # MLPModel
    interpretability.py # SHAPExplainer      -- fold stability, final SHAP, axiom checks
    benchmarking.py     # ZouBenchmark       -- comparison against Zou et al. Fig. 3a
    diagnostics.py       # RowLevelDiagnostic, KFoldSensitivity -- robustness checks, not the protocol
    figures.py           # FigureGenerator    -- the three paper figures
scripts/
    run_pipeline.py      # CLI entry point tying the above together
data/                   # raw + derived CSVs (see below)
results/                # JSON/NPZ/CSV/PDF artifacts (checked in as reference)
tests/                  # synthetic-data smoke tests (never touch data/)
```

Every model family subclasses `ModelFamily` and implements `build_pipeline`
(its `[("prep", ...), ("model", ...)]` sklearn `Pipeline`) and `param_grid`;
`ModelFamily.run()` then drives the identical grid-search → pooled-OOF →
locked-test sequence for all six families via `ExperimentProtocol`, so no
family gets a different evaluation procedure. `GaussianProcessModel`
overrides `default_n_jobs` down from the base's `-1` to `2`, since its
kernel-matrix fits are O(n³) and each worker holds a large matrix in memory.

## Data files

| file | rows | description |
|---|---|---|
| `data/database_marked_original.csv` | 896 | Zou et al.'s raw export, columns stripped only |
| `data/database_marked.csv` | 896 | working copy of the raw export used as `DataCleaner`'s input |
| `data/database_prepared.csv` | 896 | after `DataCleaner`: conformer descriptors averaged per molecule, junk/leaky columns dropped, `is_halflife_proxy` added |
| `data/database_with_splits.csv` | 896 | `database_prepared.csv` + the three split columns (`random_split_diagnostic`, `test_holdout`, `cv_fold`) |

150 unique molecules span the 896 rows (each molecule measured at several
time/temperature/concentration/solvent conditions), grouped 120/30
dev/test by `GroupShuffleSplit(random_state=42)` on SMILES.

## Results

`results/` ships, per model family, `<name>.json` (best hyperparameters,
pooled dev-CV R²/RMSE/MAE, locked-test R²/RMSE/MAE) and `<name>_preds.npz`
(the raw out-of-fold and test predictions). Headline dev-CV / locked-test
R² for each family (see `results/<name>.json` for full metrics):

| family | dev-CV R² | locked-test R² |
|---|---|---|
| Random Forest (**winner**) | 0.360 | 0.436 |
| Gradient Boosting | 0.251 | 0.447 |
| Support Vector Regression | 0.256 | 0.356 |
| Gaussian Process | 0.184 | 0.389 |
| MLP | −2.86 | 0.447 |
| PLS | −2.75 | −0.009 |

PLS and MLP collapse to a negative pooled dev-CV R² (worse than predicting
the dev-set mean) despite the grid search choosing a low-complexity
configuration for each (PLS: 2 latent components; MLP: strong L2
regularization) — see the paper's Results for the discussion. Random
Forest is the interpretability winner (`results/rf_interpretability.json`,
`results/rf_final_shap_importance.csv`, `results/rf_shap_by_fold.csv`):
mean pairwise Spearman ρ of 0.87 across the 10 dev folds' top-feature
rankings, and all three Shapley axiom checks pass (additivity residual
~1e-13, missingness residual exactly 0, consistency check passes on the
synthetic AND-function pair).

`results/rf_rowlevel_diagnostic.json` quantifies why the split must be
molecule-grouped: a naive row-level 80/20 split lets 72/150 molecules
(48%) appear on both sides, inflating RF's dev-CV R² from 0.36 to 0.84 and
its test R² from 0.44 to 0.83 — over 0.4 R² of pure leakage, not signal.
`results/kfold_sensitivity.json` shows the pooled dev-CV R² moving by up to
~0.12 across fold counts 3–20 (10 random partitions each), which is the
basis for the ±0.05 R² caveat above.

## What is *not* included

Presenting only molecular-descriptor SHAP rankings against Zou et al.'s
Fig. 3a required excluding the experimental-condition features (time,
temperature, concentration, solvent) from that one comparison — they are
still part of every model's actual feature set and are reported separately
as a "share of |SHAP|" figure in `benchmarking.py`'s output. A third split
strategy (train on some chemical families, test on a held-out family) is
listed in `splits.py`'s docstring as future work and is not implemented
here.
