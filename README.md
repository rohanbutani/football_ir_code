# Football IR Research

Repository for the football injured reserve research pipeline, focused on season-ahead prediction of injured reserve placement for NFL wide receivers and tight ends.

This repo contains:

- source and derived datasets used across the project
- collection, cleaning, matching, feature-engineering, and analysis scripts
- paper-adjacent materials such as abstracts, templates, and supplementary artifacts

The working style of this repository is research-first rather than package-first: most scripts are runnable entry points that read and write CSV artifacts through the project data directories.

## Top-level layout

- `scripts/`: code grouped by workflow stage
  - `data_collection/`: scraping and source ingestion
  - `data_cleaning/`: cleanup, normalization, deduplication
  - `data_matching/`: fuzzy matching and IR-history linkage
  - `feature_engineering/`: dataset enrichment and feature joins
  - `analysis/`: EDA, diagnostics, and modeling entry points
- `data/`: datasets grouped by maturity
  - `raw/`: source and near-source extracts
  - `intermediate/`: cleaned and joined working tables
  - `processed/final_datasets/`: final modeling-ready outputs
- `docs/`: abstracts, templates, and supplementary material
- `artifacts/`: debug files and archives
- `tools/`: local helper tooling and config

## Data layout

- `data/raw/`
  - near-source exports and externally collected tables
  - examples: injury reserve transactions, Next Gen Stats exports, snap counts, travel and surface context, combine/pro-day inputs
- `data/intermediate/`
  - cleaned, matched, or partially enriched datasets produced by pipeline steps
  - examples: IR name extraction outputs, fuzzy-match tables, merged enrichment tables
- `data/processed/final_datasets/`
  - final modeling-ready datasets

## Script layout

- `scripts/data_collection/`
  - pull or derive source data from external systems
- `scripts/data_cleaning/`
  - normalize names, deduplicate records, and patch source inconsistencies
- `scripts/data_matching/`
  - fuzzy matching and IR-history linkage
- `scripts/feature_engineering/`
  - add contextual features such as EPA, SOS, travel, turf exposure, height, weight, age, and 40-yard metrics
- `scripts/analysis/`
  - EDA, diagnostics, and modeling entry points

## Typical pipeline flow

At a high level, the project flows like this:

1. collect raw source data into `data/raw/`
2. clean injury and player identity fields
3. fuzzy-match IR records to player-season data
4. merge feature sources into enriched intermediate tables
5. produce modeling-ready outputs in `data/processed/final_datasets/`
6. run EDA and modeling scripts

## Environment expectations

Most scripts are plain Python entry points, but several require optional third-party packages depending on the task. The code now fails with explicit messages when major optional dependencies are missing.

Common dependencies used in this repo include:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `beautifulsoup4`
- `requests`
- `tensorflow`
- `xgboost`
- `imblearn`
- `scikeras`
- `rapidfuzz` (optional; some matching scripts now fall back to standard-library matching)
- `tqdm` (optional)
- `nfl_data_py` (required for scripts that pull roster, schedule, snap-count, combine, or play-by-play data)

## Operational notes

- Scripts generally resolve paths relative to the repository root.
- Many scripts write outputs back into tracked CSVs under `data/intermediate/` or `data/processed/`.
- Running scripts can therefore modify checked-in artifacts. Review `git status` before committing.
- Some analysis files at the repository root are legacy entry points retained for compatibility with earlier work.

## Documents

- `docs/abstracts/` contains shorter abstract-format research materials.
- `docs/supplementary/` contains supplementary modeling materials and related artifacts.

The repository has been refactored so scripts now resolve data paths relative to the repository root.
