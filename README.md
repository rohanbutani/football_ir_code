# Football IR Research

Organized project structure for injury reserve research workflows.

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

## Notes

This reorganization preserves filenames while moving them into functional groups.
If any script assumes the old flat directory layout, its input/output paths will need to be updated before rerunning.
