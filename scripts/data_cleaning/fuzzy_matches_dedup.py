from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import pandas as pd

# File paths
INPUT_PATH = str(ROOT / "data/intermediate/injury_matching/fuzzy_match_all_with_season7_cleaned.csv")
OUTPUT_PATH = str(ROOT / "data/intermediate/injury_matching/fuzzy_match_all_with_season7_deduped.csv")

# Load CSV
df = pd.read_csv(INPUT_PATH)

# Drop exact duplicates across all columns
df_dedup = df.drop_duplicates()

# Save deduplicated CSV
df_dedup.to_csv(OUTPUT_PATH, index=False)

# Reporting
print(f"Original rows: {len(df)}")
print(f"Deduplicated rows: {len(df_dedup)}")
print(f"Duplicates dropped: {len(df) - len(df_dedup)}")
print(f"Deduplicated file saved to: {OUTPUT_PATH}")