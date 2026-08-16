from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd

# Paths — adjust as needed
INPUT_CSV  = str(ROOT / "data/intermediate/injury_matching/fuzzy_match_all_with_season7.csv")
OUTPUT_CSV = str(ROOT / "data/intermediate/injury_matching/fuzzy_match_all_with_season7_cleaned.csv")

# Load the data
df = pd.read_csv(INPUT_CSV, dtype=str)

# Treat empty strings as missing
df = df.replace(r'^\s*$', pd.NA, regex=True)

# Drop rows where all columns are missing/empty
df_clean = df.dropna(how='all')

# Save the result
df_clean.to_csv(OUTPUT_CSV, index=False)

print(f"Cleaned data saved to {OUTPUT_CSV}, {len(df_clean)} rows remain.")