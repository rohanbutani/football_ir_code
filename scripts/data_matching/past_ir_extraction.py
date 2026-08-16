from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import pandas as pd

# File paths – adjust as needed
INPUT_PATH  = str(ROOT / "data/intermediate/injury_matching/fuzzy_match_all_with_season7_deduped.csv")
OUTPUT_PATH = str(ROOT / "data/intermediate/injury_matching/fuzzy_match_ir_past.csv")

# Load the data
df = pd.read_csv(INPUT_PATH, dtype=str)

# Make sure the season columns are numeric
df['season_ir'] = pd.to_numeric(df['season_ir'], errors='coerce')
df['season_ps'] = pd.to_numeric(df['season_ps'], errors='coerce')

# Filter: keep only rows where season_ir < season_ps
df_filtered = df[df['season_ir'] < df['season_ps']].copy()

# Save the result
df_filtered.to_csv(OUTPUT_PATH, index=False)

# Report
print(f"Original rows:  {len(df)}")
print(f"Filtered rows:  {len(df_filtered)}")
print(f"Rows dropped:   {len(df) - len(df_filtered)}")
print(f"Filtered file saved to: {OUTPUT_PATH}")