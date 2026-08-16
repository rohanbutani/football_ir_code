from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd

# Paths — adjust as needed
INPUT_CSV  = str(ROOT / "data/intermediate/injury_matching/fuzzy_match_all_with_season7_cleaned.csv")
OUTPUT_CSV = str(ROOT / "data/intermediate/injury_matching/fuzzy_match_season_ir_plus1.csv")

# Load the data
df = pd.read_csv(INPUT_CSV, dtype=str)

# Convert season columns to numeric (coerce errors to NaN)
df['season_ps'] = pd.to_numeric(df['season_ps'], errors='coerce')
df['season_ir'] = pd.to_numeric(df['season_ir'], errors='coerce')

# Drop rows where either season is missing or non‐numeric
df = df.dropna(subset=['season_ps', 'season_ir'])

# Filter for season_ir exactly one greater than season_ps
df_filtered = df[df['season_ir'] == df['season_ps'] + 1]

# Save the filtered result
df_filtered.to_csv(OUTPUT_CSV, index=False)

print(f"Filtered data saved to {OUTPUT_CSV}, {len(df_filtered)} rows match season_ir = season_ps + 1.")