from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd

# File paths
INPUT_PATH  = str(ROOT / "data/intermediate/injury_matching/fuzzy_match_ir_past.csv")
OUTPUT_PATH = str(ROOT / "data/intermediate/injury_matching/fuzzy_match_ir_past_with_counts.csv")

# Load the filtered data
df = pd.read_csv(INPUT_PATH, dtype=str)

# Ensure season_ps is numeric for safety
df['season_ps'] = pd.to_numeric(df['season_ps'], errors='coerce')

# Count number of IR entries per player-season based on player_full_name
player_season_counts = df.groupby(['player_full_name', 'season_ps']).size().reset_index(name='number')

# Merge counts into original DataFrame
df_merged = df.merge(player_season_counts, on=['player_full_name', 'season_ps'], how='left')

# Save the updated file
df_merged.to_csv(OUTPUT_PATH, index=False)

# Report
print(f"Merged file with IR count per player-season saved to: {OUTPUT_PATH}")
print(f"Unique player-season pairs: {len(player_season_counts)}")