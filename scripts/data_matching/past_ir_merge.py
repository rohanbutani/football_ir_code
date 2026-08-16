from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import pandas as pd

# File paths — adjust as needed
PLAYER_SEASON_PATH = str(ROOT / "data/intermediate/player_enrichment/player_season_with_forty_dedup.csv")
IR_COUNT_PATH      = str(ROOT / "data/intermediate/injury_matching/fuzzy_match_ir_past_with_counts.csv")
OUTPUT_PATH        = str(ROOT / "data/intermediate/player_enrichment/player_season_with_ir_count.csv")

# Load datasets
ps_df = pd.read_csv(PLAYER_SEASON_PATH, dtype=str)
ir_df = pd.read_csv(IR_COUNT_PATH, dtype=str)

# Convert season columns to numeric
ps_df['season'] = pd.to_numeric(ps_df['season'], errors='coerce')
ir_df['season_ps'] = pd.to_numeric(ir_df['season_ps'], errors='coerce')

# Prepare and deduplicate IR count data
ir_counts = (
    ir_df[['player_full_name', 'season_ps', 'number']]
    .drop_duplicates(subset=['player_full_name', 'season_ps'])  # <-- deduplicate before merge
    .rename(columns={
        'player_full_name': 'player_clean',
        'season_ps': 'season',
        'number': 'past_ir_count'
    })
)

# Merge on player_clean and season
merged_df = ps_df.merge(ir_counts, on=['player_clean', 'season'], how='left')

# Fill missing IR counts with 0
merged_df['past_ir_count'] = merged_df['past_ir_count'].fillna(0).astype(int)

# Save final output
merged_df.to_csv(OUTPUT_PATH, index=False)

# Report
print(f"✅ Merged file saved to: {OUTPUT_PATH}")
print(f"🔢 Total player-seasons in output: {len(merged_df)}")
print(f"🟠 With past IRs: {(merged_df['past_ir_count'] > 0).sum()}")