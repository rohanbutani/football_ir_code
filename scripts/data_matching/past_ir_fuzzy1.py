from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import pandas as pd
from rapidfuzz import fuzz, process

# File paths
IR_CSV     = str(ROOT / "data/intermediate/injury_matching/ir_player_names_deduplicated.csv")
PS_CSV     = str(ROOT / "data/intermediate/player_enrichment/player_season_with_forty_yard.csv")
OUTPUT_CSV = str(ROOT / "data/intermediate/player_enrichment/player_season_with_ir_counts.csv")

# Load and clean IR dataset
ir = pd.read_csv(IR_CSV, dtype=str)
ir = ir.loc[ir['player_clean_extracted'].notna()].copy()
ir['Season'] = ir['Season'].astype(int)
ir['player'] = ir['player_clean_extracted'].str.strip()

# Cap to 1 IR visit per player per season
ir = ir.drop_duplicates(subset=['player', 'Season'])

# Load and clean player-season dataset
ps = pd.read_csv(PS_CSV, dtype=str)
ps['Season'] = ps['season'].astype(int)
ps['player'] = ps['playerName'].str.strip()

# Fuzzy match: map player-season names to IR names
ir_players = ir['player'].unique().tolist()
ps_players = ps['player'].unique().tolist()

mapping = {}
THRESHOLD = 80

for name in ps_players:
    match, score, _ = process.extractOne(name, ir_players, scorer=fuzz.token_sort_ratio)
    mapping[name] = match if score >= THRESHOLD else None

ps['ir_player_match'] = ps['player'].map(mapping)

# Count cumulative IR visits per player up through each season
def cumulative_ir_visits(row):
    m = row['ir_player_match']
    if not m:
        return 0
    return int(((ir['player'] == m) & (ir['Season'] <= row['Season'])).sum())

ps['ir_visits_cumulative'] = ps.apply(cumulative_ir_visits, axis=1)

# Save output
ps.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved enriched file to: {OUTPUT_CSV}")