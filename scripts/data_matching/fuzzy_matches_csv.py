import pandas as pd
from rapidfuzz import fuzz

# File paths
IR_CSV     = 'C:/Users/rohan/ir_player_names_deduplicated.csv'
PS_CSV     = 'C:/Users/rohan/player_season_with_forty_yard.csv'
MATCH_LOG  = 'C:/Users/rohan/fuzzy_match_all_with_season.csv'

# Parameters
THRESHOLD = 80

# Load and clean IR player names
ir = pd.read_csv(IR_CSV, dtype=str)
ir = ir[ir['player_clean_extracted'].notna()].copy()
ir_names = ir['player_clean_extracted'].str.strip().unique()

# Load and clean player-season names and seasons
ps = pd.read_csv(PS_CSV, dtype=str)
ps['playerName'] = ps['playerName'].str.strip()
ps['season'] = ps['season'].astype(int)

# Prepare results
matches = []

# For each (player, season) pair, match to all IR names
for _, row in ps.iterrows():
    ps_name = row['playerName']
    season = row['season']
    for ir_name in ir_names:
        score = fuzz.token_sort_ratio(ps_name, ir_name)
        if score >= THRESHOLD:
            matches.append({
                'player_season_name': ps_name,
                'season': season,
                'ir_name': ir_name,
                'fuzzy_score': score
            })

# Save result
match_df = pd.DataFrame(matches)
match_df.to_csv(MATCH_LOG, index=False)

print(f"✅ Saved {len(match_df)} fuzzy matches with score ≥ {THRESHOLD} to:\n   {MATCH_LOG}")