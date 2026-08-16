from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

# File paths
IR_CSV     = str(ROOT / "data/intermediate/injury_matching/ir_player_names_truncated.csv")
PS_CSV     = str(ROOT / "data/intermediate/player_enrichment/player_season_with_forty_yard.csv")
MATCH_LOG  = str(ROOT / "data/intermediate/injury_matching/fuzzy_match_all_with_season6.csv")

# Parameters
THRESHOLD = 80


# Load and clean IR player names (keep the Season column)
ir = pd.read_csv(IR_CSV, dtype=str)
ir = ir[ir['player_clean_cleaned'].notna()].copy()
ir['player_cleaned'] = ir['player_clean_cleaned'].str.strip()

# Load and clean player-season names and seasons
ps = pd.read_csv(PS_CSV, dtype=str)
ps['player_clean'] = ps['player_clean'].str.strip()
ps['season_ps'] = ps['season'].astype(int)

# Prepare results
matches = []

# Fuzzy match with progress bar, now capturing both seasons
print("🔍 Starting fuzzy matching…")
for _, ps_row in tqdm(ps.iterrows(), total=len(ps), desc="🔄 Matching players"):
    ps_name   = ps_row['player_clean']
    season_ps = ps_row['season_ps']

    for _, ir_row in ir.iterrows():
        ir_name   = ir_row['player_cleaned']
        season_ir = int(ir_row['Season'])        # your IR CSV’s year column

        score = fuzz.token_sort_ratio(ps_name, ir_name)
        if score >= THRESHOLD:
            matches.append({
                'player_season_name': ps_name,
                'season_ps':          season_ps,
                'ir_name':            ir_name,
                'season_ir':          season_ir,
                'fuzzy_score':        score
            })

# Save result
match_df = pd.DataFrame(matches)
match_df.to_csv(MATCH_LOG, index=False)

print(f"✅ Saved {len(match_df)} fuzzy matches with score ≥ {THRESHOLD} to:\n   {MATCH_LOG}")