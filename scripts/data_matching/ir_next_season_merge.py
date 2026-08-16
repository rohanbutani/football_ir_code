from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd

# ─── CONFIGURE PATHS ───────────────────────────────────────────────────────────
MAIN_CSV   = str(ROOT / "data/intermediate/player_enrichment/player_season_with_ir_count.csv")             # your big dataset
FUZZY_CSV  = str(ROOT / "data/intermediate/injury_matching/fuzzy_match_season_ir_plus1.csv")            # fuzzy‐match results
OUTPUT_CSV = str(ROOT / "data/intermediate/player_enrichment/player_season_with_ir_count_next_season_ir.csv")  # merged output

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df_main  = pd.read_csv(MAIN_CSV)
df_fuzzy = pd.read_csv(FUZZY_CSV)

# ─── BUILD next_season_ir FLAG ───────────────────────────────────────────────
# fuzzy file uses 'player_two_words' (matches your df_main.player_clean)
# and 'season_ps'  (matches your df_main.season)
df_ir_next = (
    df_fuzzy
    [['player_full_name', 'season_ps']]
    .drop_duplicates()
    .rename(columns={
        'player_full_name': 'player_clean',
        'season_ps':       'season'
    })
)
df_ir_next['next_season_ir'] = 1

# ─── MERGE INTO MAIN DATASET ─────────────────────────────────────────────────
df_merged = pd.merge(
    df_main,
    df_ir_next,
    how   = 'left',
    on    = ['player_clean', 'season']
)

# missing → no IR next season → fill with 0
df_merged['next_season_ir'] = df_merged['next_season_ir'].fillna(0).astype(int)

# ─── SAVE RESULT ─────────────────────────────────────────────────────────────
df_merged.to_csv(OUTPUT_CSV, index=False)
print(f'Merged dataset with next_season_ir saved to: {OUTPUT_CSV}')