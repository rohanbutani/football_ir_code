from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd
import re

# ————————————————
# File paths (edit as needed)
BIG_CSV    = str(ROOT / "data/intermediate/player_enrichment/player_season_with_forty_yard.csv")
TIMES_CSV  = str(ROOT / "data/raw/combine_speed/40yd_times.csv")
OUT_CSV    = str(ROOT / "data/intermediate/player_enrichment/player_season_with_forty_yard_merged2.csv")
# ————————————————

def normalize(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if pd.isna(name):
        return ""
    s = name.lower()
    # remove anything that’s not a letter, number, or space
    s = re.sub(r'[^a-z0-9 ]+', '', s)
    # collapse multiple spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# 1. Load datasets
df_big   = pd.read_csv(BIG_CSV)
df_times = pd.read_csv(TIMES_CSV)

# 2. Normalize both name columns
df_big['key_name']   = df_big['playerName'].apply(normalize)
df_times['key_name'] = df_times['Player'].apply(normalize)

# 3. Rename the times column so we don’t overwrite until ready
df_times = df_times.rename(columns={'40yd': 'forty_new'})

# 4. Merge on our normalized key
df_merged = pd.merge(
    df_big,
    df_times[['key_name', 'forty_new']],
    on='key_name',
    how='left'
)

# 5. Fill only the originally missing “forty” entries
df_merged['forty'] = df_merged['forty'].fillna(df_merged['forty_new'])

# 6. Clean up helper cols and save
df_merged = df_merged.drop(columns=['forty_new', 'key_name'])
df_merged.to_csv(OUT_CSV, index=False)

print(f"Done! Merged file saved to {OUT_CSV}")