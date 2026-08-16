from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd
import nfl_data_py as nfl

# --- Step 1: Load your uploaded file ---
df = pd.read_csv(str(ROOT / "data/intermediate/player_enrichment/nextgen_with_teams_birthdates_and_travel.csv"))

# --- Step 2: Drop the current birth_date column ---
if 'birth_date' in df.columns:
    df.drop(columns=['birth_date'], inplace=True)

# --- Step 3: Load NFL rosters from 2018–2024 ---
seasons = list(range(2018, 2025))
rosters = nfl.import_seasonal_rosters(seasons)

# --- Step 4: Merge correct birthdates based on cleaned playerName column ---
df = pd.merge(
    df,
    rosters[['player_name', 'birth_date']],
    left_on='playerName',
    right_on='player_name',
    how='left'
)

# --- Step 5: Drop the extra player_name column from the merge ---
df.drop(columns=['player_name'], inplace=True)

# --- Step 6: Save the updated file ---
df.to_csv(str(ROOT / "data/intermediate/player_enrichment/nextgen_with_teams_birthdates_and_travel_UPDATED.csv"), index=False)

print(f"✅ Birthdates replaced and saved to {ROOT / 'data/intermediate/player_enrichment/nextgen_with_teams_birthdates_and_travel_UPDATED.csv'}")
