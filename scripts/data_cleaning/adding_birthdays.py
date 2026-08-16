from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd
try:
    import nfl_data_py as nfl
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"{Path(__file__).name} requires nfl_data_py. Install it before running this script."
    ) from exc

# --- Step 1: Load original dataset ---
df = pd.read_csv(str(ROOT / "data/raw/nextgen/nextgen_with_teams_FIXED.csv"))

# --- Step 2: Load seasonal rosters from 2018–2024 ---
seasons = list(range(2018, 2025))  # Inclusive of 2024
rosters = nfl.import_seasonal_rosters(seasons)

# --- Step 3: Merge birthdates based on player name ---
df_with_birthdates = pd.merge(
    df,
    rosters[['player_name', 'birth_date']],
    left_on='playerName',  # Assumes cleaned names exist in this column
    right_on='player_name',
    how='left'
)

# --- Step 4: Remove redundant column ---
df_with_birthdates.drop('player_name', axis=1, inplace=True)

# --- Step 5: Save result ---
df_with_birthdates.to_csv(str(ROOT / "data/intermediate/player_enrichment/nextgen_with_teams_and_birthdates.csv"), index=False)

print(f"✅ Birthdates successfully merged and saved to {ROOT / 'data/intermediate/player_enrichment/nextgen_with_teams_and_birthdates.csv'}.")