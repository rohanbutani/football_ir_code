from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd
try:
    import nfl_data_py as nfl
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "player_height_merge.py requires nfl_data_py. Install it before running this script."
    ) from exc

# --- Step 1: Load your main player-season dataset ---
df = pd.read_csv(str(ROOT / "data/intermediate/player_enrichment/player_season_with_epa_sos_age.csv"))

# --- Step 2: Pull seasonal roster data for 2018–2024 ---
seasons = list(range(2018, 2025))  # Includes 2024
rosters = nfl.import_seasonal_rosters(seasons)

# --- Step 3: Standardize player names in both datasets ---
df["playerName"] = df["playerName"].str.lower().str.strip()
rosters["player_name"] = rosters["player_name"].str.lower().str.strip()

# --- Step 4: Merge height based on player name and season ---
df_with_height = pd.merge(
    df,
    rosters[["player_name", "season", "height"]],
    left_on=["playerName", "season"],
    right_on=["player_name", "season"],
    how="left"
)

# --- Step 5: Drop redundant merge column ---
df_with_height.drop(columns=["player_name"], inplace=True)

# --- Step 6: Save the final dataset with height ---
df_with_height.to_csv(str(ROOT / "data/intermediate/player_enrichment/player_season_with_epa_sos_age_height.csv"), index=False)
print(f"✅ Height successfully merged and saved to {ROOT / 'data/intermediate/player_enrichment/player_season_with_epa_sos_age_height.csv'}.")
