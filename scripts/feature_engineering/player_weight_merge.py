from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import pandas as pd
import nfl_data_py as nfl

# --- Step 1: Load your player-season dataset ---
df = pd.read_csv(str(ROOT / "data/intermediate/player_enrichment/player_season_with_epa_sos_age_height_FINAL.csv"))
df["playerName"] = df["playerName"].str.lower().str.strip()

# --- Step 2: Pull roster data for all seasons and standardize ---
seasons = list(range(2018, 2025))
rosters = nfl.import_seasonal_rosters(seasons)
rosters["player_name"] = rosters["player_name"].str.lower().str.strip()

# --- Step 3: Merge weight by playerName and season ---
df = pd.merge(
    df,
    rosters[["player_name", "season", "weight"]],
    left_on=["playerName", "season"],
    right_on=["player_name", "season"],
    how="left"
)

# Drop redundant column
df.drop(columns=["player_name"], inplace=True)

# --- Step 4: Fallback patch for missing weights using name-only match ---
# Get fallback weight from first non-null per player
fallback_weights = (
    rosters[["player_name", "weight"]]
    .dropna()
    .drop_duplicates(subset=["player_name"])
    .set_index("player_name")["weight"]
)

# Patch missing weight entries
mask_missing_weight = df["weight"].isnull()
df.loc[mask_missing_weight, "weight"] = df.loc[mask_missing_weight, "playerName"].map(fallback_weights)

# --- Step 5: Manual patch example (e.g., 'ben watson') ---
df.loc[df["playerName"] == "ben watson", "weight"] = 251  # from PFR: 251 lbs

# --- Step 6: Save final dataset ---
df.to_csv(str(ROOT / "data/intermediate/player_enrichment/player_season_with_epa_sos_age_height_weight_FINAL.csv"), index=False)
print(f"✅ Weight merged and patched. Final file saved as {ROOT / 'data/intermediate/player_enrichment/player_season_with_epa_sos_age_height_weight_FINAL.csv'}")