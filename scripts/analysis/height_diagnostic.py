import pandas as pd
import nfl_data_py as nfl

# Load your merged dataset
df = pd.read_csv("player_season_with_epa_sos_age_height.csv")

# Identify rows with missing height
missing_df = df[df["height"].isnull()].copy()

# Pull roster data
seasons = list(range(2018, 2025))
rosters = nfl.import_seasonal_rosters(seasons)

# Standardize and deduplicate
rosters["player_name"] = rosters["player_name"].str.lower().str.strip()
rosters_dedup = rosters.drop_duplicates(subset=["player_name"])
rosters_dedup = rosters_dedup[["player_name", "height"]].copy()  # Ensure height is present

# Debug: Print column names if needed
print("✅ Roster columns:", rosters_dedup.columns.tolist())

# Prepare playerName column for matching
df["playerName"] = df["playerName"].str.lower().str.strip()

# Fallback merge on player name only
fallback_patch = pd.merge(
    df[df["height"].isnull()],
    rosters_dedup,
    left_on="playerName",
    right_on="player_name",
    how="left"
)

# Debug: Check sample of fallback_patch to ensure height column exists
print("✅ Fallback patch sample:\n", fallback_patch[["playerName", "height_y"]].head())

# Patch height where missing
df.loc[df["height"].isnull(), "height"] = fallback_patch["height_y"].values

# Save the updated dataset
df.to_csv("player_season_with_epa_sos_age_height_PATCHED.csv", index=False)
print("✅ Missing height values patched and saved.")