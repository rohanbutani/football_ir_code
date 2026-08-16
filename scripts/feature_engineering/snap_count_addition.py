import pandas as pd
import nfl_data_py as nfl

# --- Step 1: Load your player-season dataset ---
df = pd.read_csv("player_season_with_turf_percentage2.csv")

# --- Step 2: Extract seasons to query snap counts ---
seasons = df["season"].unique().tolist()

# --- Step 3: Load weekly snap counts from nfl_data_py ---
snap_weekly = nfl.import_snap_counts(seasons)

# --- Step 4: Aggregate total offensive snaps per player-season ---
snap_by_season = (
    snap_weekly.groupby(["player", "season"])
    .agg(total_snaps=("offense_snaps", "sum"))
    .reset_index()
)

# --- Step 5: Merge snap data with your player-season data ---
df = df.merge(
    snap_by_season,
    left_on=["playerName", "season"],
    right_on=["player", "season"],
    how="left"
)

# --- Step 6: Remove 'player' column if it exists after merge ---
if "player" in df.columns:
    df.drop(columns=["player"], inplace=True)

# --- Step 7: Save updated dataset ---
df.to_csv("player_season_with_snaps.csv", index=False)

# --- Optional: Debug print ---
matched = df["total_snaps"].notna().sum()
print(f"✅ Total offensive snaps added and saved to 'player_season_with_snaps.csv'")
print(f"🔎 Snap counts matched for {matched} of {len(df)} player-seasons.")