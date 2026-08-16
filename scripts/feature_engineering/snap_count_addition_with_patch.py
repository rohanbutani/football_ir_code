from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import pandas as pd

# --- Step 1: Load your player-season dataset ---
df = pd.read_csv(str(ROOT / "data/intermediate/player_enrichment/player_season_with_turf_percentage2.csv"))

# --- Step 2: Apply manual name fixes for known mismatches ---
manual_name_map = {
    "Benjamin Watson": "Ben Watson",
    "Bisi Johnson": "Olabisi Johnson",
    "Chig Okonkwo": "Chigoziem Okonkwo",
    "Christopher Herndon": "Chris Herndon",
    "Gabe Davis": "Gabriel Davis"
}
df["playerName"] = df["playerName"].replace(manual_name_map)

# --- Step 3: Load snap weekly data (pre-downloaded) ---
snap_weekly = pd.read_csv(str(ROOT / "data/raw/snaps/snap_weekly_full.csv"))

# --- Step 4: Aggregate total offensive snaps per player-season ---
snap_by_season = (
    snap_weekly.groupby(["player", "season"])
    .agg(total_snaps=("offense_snaps", "sum"))
    .reset_index()
)

# --- Step 5: Merge into your player-season dataset ---
df = df.merge(
    snap_by_season,
    left_on=["playerName", "season"],
    right_on=["player", "season"],
    how="left"
)

# --- Step 6: Cleanup ---
if "player" in df.columns:
    df.drop(columns=["player"], inplace=True)

# --- Step 7: Save updated dataset ---
df.to_csv(str(ROOT / "data/intermediate/player_enrichment/player_season_with_snaps.csv"), index=False)

print(f"✅ Snap counts merged and saved to {ROOT / 'data/intermediate/player_enrichment/player_season_with_snaps.csv'}")