from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd

# === Step 1: Load player-level data ===
main_df = pd.read_csv(str(ROOT / "data/intermediate/player_enrichment/nextgen_with_teams_birthdates_and_travel.csv"))

# === Step 2: Load turf game percentage data ===
turf_df = pd.read_csv(str(ROOT / "data/raw/travel_surface/turf_game_percentage_2018_2024.csv"))

# === Step 3: Rename 'team' to match 'team_abbr' for merge ===
turf_df = turf_df.rename(columns={"team": "team_abbr"})

# === Step 4: Merge turf percentage into player-level dataset ===
merged_df = main_df.merge(
    turf_df,
    how="left",
    on=["season", "team_abbr"]
)

# === Step 5 (Optional): Save output ===
merged_df.to_csv(str(ROOT / "data/intermediate/player_enrichment/nextgen_with_turf_percentage.csv"), index=False)

# === Step 6: Print sample rows to verify ===
print(merged_df[['playerName', 'season', 'team_abbr', 'turf_game_pct']].head())