from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd
try:
    import nfl_data_py as nfl
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"{Path(__file__).name} requires nfl_data_py. Install it before running this script."
    ) from exc

# --- Step 1: Load your player-season-team-level data ---
df = pd.read_csv(str(ROOT / "data/intermediate/player_enrichment/nextgen_with_teams_and_birthdates_deduped.csv"))

# --- Step 2: Load schedule data from nfl_data_py ---
seasons = df["season"].unique().tolist()
schedule = nfl.import_schedules(seasons)

# --- Step 3: Clean & filter schedule ---
# Keep regular season only
schedule = schedule[schedule["game_type"] == "REG"]

# Standardize turf-like surfaces
turf_keywords = ["turf", "fieldturf", "matrixturf", "astroturf", "sportturf"]
schedule["is_turf"] = schedule["surface"].fillna("").str.lower().apply(
    lambda x: any(keyword in x for keyword in turf_keywords)
)

# --- Step 4: Reshape schedule into long format (team, season, is_turf) ---
home = schedule[["season", "week", "home_team", "is_turf"]].rename(
    columns={"home_team": "team_abbr"}
)
away = schedule[["season", "week", "away_team", "is_turf"]].rename(
    columns={"away_team": "team_abbr"}
)
games_long = pd.concat([home, away])

# --- Step 5: Calculate team-season turf percentages ---
turf_pct = (
    games_long.groupby(["team_abbr", "season"])["is_turf"]
    .mean()
    .reset_index()
    .rename(columns={"is_turf": "turf_percentage"})
)

# --- Step 6: Merge turf percentage onto player-season-team data ---
df = df.merge(turf_pct, on=["team_abbr", "season"], how="left")

# --- Step 7: Save result ---
df.to_csv(str(ROOT / "data/intermediate/player_enrichment/player_season_with_turf_percentage2.csv"), index=False)

print("✅ Done: Turf percentages added for each player-season-team.")