from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import pandas as pd
import nfl_data_py as nfl

# === Step 1: Load schedule data (2018–2024) ===
schedule = nfl.import_schedules([2018, 2019, 2020, 2021, 2022, 2023, 2024])
schedule = schedule[~schedule.game_type.isin(["PRE", "POST"])]  # Regular season only

# Keep relevant columns
schedule = schedule[["season", "week", "home_team", "away_team", "home_score", "away_score"]]

# Create one row per matchup per team (team vs opponent)
home = schedule.rename(columns={"home_team": "team", "away_team": "opponent", "home_score": "points_for", "away_score": "points_against"})
away = schedule.rename(columns={"away_team": "team", "home_team": "opponent", "away_score": "points_for", "home_score": "points_against"})
team_games = pd.concat([home, away], ignore_index=True)

# === Step 2: Compute win % for each team-season ===
# Determine wins
team_games["win"] = team_games["points_for"] > team_games["points_against"]

# Count total games and wins
team_record = (
    team_games.groupby(["team", "season"])
    .agg(games_played=("week", "count"), wins=("win", "sum"))
    .reset_index()
)
team_record["win_pct"] = team_record["wins"] / team_record["games_played"]

# === Step 3: Merge opponent win % into the original matchups ===
# Left: each team vs opponent (all 17 games)
# Right: opponent's win percentage for that season
opponent_win_pct = team_record.rename(columns={"team": "opponent", "win_pct": "opponent_win_pct"})

team_games = pd.merge(
    team_games,
    opponent_win_pct[["opponent", "season", "opponent_win_pct"]],
    on=["opponent", "season"],
    how="left"
)

# === Step 4: Compute average opponent win % for each team-season ===
sos_win_pct = (
    team_games.groupby(["team", "season"])
    .agg(sos_win_pct=("opponent_win_pct", "mean"))
    .reset_index()
    .rename(columns={"team": "team_abbr"})
)

# === Step 5: Save or preview ===
sos_win_pct.to_csv(str(ROOT / "data/raw/team_context/sos_win_pct_2018_2024.csv"), index=False)
print("✅ Done. File saved as sos_win_pct_2018_2024.csv")