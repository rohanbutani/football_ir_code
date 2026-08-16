from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd
import re
try:
    from nfl_data_py import import_seasonal_rosters
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"{Path(__file__).name} requires nfl_data_py. Install it before running this script."
    ) from exc

# Define list of players with missing birth dates
missing_players = [
    {"playerName": "Odell Beckham", "team": "BAL", "season": 2023},
    {"playerName": "Marvin Mims", "team": "DEN", "season": 2024},
    {"playerName": "Chigoziem Okonkwo", "team": "TEN", "season": 2024},
    {"playerName": "Calvin Austin", "team": "PIT", "season": 2024},
    {"playerName": "Kavontae Turpin", "team": "DAL", "season": 2024},
    {"playerName": "Brian Thomas", "team": "JAX", "season": 2024},
    {"playerName": "D.J. Moore", "team": "CHI", "season": 2024},
    {"playerName": "Deebo Samuel", "team": "SF", "season": 2024},
    {"playerName": "Marvin Harrison", "team": "ARI", "season": 2024}
]

# Collect unique seasons
seasons = sorted(set(player["season"] for player in missing_players))

# Pull rosters
rosters = import_seasonal_rosters(seasons)

# Filter to WR and TE
relevant_rosters = rosters[rosters["position"].isin(["WR", "TE"])]

# Display for each player
for entry in missing_players:
    player, team, season = entry["playerName"], entry["team"], entry["season"]
    print(f"\n--- {player} | Team: {team}, Season: {season} ---")
    matching_roster = relevant_rosters[(relevant_rosters["team"] == team) & (relevant_rosters["season"] == season)]
    print(matching_roster[["player_name", "position"]])