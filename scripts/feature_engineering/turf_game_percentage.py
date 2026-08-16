from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd
try:
    from nfl_data_py import import_schedules
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"{Path(__file__).name} requires nfl_data_py. Install it before running this script."
    ) from exc

# Surface overrides (assume no new stadium changes in 2024 unless known)
stadium_surface_by_year = {
    2018: {"OAK": "turf", "LA": "grass", "LAC": "grass"},
    2019: {"OAK": "turf", "LA": "grass", "LAC": "grass"},
    2020: {"LV": "grass", "LA": "turf", "LAC": "turf"},
    2021: {"LV": "grass", "LA": "turf", "LAC": "turf"},
    2022: {"LV": "grass", "LA": "turf", "LAC": "turf"},
    2023: {"LV": "grass", "LA": "turf", "LAC": "turf"},
    2024: {"LV": "grass", "LA": "turf", "LAC": "turf"}  # Assume no changes yet
}

# Default surface map for other teams
default_surface_map = {
    "ATL": "turf", "ARI": "grass", "BAL": "grass", "BUF": "turf", "CAR": "turf",
    "CHI": "grass", "CIN": "turf", "CLE": "grass", "DAL": "turf", "DEN": "grass",
    "DET": "turf", "GB": "grass", "HOU": "turf", "IND": "turf", "JAX": "grass",
    "KC": "grass", "MIA": "grass", "MIN": "turf", "NE": "turf", "NO": "turf",
    "NYG": "turf", "NYJ": "turf", "PHI": "grass", "PIT": "grass", "SF": "grass",
    "SEA": "turf", "TB": "grass", "TEN": "grass", "WAS": "grass"
}

def get_surface(team, season):
    if season in stadium_surface_by_year and team in stadium_surface_by_year[season]:
        return stadium_surface_by_year[season][team]
    return default_surface_map.get(team, "unknown")

# Add 2024 to schedule query
seasons = list(range(2018, 2025))
schedules = import_schedules(seasons)

# Assign surface to each game
schedules['surface'] = schedules.apply(lambda row: get_surface(row['home_team'], row['season']), axis=1)

# Create team-level game entries and drop duplicates
home_games = schedules[['season', 'week', 'home_team', 'surface']].rename(columns={'home_team': 'team'})
away_games = schedules[['season', 'week', 'away_team', 'surface']].rename(columns={'away_team': 'team'})

team_games = pd.concat([home_games, away_games]).drop_duplicates(subset=['season', 'week', 'team'])

# Label turf games
team_games['is_turf'] = team_games['surface'] == 'turf'

# Aggregate and compute percentages
summary = team_games.groupby(['team', 'season'])['is_turf'].agg(['sum', 'count']).reset_index()
summary['turf_game_pct'] = summary['sum'] / summary['count']

# Save output
summary[['team', 'season', 'turf_game_pct']].to_csv(str(ROOT / "data/raw/travel_surface/turf_game_percentage_2018_2024.csv"), index=False)
print(summary[['team', 'season', 'turf_game_pct']].tail(10))