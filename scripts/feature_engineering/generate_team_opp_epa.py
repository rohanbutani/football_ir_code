from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import pandas as pd
import nfl_data_py as nfl

def load_defensive_epa(season):
    import nfl_data_py as nfl
    pbp = nfl.import_pbp_data([season])

    # Filter to pass plays where defense is involved
    pass_plays = pbp[
        (pbp['play_type'] == 'pass') &
        (pbp['epa'].notna())
    ]

    # Group by defensive team
    def_epa = (
        pass_plays.groupby('defteam')['epa']
        .mean()
        .reset_index()
        .rename(columns={'defteam': 'team', 'epa': 'def_pass_epa_per_play'})
    )
    def_epa['season'] = season
    return def_epa
def load_team_schedule(season):
    """
    Load NFL regular season schedule and format team-opponent pairs.
    """
    sched = nfl.import_schedules([season])
    sched = sched[sched['game_type'] == 'REG']

    # One row per team per game with opponent
    home = sched[['home_team', 'away_team', 'season']].rename(columns={'home_team': 'team', 'away_team': 'opponent'})
    away = sched[['away_team', 'home_team', 'season']].rename(columns={'away_team': 'team', 'home_team': 'opponent'})
    return pd.concat([home, away], ignore_index=True)

def calculate_def_epa_sos(season):
    """
    Calculate average opponent defensive pass EPA/play for each team in a season.
    """
    epa_df = load_defensive_epa(season)
    schedule_df = load_team_schedule(season)

    # Join opponent EPA
    merged = schedule_df.merge(
        epa_df,
        left_on=['opponent', 'season'],
        right_on=['team', 'season'],
        how='left'
    ).rename(columns={'def_pass_epa_per_play': 'opponent_pass_epa'})

    # Drop extra column created during merge
    merged = merged.drop(columns='team_y')
    merged = merged.rename(columns={'team_x': 'team'})

    # Aggregate: average opponent EPA per team-season
    sos_df = (
        merged.groupby(['team', 'season'])['opponent_pass_epa']
        .mean()
        .reset_index()
        .rename(columns={'team': 'team_abbr', 'opponent_pass_epa': 'avg_opponent_pass_epa'})
    )

    return sos_df

if __name__ == "__main__":
    season = 2023
    sos = calculate_def_epa_sos(season)
    sos.to_csv(f"epa_sos_{season}.csv", index=False)
    print(sos.head())