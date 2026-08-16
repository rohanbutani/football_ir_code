import pandas as pd
import nfl_data_py as nfl

def load_defensive_epa(season):
    print(f"Loading PBP for {season}...")
    pbp = nfl.import_pbp_data([season])
    pass_plays = pbp[(pbp['play_type'] == 'pass') & (pbp['epa'].notna())]

    def_epa = (
        pass_plays.groupby('defteam')['epa']
        .mean()
        .reset_index()
        .rename(columns={'defteam': 'team', 'epa': 'def_pass_epa_per_play'})
    )
    def_epa['season'] = season
    return def_epa

def load_team_schedule(season):
    sched = nfl.import_schedules([season])
    sched = sched[sched['game_type'] == 'REG']

    home = sched[['home_team', 'away_team', 'season']].rename(columns={'home_team': 'team', 'away_team': 'opponent'})
    away = sched[['away_team', 'home_team', 'season']].rename(columns={'away_team': 'team', 'home_team': 'opponent'})
    return pd.concat([home, away], ignore_index=True)

def calculate_def_epa_sos(season):
    epa_df = load_defensive_epa(season)
    schedule_df = load_team_schedule(season)

    merged = schedule_df.merge(
        epa_df,
        left_on=['opponent', 'season'],
        right_on=['team', 'season'],
        how='left'
    ).rename(columns={'def_pass_epa_per_play': 'opponent_pass_epa'})

    merged = merged.drop(columns='team_y').rename(columns={'team_x': 'team'})

    sos_df = (
        merged.groupby(['team', 'season'])['opponent_pass_epa']
        .mean()
        .reset_index()
        .rename(columns={'team': 'team_abbr', 'opponent_pass_epa': 'avg_opponent_pass_epa'})
    )

    return sos_df

if __name__ == "__main__":
    all_sos = []

    for year in range(2018, 2025):  # Loop through 2018 to 2024
        print(f"Processing season {year}...")
        sos_df = calculate_def_epa_sos(year)
        all_sos.append(sos_df)

    combined_df = pd.concat(all_sos, ignore_index=True)
    combined_df.to_csv("epa_sos_2018_2024.csv", index=False)
    print("✅ Done. File saved as epa_sos_2018_2024.csv")