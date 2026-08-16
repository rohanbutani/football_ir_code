import pandas as pd

# Step 1: Load your player-level dataset
df_players = pd.read_csv("nextgen_with_teams_and_birthdates_fixed.csv")

# Step 2: Load precomputed team travel mileage
# (generated from the script using nfl_data_py and haversine)
team_miles = pd.read_csv("team_season_travel_miles.csv")

# Step 3: Merge on both team and season
df_with_miles = pd.merge(
    df_players,
    team_miles,
    on=["team_abbr", "season"],
    how="left"  # Keep all players, even if a team-season match is missing
)

# Step 4: Save to new CSV
df_with_miles.to_csv("nextgen_with_teams_birthdates_and_travel.csv", index=False)

print("✅ Added 'total_round_trip_miles' to player data. Output saved as 'nextgen_with_teams_birthdates_and_travel.csv'.")
