import pandas as pd

# Load the player-season dataset
player_df = pd.read_csv("player_season_with_snaps.csv")

# Load the EPA dataset (team-level EPA per season)
epa_df = pd.read_csv("epa_sos_2018_2024.csv")



# Normalize column formatting for consistent merging
epa_df["team_abbr"] = epa_df["team_abbr"].str.upper()
player_df["team_abbr"] = player_df["team_abbr"].str.upper()

epa_df["season"] = epa_df["season"].astype(int)
player_df["season"] = player_df["season"].astype(int)

# Perform the merge on team_abbr and season
# Replace 'def_epa_per_play' below if your EPA column is named differently
merged_df = pd.merge(
    player_df,
    epa_df[["team_abbr", "season", "avg_opponent_pass_epa"]],
    on=["team_abbr", "season"],
    how="left"
)

# Save the merged output to a new file
merged_df.to_csv("player_season_with_snaps_with_epa.csv", index=False)

print("✅ Merge complete. File saved as player_season_with_snaps_with_epa.csv")