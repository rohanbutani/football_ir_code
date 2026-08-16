from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd

# === Step 1: Load main player dataset (with EPA already included) ===
player_df = pd.read_csv(str(ROOT / "data/intermediate/player_enrichment/player_season_with_snaps_with_epa.csv"))

# === Step 2: Load SoS win percentage data ===
sos_df = pd.read_csv(str(ROOT / "data/raw/team_context/sos_win_pct_2018_2024.csv"))

# Ensure consistent formatting for merge keys
player_df["team_abbr"] = player_df["team_abbr"].str.upper()
sos_df["team_abbr"] = sos_df["team_abbr"].str.upper()

player_df["season"] = player_df["season"].astype(int)
sos_df["season"] = sos_df["season"].astype(int)

# === Step 3: Merge SoS into player dataset ===
merged_df = pd.merge(
    player_df,
    sos_df,
    on=["team_abbr", "season"],
    how="left"
)

# === Step 4: Save merged output ===
merged_df.to_csv(str(ROOT / "data/intermediate/player_enrichment/player_season_with_epa_and_sos.csv"), index=False)
print("✅ Merge complete. File saved as player_season_with_epa_and_sos.csv")