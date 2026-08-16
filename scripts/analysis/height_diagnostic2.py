import pandas as pd
import nfl_data_py as nfl

# --- Load your dataset ---
df = pd.read_csv("player_season_with_epa_sos_age_height_PATCHED.csv")
df["playerName"] = df["playerName"].str.lower().str.strip()

# --- Pull roster data and standardize names ---
seasons = list(range(2018, 2025))
rosters = nfl.import_seasonal_rosters(seasons)
rosters["player_name"] = rosters["player_name"].str.lower().str.strip()

# --- Check if 'benjamin watson' is in the roster ---
watson_match = rosters[rosters["player_name"] == "benjamin watson"]

if not watson_match.empty and "height" in watson_match.columns:
    watson_height = watson_match["height"].dropna().iloc[0]  # Use first non-null height
    print(f"✅ Found 'benjamin watson' with height: {watson_height} inches")

    # --- Patch 'ben watson' rows in your dataset ---
    df.loc[df["playerName"] == "ben watson", "height"] = watson_height

    # --- Save updated file ---
    df.to_csv("player_season_with_epa_sos_age_height_FINAL.csv", index=False)
    print("✅ Patched 'ben watson' height and saved to 'player_season_with_epa_sos_age_height_FINAL.csv'")
else:
    print("❌ 'benjamin watson' not found in rosters or missing height value.")