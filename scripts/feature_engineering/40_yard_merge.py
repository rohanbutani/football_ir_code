from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd
import nfl_data_py as nfl

# --- Load main dataset ---
df = pd.read_csv(str(ROOT / "data/intermediate/player_enrichment/player_season_with_epa_sos_age_height_weight_FINAL.csv"))
df["playerName"] = df["playerName"].str.lower().str.strip()

# --- Load Combine data with correct column ---
combine = nfl.import_combine_data()
combine["player_name"] = combine["player_name"].str.lower().str.strip()
combine = combine[["player_name", "season", "forty"]]

# --- Merge by name and season ---
merged = pd.merge(
    df,
    combine,
    how="left",
    left_on=["playerName", "season"],
    right_on=["player_name", "season"]
)
merged.drop(columns=["player_name"], inplace=True)

# --- Fallback: patch missing values using name-only match ---
missing_mask = merged["forty"].isnull()
fallback_lookup = (
    combine.dropna(subset=["forty"])
           .drop_duplicates(subset=["player_name"])
           .set_index("player_name")["forty"]
)

merged.loc[missing_mask, "forty"] = (
    merged.loc[missing_mask, "playerName"].map(fallback_lookup)
)

# --- Save final output ---
merged.to_csv(str(ROOT / "data/intermediate/player_enrichment/player_season_with_forty_yard.csv"), index=False)
print(f"✅ 40-yard dash data merged and patched. Saved as {ROOT / 'data/intermediate/player_enrichment/player_season_with_forty_yard.csv'}")