from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd
import re
from nfl_data_py import import_seasonal_rosters

# --- Helper to clean names ---
def clean_name(name):
    if pd.isna(name):
        return ""
    name = str(name)
    name = name.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')  # fix symbols
    name = re.sub(r"â€¢", "", name)
    name = re.sub(r"[^\w\s\-\']", "", name)
    name = name.lower().strip()
    return name


# --- Load the cleaned NGS file ---
df = pd.read_csv(str(ROOT / "data/raw/nextgen/nextgen_with_cleaned_names2.csv"))

# --- Get all unique years ---
years = sorted(df["season"].dropna().unique().astype(int))

# --- Pull rosters from nfl_data_py ---
rosters = import_seasonal_rosters(years)

# --- Clean player names in the roster ---
rosters["player_clean"] = rosters["player_name"].apply(clean_name)

# --- Merge rosters onto nextgen data using player_clean and season ---
merged = df.merge(
    rosters[["season", "player_clean", "team"]],
    on=["season", "player_clean"],
    how="left"
)

# --- Rename for clarity ---
merged = merged.rename(columns={"team": "team_abbr"})

# --- Save to CSV ---
merged.to_csv(str(ROOT / "data/raw/nextgen/nextgen_with_teams.csv"), index=False)
print(f"✅ Done! Saved as {ROOT / 'data/raw/nextgen/nextgen_with_teams.csv'}")