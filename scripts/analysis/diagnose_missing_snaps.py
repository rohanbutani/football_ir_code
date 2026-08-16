from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import pandas as pd

# --- Step 1: Load files ---
df = pd.read_csv(str(ROOT / "data/intermediate/player_enrichment/player_season_with_snaps.csv"))
snap_weekly = pd.read_csv(str(ROOT / "data/raw/snaps/snap_weekly_full.csv"))

# --- Step 2: Identify unmatched player-season rows ---
missing = df[df["total_snaps"].isna()].copy()

# --- Step 3: Clean names for fuzzy matching ---
def clean_name(name):
    if pd.isna(name):
        return ""
    return (
        name.lower()
        .replace(" jr", "")
        .replace(" sr", "")
        .replace(" iii", "")
        .replace(" ii", "")
        .replace(".", "")
        .strip()
    )

missing.loc[:, "playerName_clean"] = missing["playerName"].apply(clean_name)
snap_weekly["player_clean"] = snap_weekly["player"].apply(clean_name)

# --- Step 4: Check for name mismatches ---
unmatched_names = set(missing["playerName_clean"])
snap_names = set(snap_weekly["player_clean"])
missing_due_to_name = unmatched_names - snap_names

print("🔍 Players not found in snap data (name mismatch likely):")
for name in sorted(missing_due_to_name):
    print("-", name)

# --- Step 5: For those that do exist, check for team mismatch ---
print("\n🔄 Players with name match but team mismatch:")
for _, row in missing.iterrows():
    pname_clean = row["playerName_clean"]
    season = row["season"]
    team = row["team_abbr"]

    player_rows = snap_weekly[(snap_weekly["player_clean"] == pname_clean) & (snap_weekly["season"] == season)]
    
    if not player_rows.empty:
        teams_played = player_rows["team"].dropna().unique()
        if team not in teams_played:
            print(f"- {row['playerName']} ({season}): CSV team = {team}, Snap data teams = {teams_played}")