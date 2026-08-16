from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import pandas as pd

# Load the file
df = pd.read_csv(str(ROOT / "data/intermediate/player_enrichment/nextgen_with_teams_and_birthdates_fixed.csv"))

# Step 1: Resolve birth_date_x and birth_date_y -> keep only one column with priority
#df["birth_date"] = df["birth_date_x"].combine_first(df["birth_date_y"])
#df.drop(columns=["birth_date_x", "birth_date_y"], inplace=True)

# Step 2: Drop exact duplicate rows
df = df.drop_duplicates()

# Step 3: Drop duplicates based on playerName and season (keep first)
df = df.drop_duplicates(subset=["playerName", "season"])

# Step 4: Save cleaned version
df.to_csv(str(ROOT / "data/intermediate/player_enrichment/nextgen_with_teams_and_birthdates_deduped.csv"), index=False)

print(f"✅ Cleaned file saved as {ROOT / 'data/intermediate/player_enrichment/nextgen_with_teams_and_birthdates_deduped.csv'}")