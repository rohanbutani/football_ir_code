from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd
from datetime import datetime

# Load your cleaned IR "additions only" dataset
df = pd.read_csv(str(ROOT / "data/raw/injury_reserve/ir_additions_only.csv"))

# Convert the Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Define function to calculate season year
def get_season_year(date):
    if pd.isna(date):
        return None
    season_cutoff = datetime(year=date.year, month=3, day=14)
    return date.year - 1 if date < season_cutoff else date.year

# Apply it to each row
df['Season'] = df['Date'].apply(get_season_year)

# Save updated file
df.to_csv(str(ROOT / "data/intermediate/injury_matching/ir_additions_with_season.csv"), index=False)
print(f"✅ Added 'Season' column using 3/14 cutoff. Saved to {ROOT / 'data/intermediate/injury_matching/ir_additions_with_season.csv'}")