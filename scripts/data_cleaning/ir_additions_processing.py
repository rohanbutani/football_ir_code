from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import pandas as pd

# Load your full IR data
df = pd.read_csv(str(ROOT / "data/raw/injury_reserve/ir_data_2010_onward.csv"))  # change filename as needed

# Normalize columns
df['Notes'] = df['Notes'].astype(str).str.lower()
df['Relinquished'] = df['Relinquished'].astype(str).str.strip()

# Keep only rows where:
# - Relinquished column has a player name (not empty or null)
# - Notes indicate player was placed on IR
df_filtered = df[
    df['Relinquished'].notna() &
    df['Relinquished'].str.len().gt(0) &
    df['Notes'].str.contains("placed on ir")
]

# Save the filtered result
df_filtered.to_csv(str(ROOT / "data/raw/injury_reserve/ir_additions_only.csv"), index=False)
print(f"✅ Filtered {len(df_filtered)} 'placed on IR' entries saved to {ROOT / 'data/raw/injury_reserve/ir_additions_only.csv'}")