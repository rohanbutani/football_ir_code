from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd
import re

# Load your CSV
df = pd.read_csv(str(ROOT / "data/raw/nextgen/nfl_receiving_nextgen_2018_2024.csv"))

# Clean name function with suffix preservation
def clean_name(name):
    if pd.isna(name):
        return ""
    name = str(name)
    name = name.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')

    # Keep only letters, numbers, hyphens, apostrophes, and whitespace
    name = re.sub(r"[^\w\s\-\']", "", name)

    # Normalize whitespace
    name = " ".join(name.split())

    # Lowercase everything
    name = name.lower().strip()

    # Make sure suffixes like jr, sr, iii, iv stay intact (no special handling required here if they are already part of the name)
    return name

# Apply cleaning
df["player_clean"] = df["playerName"].apply(clean_name)

# Save cleaned version
df.to_csv(str(ROOT / "data/raw/nextgen/nextgen_with_cleaned_names3.csv"), index=False, encoding="utf-8")
print(f"✅ Done! Saved as {ROOT / 'data/raw/nextgen/nextgen_with_cleaned_names3.csv'}")