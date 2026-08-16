from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd
import re

# Load your cleaned IR dataset
df = pd.read_csv(str(ROOT / "data/intermediate/injury_matching/ir_cleaned_for_matching2.csv"), encoding="utf-8")

# Function to extract clean player name
def extract_clean_name(name):
    if pd.isna(name):
        return ""
    # Only keep letters, hyphens, apostrophes, and whitespace
    name = re.sub(r"[^a-zA-Z\-\s']", "", name)
    name = name.lower().strip()
    name = " ".join(name.split())  # Normalize whitespace
    return name

# Apply the function to the 'Player' column
df["player_clean_extracted"] = df["Relinquished"].apply(extract_clean_name)

# Save to a new CSV
df.to_csv(str(ROOT / "data/intermediate/injury_matching/ir_player_names_extracted.csv"), index=False, encoding="utf-8")
print(f"✅ Done! Clean player names saved to {ROOT / 'data/intermediate/injury_matching/ir_player_names_extracted.csv'}")