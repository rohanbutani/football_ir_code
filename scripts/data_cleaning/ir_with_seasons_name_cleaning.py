from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd
import re

# Load your IR data
df = pd.read_csv(str(ROOT / "data/intermediate/injury_matching/ir_additions_with_season.csv"), encoding='utf-8')

# Define function to clean player names
def clean_name(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')  # Fix encoding
    text = re.sub(r"â€¢", "", text)  # Remove bullet encoding
    text = re.sub(r"[^\w\s\-\']", "", text)  # Remove unwanted characters but keep hyphens/apostrophes
    text = text.lower().strip()
    return text

# Apply to both Acquired and Relinquished columns
df['Acquired'] = df['Acquired'].apply(clean_name)
df['Relinquished'] = df['Relinquished'].apply(clean_name)

# Create a unified player column (some rows only have one or the other)
df['Player'] = df['Acquired'].combine_first(df['Relinquished'])

# Optional: convert to Title Case for display (but match on lowercase)
# df['Player'] = df['Player'].str.title()

# Save the cleaned dataset
df.to_csv(str(ROOT / "data/intermediate/injury_matching/ir_cleaned_for_matching.csv"), index=False, encoding='utf-8')
print(f"✅ Names cleaned and ready for cross-referencing. Saved to {ROOT / 'data/intermediate/injury_matching/ir_cleaned_for_matching.csv'}")