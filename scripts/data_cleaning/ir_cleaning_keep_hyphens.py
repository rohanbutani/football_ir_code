import pandas as pd
import re

# Load your IR data
df = pd.read_csv("ir_additions_with_season.csv", encoding='utf-8')

# Define function to clean player names (preserve hyphens)
def clean_name(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')  # Fix encoding
    text = re.sub(r"â€¢", "", text)  # Remove bullet-like encoding issues
    text = re.sub(r"[^\w\s\-]", "", text)  # removes all except word chars, spaces, and hyphens
    text = text.lower().strip()
    return text

# Apply to both Acquired and Relinquished columns
df['Acquired'] = df['Acquired'].apply(clean_name)
df['Relinquished'] = df['Relinquished'].apply(clean_name)

# Create unified Player column
df['Player'] = df['Acquired'].combine_first(df['Relinquished'])

# Save cleaned dataset
df.to_csv("ir_cleaned_for_matching2.csv", index=False, encoding='utf-8')
print("✅ Hyphen-preserving names cleaned and saved to 'ir_cleaned_for_matching2.csv'")