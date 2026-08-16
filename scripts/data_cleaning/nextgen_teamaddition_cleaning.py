import pandas as pd
import re

# Load your CSV
df = pd.read_csv("nfl_receiving_nextgen_2018_2024.csv")  # Change this to your actual file path

# Clean name function
def clean_name(name):
    if pd.isna(name):
        return ""
    name = str(name)
    name = name.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    name = re.sub(r"[^\w\s\-']", "", name)  # Remove non-alphanumeric symbols except - and '
    name = name.lower().strip()
    name = name.replace("-", " ").replace("'", "")
    return " ".join(name.split())

# Apply cleaning
df["player_clean"] = df["playerName"].apply(clean_name)

# Save cleaned version
df.to_csv("nextgen_with_cleaned_names.csv", index=False, encoding="utf-8")
print("✅ Done! Saved as 'nextgen_with_cleaned_names.csv'")