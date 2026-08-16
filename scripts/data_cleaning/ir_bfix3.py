import pandas as pd

# === User-adjustable file paths ===
INPUT_CSV = 'C:/Users/rohan/ir_player_names_deduplicated_cleaned3.csv'
OUTPUT_CSV = 'C:/Users/rohan/ir_player_names_truncated.csv'

# === Load data ===
df = pd.read_csv(INPUT_CSV, dtype=str)

# === Truncate 'player_clean_cleaned' to first two words ===
def truncate_name(name: str) -> str:
    # Split on whitespace and keep up to the first two tokens
    parts = name.split()
    return ' '.join(parts[:2]) if parts else ''

# Apply truncation
if 'player_clean_cleaned' in df.columns:
    df['player_clean_cleaned'] = df['player_clean_cleaned'].fillna('').apply(truncate_name)
else:
    raise KeyError("Column 'player_clean_cleaned' not found in input CSV.")

# === Save output ===
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Truncated names saved to {OUTPUT_CSV}")
