import pandas as pd

# Input and output paths
INPUT_CSV  = 'C:/Users/rohan/ir_player_names_extracted.csv'
OUTPUT_CSV = 'C:/Users/rohan/ir_player_names_deduplicated.csv'

# Load IR dataset
df = pd.read_csv(INPUT_CSV, dtype=str)

# Strip whitespace and ensure proper types
df['player_clean_extracted'] = df['player_clean_extracted'].str.strip()
df['Season'] = df['Season'].astype(int, errors='ignore')
df['Date'] = df['Date'].str.strip()

# Drop duplicates on player name + season + date
deduped = df.drop_duplicates(subset=['player_clean_extracted', 'Season', 'Date'])

# Save cleaned file
deduped.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Deduplicated IR dataset saved to: {OUTPUT_CSV}")
print(f"Original rows: {len(df)}, After deduplication: {len(deduped)}")