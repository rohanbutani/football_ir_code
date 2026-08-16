import pandas as pd

# Paths to your input and output CSV files
IN_CSV = 'C:/Users/rohan/ir_player_names_deduplicated_cleaned.csv'
OUT_CSV = 'C:/Users/rohan/ir_player_names_deduplicated_cleaned2.csv'

# Load the data
df = pd.read_csv(IN_CSV, dtype=str)

# Remove the trailing "b --" artifact from the player_clean_cleaned column
df['player_clean_cleaned'] = (
    df['player_clean_cleaned']
    .str.replace(r'\s*b --$', '', regex=True)  # drop any space(s) + "b --" at end
    .str.strip()                                # trim any leftover whitespace
)

# Save the cleaned data
df.to_csv(OUT_CSV, index=False)
print(f"Cleaned {len(df)} rows and wrote output to {OUT_CSV}")