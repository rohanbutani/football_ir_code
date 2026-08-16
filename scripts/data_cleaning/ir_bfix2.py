import pandas as pd

# Input/output paths
IN_CSV  = 'C:/Users/rohan/ir_player_names_deduplicated_cleaned2.csv'
OUT_CSV = 'C:/Users/rohan/ir_player_names_deduplicated_cleaned3.csv'

# Load data
df = pd.read_csv(IN_CSV, dtype=str)

# Method 1: vectorized regex replace
df['player_clean_cleaned'] = (
    df['player_clean_cleaned']
      # remove any space + single letter at end of the string
      .str.replace(r'\s+[A-Za-z]$', '', regex=True)
      .str.strip()
)

# Alternatively, Method 2: split/apply (uncomment to use)
# def drop_trailing_letter(name):
#     if pd.isna(name): return name
#     parts = name.strip().split()
#     if parts and len(parts[-1]) == 1 and parts[-1].isalpha():
#         parts.pop()
#     return ' '.join(parts)
# df['player_clean_cleaned'] = df['player_clean_cleaned'].apply(drop_trailing_letter)

# Save cleaned CSV
df.to_csv(OUT_CSV, index=False)
print(f"Processed {len(df)} rows → saved cleaned names to {OUT_CSV}")