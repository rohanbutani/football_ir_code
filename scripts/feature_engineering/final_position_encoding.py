from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

import pandas as pd

# ─── FILE PATHS ───────────────────────────────────────────────────────────────
INPUT_CSV  = str(ROOT / "data/processed/final_datasets/FINAL_dataset.csv")
OUTPUT_CSV = str(ROOT / "data/processed/final_datasets/FINAL_dataset_with_position_encoded.csv")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)

# ─── CLEAN POSITION COLUMN ────────────────────────────────────────────────────
df['position'] = df['position'].astype(str).str.strip().str.upper()

# ─── ONE-HOT ENCODE WR AND TE ─────────────────────────────────────────────────
df['is_wr'] = (df['position'] == 'WR').astype(int)
df['is_te'] = (df['position'] == 'TE').astype(int)

# ─── SAVE OUTPUT ──────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"One-hot encoded file saved to: {OUTPUT_CSV}")