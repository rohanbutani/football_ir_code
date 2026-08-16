from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd

# —————————————————————————
# File paths (edit if needed)
IN_CSV  = rstr(ROOT / "data/intermediate/player_enrichment/player_season_with_forty_yard_merged2.csv")
OUT_CSV = rstr(ROOT / "data/intermediate/player_enrichment/player_season_with_forty_dedup.csv")
# —————————————————————————

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate rows based on playerName and season,
    keeping the first occurrence.
    """
    return df.drop_duplicates(subset=['playerName', 'season'], keep='first')

def main():
    # 1. Load the merged data
    df = pd.read_csv(IN_CSV)

    # 2. Deduplicate
    before = len(df)
    df_clean = deduplicate(df)
    after = len(df_clean)

    print(f"Rows before deduplication: {before}")
    print(f"Rows after  deduplication: {after}")
    print(f"Dropped {before - after} duplicate rows.")

    # 3. Save the clean file
    df_clean.to_csv(OUT_CSV, index=False)
    print(f"Deduplicated file saved to: {OUT_CSV}")

if __name__ == '__main__':
    main()