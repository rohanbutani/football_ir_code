from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd

# --- Set input and output file paths here ---
INPUT_CSV = str(ROOT / "data/intermediate/injury_matching/ir_player_names_deduplicated.csv")
OUTPUT_CSV = str(ROOT / "data/intermediate/injury_matching/ir_player_names_deduplicated_cleaned.csv")


def clean_name(name: str) -> str:
    """
    Cleans a player name by:
    1. Removing the last token if it already appears earlier in the name.
    2. Deduplicating all tokens (case insensitive).
    """
    if not isinstance(name, str) or not name.strip():
        return name  # Leave empty or non-string values unchanged

    parts = name.split()
    lower_parts = [p.lower() for p in parts]

    # Remove trailing last name if it appears earlier
    if len(parts) > 1 and lower_parts[-1] in lower_parts[:-1]:
        parts = parts[:-1]

    # Deduplicate tokens while preserving order
    seen = set()
    cleaned = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            cleaned.append(p)
            seen.add(key)

    return " ".join(cleaned)


# --- Load, process, and save ---
df = pd.read_csv(INPUT_CSV, dtype=str)

if 'player_clean_extracted' not in df.columns:
    raise KeyError("Input CSV must have a column named 'player_clean_extracted'")

df['player_clean_cleaned'] = df['player_clean_extracted'].apply(clean_name)

df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Cleaned data saved to: {OUTPUT_CSV}")