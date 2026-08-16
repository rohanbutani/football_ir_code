from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import pandas as pd
try:
    import nfl_data_py as nfl
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"{Path(__file__).name} requires nfl_data_py. Install it before running this script."
    ) from exc

# Load your merged player-season dataset to extract valid seasons
df = pd.read_csv(str(ROOT / "data/intermediate/player_enrichment/player_season_with_snaps.csv"))
seasons = df["season"].dropna().unique().tolist()

# Load weekly-level snap count data for those seasons
snap_weekly = nfl.import_snap_counts(seasons)

# Show the first few rows and column structure for inspection
print("✅ Snap weekly data loaded:")
print(snap_weekly.head())
print("\nColumns:", snap_weekly.columns.tolist())

# Optional: Save to CSV for inspection or backup
snap_weekly.to_csv(str(ROOT / "data/raw/snaps/snap_weekly_full.csv"), index=False)