from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

import requests
import pandas as pd

all_data = []

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://nextgenstats.nfl.com/stats/receiving"
}

for year in range(2018, 2025):  # You can change this range if needed
    url = f"https://nextgenstats.nfl.com/api/statboard/receiving?season={year}&seasonType=REG"
    print(f"Fetching data for {year}...")
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        json_data = response.json()
        for player in json_data['stats']:
            player['season'] = year
            all_data.append(player)
    else:
        print(f"Failed to fetch data for {year}")

# Convert and export
df = pd.DataFrame(all_data)
df.to_csv(str(ROOT / "data/raw/nextgen/nfl_receiving_nextgen_2018_2024.csv"), index=False)
print("✅ Data exported to nfl_receiving_nextgen_2018_2024.csv")