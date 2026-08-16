import pandas as pd
import requests
from bs4 import BeautifulSoup
from sportsdataio import NFL

# Concatenate per-year combine files
files = glob.glob('combine_data/*_combine.csv')
df_combine = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Merge official combine times
df_missing = pd.read_csv('Missing_40_Yard_Dash_Times.csv')
df = df_missing.merge(
    df_combine[['Player','DraftYear','FortyYardDash']],
    left_on=['name','draft_year'],
    right_on=['Player','DraftYear'],
    how='left'
).rename(columns={'FortyYardDash':'combine_40'})

# Fetch via API for still-missing entries
api = NFL(api_key='YOUR_KEY')
for idx, row in df[df['combine_40'].isna()].iterrows():
    result = api.get_combine_stats(player_name=row['name'])
    df.at[idx, 'api_40'] = result.get('forty_time')

# Scrape pro days for remaining gaps
def scrape_pro_day(name):
    url = 'https://bnbfootball.com/complete-pro-day-results-2025/'
    text = requests.get(url).text
    soup = BeautifulSoup(text, 'html.parser')
    # implement name-matching logic here...
    return None

df['proday_40'] = df.apply(
    lambda r: r['combine_40'] if pd.notnull(r['combine_40'])
    else scrape_pro_day(r['name']), axis=1
)

# Finalize by prioritizing sources
df['final_40'] = df['combine_40'].fillna(df['api_40']).fillna(df['proday_40'])
df.to_csv('filled_40_Yard_Dash_Times.csv', index=False)