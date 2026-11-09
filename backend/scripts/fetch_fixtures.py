import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("API_FOOTBALL_KEY")

url = "https://v3.football.api-sports.io/fixtures"

headers = {
    "x-apisports-key": API_KEY
}

params = {
    "league": 140,
    "season": 2025
}

response = requests.get(
    url,
    headers=headers,
    params=params
)

data = response.json()

print("Errors:", data.get("errors"))
print("Number of matches:", data.get("results"))


matches = []

for match in data["response"]:
    matches.append({
        "fixture_id": match["fixture"]["id"],
        "date": match["fixture"]["date"],
        "status": match["fixture"]["status"]["short"],

        "home_team_id": match["teams"]["home"]["id"],
        "home_team": match["teams"]["home"]["name"],

        "away_team_id": match["teams"]["away"]["id"],
        "away_team": match["teams"]["away"]["name"],

        "home_goals": match["goals"]["home"],
        "away_goals": match["goals"]["away"]
    })


df = pd.DataFrame(matches)

df = df.sort_values("date")

output_file = BASE_DIR / "data" / "laliga_2025.csv"

df.to_csv(
    output_file,
    index=False
)

print()
print("Saved dataset to:")
print(output_file)

print()
print(df.head(10))