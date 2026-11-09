from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

data_file = BASE_DIR / "data" / "laliga_2025.csv"

df = pd.read_csv(data_file)

# Only matches that already have a final score
completed = df[
    df["home_goals"].notna() &
    df["away_goals"].notna()
].copy()


print("Total matches:", len(df))
print("Completed matches:", len(completed))

print()

avg_home_goals = completed["home_goals"].mean()
avg_away_goals = completed["away_goals"].mean()
avg_total_goals = (
    completed["home_goals"] +
    completed["away_goals"]
).mean()

print("Average home goals:", round(avg_home_goals, 3))
print("Average away goals:", round(avg_away_goals, 3))
print("Average total goals:", round(avg_total_goals, 3))

print()

home_wins = (completed["home_goals"] > completed["away_goals"]).sum()
draws = (completed["home_goals"] == completed["away_goals"]).sum()
away_wins = (completed["home_goals"] < completed["away_goals"]).sum()

total = len(completed)

print("Home win %:", round(home_wins / total * 100, 1))
print("Draw %:", round(draws / total * 100, 1))
print("Away win %:", round(away_wins / total * 100, 1))