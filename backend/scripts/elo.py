from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "laliga_2025.csv"

INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 60


df = pd.read_csv(DATA_FILE)

df = df[
    df["home_goals"].notna()
    & df["away_goals"].notna()
].copy()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


teams = sorted(
    set(df["home_team"])
    | set(df["away_team"])
)

ratings = {
    team: INITIAL_ELO
    for team in teams
}


def expected_score(rating_a, rating_b):
    return 1 / (
        1 + 10 ** ((rating_b - rating_a) / 400)
    )


for _, match in df.iterrows():

    home_team = match["home_team"]
    away_team = match["away_team"]

    home_rating = ratings[home_team]
    away_rating = ratings[away_team]

    expected_home = expected_score(
        home_rating + HOME_ADVANTAGE,
        away_rating
    )

    expected_away = 1 - expected_home

    home_goals = match["home_goals"]
    away_goals = match["away_goals"]

    if home_goals > away_goals:
        actual_home = 1.0
        actual_away = 0.0

    elif home_goals == away_goals:
        actual_home = 0.5
        actual_away = 0.5

    else:
        actual_home = 0.0
        actual_away = 1.0

    ratings[home_team] = (
        home_rating
        + K_FACTOR * (actual_home - expected_home)
    )

    ratings[away_team] = (
        away_rating
        + K_FACTOR * (actual_away - expected_away)
    )


ranking = pd.DataFrame({
    "team": ratings.keys(),
    "elo": ratings.values()
})

ranking = ranking.sort_values(
    "elo",
    ascending=False
).reset_index(drop=True)

ranking["rank"] = ranking.index + 1


print(
    ranking[
        ["rank", "team", "elo"]
    ].round(1).to_string(index=False)
)