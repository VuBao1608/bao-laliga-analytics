from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "laliga_2025_full.csv"

WARMUP_MATCHES = 100
RECENT_MATCHES = 5

LONG_TERM_WEIGHT = 0.80
RECENT_WEIGHT = 0.20


df = pd.read_csv(DATA_FILE)

df = df[
    df["home_corners"].notna()
    & df["away_corners"].notna()
].copy()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


def get_long_strength(train, team):

    league_home_avg = train["home_corners"].mean()
    league_away_avg = train["away_corners"].mean()

    home = train[
        train["home_team"] == team
    ]

    away = train[
        train["away_team"] == team
    ]

    if len(home) == 0 or len(away) == 0:
        return None

    return {
        "home_attack":
            home["home_corners"].mean()
            / league_home_avg,

        "home_defense":
            home["away_corners"].mean()
            / league_away_avg,

        "away_attack":
            away["away_corners"].mean()
            / league_away_avg,

        "away_defense":
            away["home_corners"].mean()
            / league_home_avg
    }


def get_recent_strength(train, team):

    matches = train[
        (train["home_team"] == team)
        | (train["away_team"] == team)
    ].tail(RECENT_MATCHES)

    if len(matches) == 0:
        return None

    corners_for = []
    corners_against = []

    for _, match in matches.iterrows():

        if match["home_team"] == team:

            corners_for.append(
                match["home_corners"]
            )

            corners_against.append(
                match["away_corners"]
            )

        else:

            corners_for.append(
                match["away_corners"]
            )

            corners_against.append(
                match["home_corners"]
            )

    league_avg = (
        train["home_corners"].sum()
        + train["away_corners"].sum()
    ) / (2 * len(train))

    return {
        "attack":
            np.mean(corners_for)
            / league_avg,

        "defense":
            np.mean(corners_against)
            / league_avg
    }


home_errors = []
away_errors = []
total_errors = []

predictions = []


for i in range(WARMUP_MATCHES, len(df)):

    train = df.iloc[:i]
    match = df.iloc[i]

    home_team = match["home_team"]
    away_team = match["away_team"]

    home_long = get_long_strength(
        train,
        home_team
    )

    away_long = get_long_strength(
        train,
        away_team
    )

    home_recent = get_recent_strength(
        train,
        home_team
    )

    away_recent = get_recent_strength(
        train,
        away_team
    )

    if (
        home_long is None
        or away_long is None
        or home_recent is None
        or away_recent is None
    ):
        continue


    home_attack = (
        LONG_TERM_WEIGHT
        * home_long["home_attack"]
        + RECENT_WEIGHT
        * home_recent["attack"]
    )

    home_defense = (
        LONG_TERM_WEIGHT
        * home_long["home_defense"]
        + RECENT_WEIGHT
        * home_recent["defense"]
    )

    away_attack = (
        LONG_TERM_WEIGHT
        * away_long["away_attack"]
        + RECENT_WEIGHT
        * away_recent["attack"]
    )

    away_defense = (
        LONG_TERM_WEIGHT
        * away_long["away_defense"]
        + RECENT_WEIGHT
        * away_recent["defense"]
    )


    league_home_avg = train["home_corners"].mean()
    league_away_avg = train["away_corners"].mean()


    expected_home = (
        league_home_avg
        * home_attack
        * away_defense
    )

    expected_away = (
        league_away_avg
        * away_attack
        * home_defense
    )


    actual_home = match["home_corners"]
    actual_away = match["away_corners"]

    expected_total = expected_home + expected_away
    actual_total = actual_home + actual_away


    home_errors.append(
        abs(expected_home - actual_home)
    )

    away_errors.append(
        abs(expected_away - actual_away)
    )

    total_errors.append(
        abs(expected_total - actual_total)
    )


    predictions.append({
        "date": match["date"],
        "home_team": home_team,
        "away_team": away_team,

        "expected_home_corners": expected_home,
        "expected_away_corners": expected_away,
        "expected_total_corners": expected_total,

        "actual_home_corners": actual_home,
        "actual_away_corners": actual_away,
        "actual_total_corners": actual_total
    })


print("CORNER MODEL V2 - RECENT FORM")
print("-" * 40)

print(
    "Backtest matches:",
    len(predictions)
)

print(
    "Home corner MAE:",
    round(np.mean(home_errors), 3)
)

print(
    "Away corner MAE:",
    round(np.mean(away_errors), 3)
)

print(
    "Total corner MAE:",
    round(np.mean(total_errors), 3)
)


pd.DataFrame(
    predictions
).to_csv(
    BASE_DIR
    / "data"
    / "corner_v2_predictions.csv",
    index=False
)

print("\nSaved corner_v2_predictions.csv")