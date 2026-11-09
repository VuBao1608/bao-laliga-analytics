from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "laliga_2025_full.csv"
)

WARMUP_MATCHES = 100


df = pd.read_csv(DATA_FILE)

df = df[
    df["home_corners"].notna()
    & df["away_corners"].notna()
].copy()

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    "date"
).reset_index(drop=True)


def get_corner_strength(train, team):

    league_home_avg = (
        train["home_corners"].mean()
    )

    league_away_avg = (
        train["away_corners"].mean()
    )

    home_matches = train[
        train["home_team"] == team
    ]

    away_matches = train[
        train["away_team"] == team
    ]

    if (
        len(home_matches) == 0
        or len(away_matches) == 0
    ):
        return None

    return {
        "home_attack":
            home_matches[
                "home_corners"
            ].mean()
            / league_home_avg,

        "home_defense":
            home_matches[
                "away_corners"
            ].mean()
            / league_away_avg,

        "away_attack":
            away_matches[
                "away_corners"
            ].mean()
            / league_away_avg,

        "away_defense":
            away_matches[
                "home_corners"
            ].mean()
            / league_home_avg
    }


home_errors = []
away_errors = []
total_errors = []

predictions = []


for i in range(
    WARMUP_MATCHES,
    len(df)
):

    train = df.iloc[:i]
    match = df.iloc[i]

    home_team = match["home_team"]
    away_team = match["away_team"]

    home_strength = get_corner_strength(
        train,
        home_team
    )

    away_strength = get_corner_strength(
        train,
        away_team
    )

    if (
        home_strength is None
        or away_strength is None
    ):
        continue

    league_home_avg = (
        train["home_corners"].mean()
    )

    league_away_avg = (
        train["away_corners"].mean()
    )

    expected_home = (
        league_home_avg
        * home_strength["home_attack"]
        * away_strength["away_defense"]
    )

    expected_away = (
        league_away_avg
        * away_strength["away_attack"]
        * home_strength["home_defense"]
    )

    actual_home = (
        match["home_corners"]
    )

    actual_away = (
        match["away_corners"]
    )

    expected_total = (
        expected_home
        + expected_away
    )

    actual_total = (
        actual_home
        + actual_away
    )

    home_errors.append(
        abs(
            expected_home
            - actual_home
        )
    )

    away_errors.append(
        abs(
            expected_away
            - actual_away
        )
    )

    total_errors.append(
        abs(
            expected_total
            - actual_total
        )
    )

    predictions.append({
        "date": match["date"],
        "home_team": home_team,
        "away_team": away_team,

        "expected_home_corners":
            expected_home,

        "expected_away_corners":
            expected_away,

        "expected_total_corners":
            expected_total,

        "actual_home_corners":
            actual_home,

        "actual_away_corners":
            actual_away,

        "actual_total_corners":
            actual_total
    })


print("CORNER MODEL V1")
print("-" * 40)

print(
    "Backtest matches:",
    len(predictions)
)

print(
    "Home corner MAE:",
    round(
        np.mean(home_errors),
        3
    )
)

print(
    "Away corner MAE:",
    round(
        np.mean(away_errors),
        3
    )
)

print(
    "Total corner MAE:",
    round(
        np.mean(total_errors),
        3
    )
)


output = pd.DataFrame(
    predictions
)

output.to_csv(
    BASE_DIR
    / "data"
    / "corner_v1_predictions.csv",
    index=False
)

print(
    "\nSaved corner_v1_predictions.csv"
)