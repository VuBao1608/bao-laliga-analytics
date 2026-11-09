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
    df["home_yellow_cards"].notna()
    & df["away_yellow_cards"].notna()
].copy()

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    "date"
).reset_index(drop=True)


def get_card_strength(train, team):

    league_home_avg = (
        train["home_yellow_cards"].mean()
    )

    league_away_avg = (
        train["away_yellow_cards"].mean()
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
        # How many cards this team receives at home
        "home_card_rate":
            home_matches[
                "home_yellow_cards"
            ].mean()
            / league_home_avg,

        # How many cards opponents receive
        # when playing at this team's home
        "home_opponent_rate":
            home_matches[
                "away_yellow_cards"
            ].mean()
            / league_away_avg,

        # How many cards this team receives away
        "away_card_rate":
            away_matches[
                "away_yellow_cards"
            ].mean()
            / league_away_avg,

        # How many cards home opponents receive
        # against this team
        "away_opponent_rate":
            away_matches[
                "home_yellow_cards"
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

    home_strength = get_card_strength(
        train,
        home_team
    )

    away_strength = get_card_strength(
        train,
        away_team
    )

    if (
        home_strength is None
        or away_strength is None
    ):
        continue

    league_home_avg = (
        train["home_yellow_cards"].mean()
    )

    league_away_avg = (
        train["away_yellow_cards"].mean()
    )

    expected_home = (
        league_home_avg
        * home_strength["home_card_rate"]
        * away_strength["away_opponent_rate"]
    )

    expected_away = (
        league_away_avg
        * away_strength["away_card_rate"]
        * home_strength["home_opponent_rate"]
    )

    actual_home = (
        match["home_yellow_cards"]
    )

    actual_away = (
        match["away_yellow_cards"]
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

        "expected_home_cards":
            expected_home,

        "expected_away_cards":
            expected_away,

        "expected_total_cards":
            expected_total,

        "actual_home_cards":
            actual_home,

        "actual_away_cards":
            actual_away,

        "actual_total_cards":
            actual_total
    })


print("YELLOW CARD MODEL V1")
print("-" * 40)

print(
    "Backtest matches:",
    len(predictions)
)

print(
    "Home card MAE:",
    round(
        np.mean(home_errors),
        3
    )
)

print(
    "Away card MAE:",
    round(
        np.mean(away_errors),
        3
    )
)

print(
    "Total card MAE:",
    round(
        np.mean(total_errors),
        3
    )
)


pd.DataFrame(
    predictions
).to_csv(
    BASE_DIR
    / "data"
    / "card_v1_predictions.csv",
    index=False
)

print(
    "\nSaved card_v1_predictions.csv"
)