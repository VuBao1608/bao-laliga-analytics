from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "laliga_2025_full.csv"

WARMUP_MATCHES = 100
RECENT_MATCHES = 5

LONG_TERM_WEIGHT = 0.70
RECENT_CARD_WEIGHT = 0.20
RECENT_FOUL_WEIGHT = 0.10


df = pd.read_csv(DATA_FILE)

df = df[
    df["home_yellow_cards"].notna()
    & df["away_yellow_cards"].notna()
    & df["home_fouls"].notna()
    & df["away_fouls"].notna()
].copy()

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    "date"
).reset_index(drop=True)


def get_long_term_strength(train, team):

    league_home_cards = (
        train["home_yellow_cards"].mean()
    )

    league_away_cards = (
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

        "home_card_rate":
            home_matches[
                "home_yellow_cards"
            ].mean()
            / league_home_cards,

        "home_opponent_rate":
            home_matches[
                "away_yellow_cards"
            ].mean()
            / league_away_cards,

        "away_card_rate":
            away_matches[
                "away_yellow_cards"
            ].mean()
            / league_away_cards,

        "away_opponent_rate":
            away_matches[
                "home_yellow_cards"
            ].mean()
            / league_home_cards
    }


def get_recent_form(train, team):

    matches = train[
        (train["home_team"] == team)
        | (train["away_team"] == team)
    ].tail(RECENT_MATCHES)

    if len(matches) == 0:
        return None

    cards = []
    fouls = []

    for _, match in matches.iterrows():

        if match["home_team"] == team:

            cards.append(
                match["home_yellow_cards"]
            )

            fouls.append(
                match["home_fouls"]
            )

        else:

            cards.append(
                match["away_yellow_cards"]
            )

            fouls.append(
                match["away_fouls"]
            )


    league_card_avg = (
        train["home_yellow_cards"].sum()
        + train["away_yellow_cards"].sum()
    ) / (2 * len(train))


    league_foul_avg = (
        train["home_fouls"].sum()
        + train["away_fouls"].sum()
    ) / (2 * len(train))


    return {

        "card_rate":
            np.mean(cards)
            / league_card_avg,

        "foul_rate":
            np.mean(fouls)
            / league_foul_avg
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


    home_long = get_long_term_strength(
        train,
        home_team
    )

    away_long = get_long_term_strength(
        train,
        away_team
    )

    home_recent = get_recent_form(
        train,
        home_team
    )

    away_recent = get_recent_form(
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


    # --------------------------------
    # Combine team card tendencies
    # --------------------------------

    home_card_factor = (

        LONG_TERM_WEIGHT
        * home_long["home_card_rate"]

        + RECENT_CARD_WEIGHT
        * home_recent["card_rate"]

        + RECENT_FOUL_WEIGHT
        * home_recent["foul_rate"]
    )


    away_card_factor = (

        LONG_TERM_WEIGHT
        * away_long["away_card_rate"]

        + RECENT_CARD_WEIGHT
        * away_recent["card_rate"]

        + RECENT_FOUL_WEIGHT
        * away_recent["foul_rate"]
    )


    league_home_cards = (
        train["home_yellow_cards"].mean()
    )

    league_away_cards = (
        train["away_yellow_cards"].mean()
    )


    # --------------------------------
    # Expected yellow cards
    # --------------------------------

    expected_home = (

        league_home_cards

        * home_card_factor

        * away_long[
            "away_opponent_rate"
        ]
    )


    expected_away = (

        league_away_cards

        * away_card_factor

        * home_long[
            "home_opponent_rate"
        ]
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
            actual_total,

        "home_recent_card_rate":
            home_recent["card_rate"],

        "away_recent_card_rate":
            away_recent["card_rate"],

        "home_recent_foul_rate":
            home_recent["foul_rate"],

        "away_recent_foul_rate":
            away_recent["foul_rate"]
    })


print("YELLOW CARD MODEL V2")
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
    / "card_v2_predictions.csv",
    index=False
)

print(
    "\nSaved card_v2_predictions.csv"
)