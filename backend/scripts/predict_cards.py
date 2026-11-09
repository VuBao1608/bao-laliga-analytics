from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "laliga_2025_full.csv"

RECENT_MATCHES = 5

LONG_TERM_WEIGHT = 0.70
RECENT_CARD_WEIGHT = 0.20
RECENT_FOUL_WEIGHT = 0.10


# Temporary test match
HOME_TEAM = "Atletico Madrid"
AWAY_TEAM = "Villarreal"


df = pd.read_csv(DATA_FILE)

df = df[
    df["home_yellow_cards"].notna()
    & df["away_yellow_cards"].notna()
    & df["home_fouls"].notna()
    & df["away_fouls"].notna()
].copy()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


def get_long_term_strength(team):

    league_home_cards = df["home_yellow_cards"].mean()
    league_away_cards = df["away_yellow_cards"].mean()

    home = df[
        df["home_team"] == team
    ]

    away = df[
        df["away_team"] == team
    ]

    return {
        "home_card_rate":
            home["home_yellow_cards"].mean()
            / league_home_cards,

        "home_opponent_rate":
            home["away_yellow_cards"].mean()
            / league_away_cards,

        "away_card_rate":
            away["away_yellow_cards"].mean()
            / league_away_cards,

        "away_opponent_rate":
            away["home_yellow_cards"].mean()
            / league_home_cards
    }


def get_recent_form(team):

    matches = df[
        (df["home_team"] == team)
        | (df["away_team"] == team)
    ].tail(RECENT_MATCHES)

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
        df["home_yellow_cards"].sum()
        + df["away_yellow_cards"].sum()
    ) / (2 * len(df))


    league_foul_avg = (
        df["home_fouls"].sum()
        + df["away_fouls"].sum()
    ) / (2 * len(df))


    return {
        "card_rate":
            np.mean(cards)
            / league_card_avg,

        "foul_rate":
            np.mean(fouls)
            / league_foul_avg
    }


home_long = get_long_term_strength(
    HOME_TEAM
)

away_long = get_long_term_strength(
    AWAY_TEAM
)

home_recent = get_recent_form(
    HOME_TEAM
)

away_recent = get_recent_form(
    AWAY_TEAM
)


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
    df["home_yellow_cards"].mean()
)

league_away_cards = (
    df["away_yellow_cards"].mean()
)


expected_home = (
    league_home_cards
    * home_card_factor
    * away_long["away_opponent_rate"]
)


expected_away = (
    league_away_cards
    * away_card_factor
    * home_long["home_opponent_rate"]
)


expected_total = (
    expected_home
    + expected_away
)


print()
print(
    f"{HOME_TEAM} vs {AWAY_TEAM}"
)

print("-" * 45)

print("\nExpected yellow cards:")

print(
    f"{HOME_TEAM}: "
    f"{expected_home:.2f}"
)

print(
    f"{AWAY_TEAM}: "
    f"{expected_away:.2f}"
)

print(
    f"Total: "
    f"{expected_total:.2f}"
)


# --------------------------------
# Poisson probabilities
# --------------------------------

def over_probability(line):

    cutoff = int(
        np.floor(line)
    )

    return (
        1
        - poisson.cdf(
            cutoff,
            expected_total
        )
    )


def under_probability(line):

    cutoff = int(
        np.floor(line)
    )

    return poisson.cdf(
        cutoff,
        expected_total
    )


print(
    "\nTotal yellow card probabilities:"
)

for line in [
    2.5,
    3.5,
    4.5,
    5.5,
    6.5
]:

    over = over_probability(line)
    under = under_probability(line)

    print(
        f"Line {line}: "
        f"Over {over * 100:.1f}% | "
        f"Under {under * 100:.1f}%"
    )


# --------------------------------
# Most likely total cards
# --------------------------------

max_cards = 15

card_values = np.arange(
    0,
    max_cards + 1
)

card_probabilities = poisson.pmf(
    card_values,
    expected_total
)


most_likely_total = (
    card_values[
        np.argmax(
            card_probabilities
        )
    ]
)


print(
    "\nMost likely total yellow cards:",
    most_likely_total
)


top_indices = np.argsort(
    card_probabilities
)[::-1][:5]


print(
    "\nTop 5 total yellow card predictions:"
)

for index in top_indices:

    cards = card_values[index]

    probability = (
        card_probabilities[index]
    )

    print(
        f"{cards} cards: "
        f"{probability * 100:.1f}%"
    )