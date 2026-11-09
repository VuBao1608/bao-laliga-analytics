from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom


# =========================================================
# SETTINGS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "laliga_2025_full.csv"

RECENT_MATCHES = 5

# Goals
GOAL_LONG_WEIGHT = 0.70
GOAL_RECENT_WEIGHT = 0.30

# Corners
CORNER_LONG_WEIGHT = 0.80
CORNER_RECENT_WEIGHT = 0.20

# Cards
CARD_LONG_WEIGHT = 0.70
CARD_RECENT_WEIGHT = 0.20
FOUL_RECENT_WEIGHT = 0.10


# Temporary test match
HOME_TEAM = "Atletico Madrid"
AWAY_TEAM = "Villarreal"


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    "date"
).reset_index(drop=True)


teams = sorted(
    set(df["home_team"])
    | set(df["away_team"])
)


if HOME_TEAM not in teams:
    raise ValueError(
        f"Unknown home team: {HOME_TEAM}"
    )

if AWAY_TEAM not in teams:
    raise ValueError(
        f"Unknown away team: {AWAY_TEAM}"
    )


# =========================================================
# GOALS MODEL
# =========================================================

def get_goal_long_strength(team):

    league_home_avg = (
        df["home_goals"].mean()
    )

    league_away_avg = (
        df["away_goals"].mean()
    )

    home = df[
        df["home_team"] == team
    ]

    away = df[
        df["away_team"] == team
    ]

    return {
        "home_attack":
            home["home_goals"].mean()
            / league_home_avg,

        "home_defense":
            home["away_goals"].mean()
            / league_away_avg,

        "away_attack":
            away["away_goals"].mean()
            / league_away_avg,

        "away_defense":
            away["home_goals"].mean()
            / league_home_avg
    }


def get_goal_recent_strength(team):

    matches = df[
        (df["home_team"] == team)
        | (df["away_team"] == team)
    ].tail(RECENT_MATCHES)

    scored = []
    conceded = []

    for _, match in matches.iterrows():

        if match["home_team"] == team:

            scored.append(
                match["home_goals"]
            )

            conceded.append(
                match["away_goals"]
            )

        else:

            scored.append(
                match["away_goals"]
            )

            conceded.append(
                match["home_goals"]
            )


    league_avg = (
        df["home_goals"].sum()
        + df["away_goals"].sum()
    ) / (2 * len(df))


    return {
        "attack":
            np.mean(scored)
            / league_avg,

        "defense":
            np.mean(conceded)
            / league_avg
    }


def predict_goals():

    home_long = get_goal_long_strength(
        HOME_TEAM
    )

    away_long = get_goal_long_strength(
        AWAY_TEAM
    )

    home_recent = get_goal_recent_strength(
        HOME_TEAM
    )

    away_recent = get_goal_recent_strength(
        AWAY_TEAM
    )


    home_attack = (
        GOAL_LONG_WEIGHT
        * home_long["home_attack"]

        + GOAL_RECENT_WEIGHT
        * home_recent["attack"]
    )


    home_defense = (
        GOAL_LONG_WEIGHT
        * home_long["home_defense"]

        + GOAL_RECENT_WEIGHT
        * home_recent["defense"]
    )


    away_attack = (
        GOAL_LONG_WEIGHT
        * away_long["away_attack"]

        + GOAL_RECENT_WEIGHT
        * away_recent["attack"]
    )


    away_defense = (
        GOAL_LONG_WEIGHT
        * away_long["away_defense"]

        + GOAL_RECENT_WEIGHT
        * away_recent["defense"]
    )


    league_home_avg = (
        df["home_goals"].mean()
    )

    league_away_avg = (
        df["away_goals"].mean()
    )


    home_lambda = (
        league_home_avg
        * home_attack
        * away_defense
    )


    away_lambda = (
        league_away_avg
        * away_attack
        * home_defense
    )


    max_goals = 10

    home_probs = poisson.pmf(
        np.arange(max_goals + 1),
        home_lambda
    )

    away_probs = poisson.pmf(
        np.arange(max_goals + 1),
        away_lambda
    )


    matrix = np.outer(
        home_probs,
        away_probs
    )

    matrix /= matrix.sum()


    home_win = np.tril(
        matrix,
        -1
    ).sum()

    draw = np.trace(
        matrix
    )

    away_win = np.triu(
        matrix,
        1
    ).sum()


    best_score = np.unravel_index(
        np.argmax(matrix),
        matrix.shape
    )


    score_predictions = []

    for home_goals in range(
        max_goals + 1
    ):

        for away_goals in range(
            max_goals + 1
        ):

            score_predictions.append(
                (
                    matrix[
                        home_goals,
                        away_goals
                    ],
                    home_goals,
                    away_goals
                )
            )


    score_predictions.sort(
        reverse=True
    )


    return {
        "home_xg": home_lambda,
        "away_xg": away_lambda,

        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,

        "best_home_goals":
            best_score[0],

        "best_away_goals":
            best_score[1],

        "top_scores":
            score_predictions[:5]
    }


# =========================================================
# CORNER MODEL
# =========================================================

def get_corner_long_strength(team):

    league_home_avg = (
        df["home_corners"].mean()
    )

    league_away_avg = (
        df["away_corners"].mean()
    )

    home = df[
        df["home_team"] == team
    ]

    away = df[
        df["away_team"] == team
    ]

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


def get_corner_recent_strength(team):

    matches = df[
        (df["home_team"] == team)
        | (df["away_team"] == team)
    ].tail(RECENT_MATCHES)

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
        df["home_corners"].sum()
        + df["away_corners"].sum()
    ) / (2 * len(df))


    return {
        "attack":
            np.mean(corners_for)
            / league_avg,

        "defense":
            np.mean(corners_against)
            / league_avg
    }


def predict_corners():

    home_long = get_corner_long_strength(
        HOME_TEAM
    )

    away_long = get_corner_long_strength(
        AWAY_TEAM
    )

    home_recent = get_corner_recent_strength(
        HOME_TEAM
    )

    away_recent = get_corner_recent_strength(
        AWAY_TEAM
    )


    home_attack = (
        CORNER_LONG_WEIGHT
        * home_long["home_attack"]

        + CORNER_RECENT_WEIGHT
        * home_recent["attack"]
    )


    home_defense = (
        CORNER_LONG_WEIGHT
        * home_long["home_defense"]

        + CORNER_RECENT_WEIGHT
        * home_recent["defense"]
    )


    away_attack = (
        CORNER_LONG_WEIGHT
        * away_long["away_attack"]

        + CORNER_RECENT_WEIGHT
        * away_recent["attack"]
    )


    away_defense = (
        CORNER_LONG_WEIGHT
        * away_long["away_defense"]

        + CORNER_RECENT_WEIGHT
        * away_recent["defense"]
    )


    league_home_avg = (
        df["home_corners"].mean()
    )

    league_away_avg = (
        df["away_corners"].mean()
    )


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


    expected_total = (
        expected_home
        + expected_away
    )


    # Negative Binomial dispersion
    historical_total = (
        df["home_corners"]
        + df["away_corners"]
    )

    mean_total = (
        historical_total.mean()
    )

    var_total = (
        historical_total.var()
    )


    alpha = (
        (var_total - mean_total)
        / (mean_total ** 2)
    )

    alpha = max(
        alpha,
        0.0001
    )


    n = 1 / alpha

    p = (
        n
        / (n + expected_total)
    )


    def over(line):

        cutoff = int(
            np.floor(line)
        )

        return (
            1
            - nbinom.cdf(
                cutoff,
                n,
                p
            )
        )


    corner_values = np.arange(
        0,
        26
    )

    probabilities = nbinom.pmf(
        corner_values,
        n,
        p
    )


    most_likely = corner_values[
        np.argmax(probabilities)
    ]


    top_indices = np.argsort(
        probabilities
    )[::-1][:5]


    top_totals = []

    for index in top_indices:

        top_totals.append(
            (
                corner_values[index],
                probabilities[index]
            )
        )


    return {
        "expected_home": expected_home,
        "expected_away": expected_away,
        "expected_total": expected_total,

        "most_likely_total":
            most_likely,

        "top_totals":
            top_totals,

        "over_8_5": over(8.5),
        "over_9_5": over(9.5),
        "over_10_5": over(10.5),
        "over_11_5": over(11.5),

        "alpha": alpha
    }


# =========================================================
# CARD MODEL
# =========================================================

def get_card_long_strength(team):

    league_home_avg = (
        df["home_yellow_cards"].mean()
    )

    league_away_avg = (
        df["away_yellow_cards"].mean()
    )

    home = df[
        df["home_team"] == team
    ]

    away = df[
        df["away_team"] == team
    ]


    return {
        "home_card_rate":
            home["home_yellow_cards"].mean()
            / league_home_avg,

        "home_opponent_rate":
            home["away_yellow_cards"].mean()
            / league_away_avg,

        "away_card_rate":
            away["away_yellow_cards"].mean()
            / league_away_avg,

        "away_opponent_rate":
            away["home_yellow_cards"].mean()
            / league_home_avg
    }


def get_card_recent_form(team):

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


def predict_cards():

    home_long = get_card_long_strength(
        HOME_TEAM
    )

    away_long = get_card_long_strength(
        AWAY_TEAM
    )

    home_recent = get_card_recent_form(
        HOME_TEAM
    )

    away_recent = get_card_recent_form(
        AWAY_TEAM
    )


    home_factor = (
        CARD_LONG_WEIGHT
        * home_long["home_card_rate"]

        + CARD_RECENT_WEIGHT
        * home_recent["card_rate"]

        + FOUL_RECENT_WEIGHT
        * home_recent["foul_rate"]
    )


    away_factor = (
        CARD_LONG_WEIGHT
        * away_long["away_card_rate"]

        + CARD_RECENT_WEIGHT
        * away_recent["card_rate"]

        + FOUL_RECENT_WEIGHT
        * away_recent["foul_rate"]
    )


    league_home_avg = (
        df["home_yellow_cards"].mean()
    )

    league_away_avg = (
        df["away_yellow_cards"].mean()
    )


    expected_home = (
        league_home_avg
        * home_factor
        * away_long[
            "away_opponent_rate"
        ]
    )


    expected_away = (
        league_away_avg
        * away_factor
        * home_long[
            "home_opponent_rate"
        ]
    )


    expected_total = (
        expected_home
        + expected_away
    )


    def over(line):

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


    card_values = np.arange(
        0,
        16
    )


    probabilities = poisson.pmf(
        card_values,
        expected_total
    )


    most_likely = card_values[
        np.argmax(probabilities)
    ]


    top_indices = np.argsort(
        probabilities
    )[::-1][:5]


    top_totals = []

    for index in top_indices:

        top_totals.append(
            (
                card_values[index],
                probabilities[index]
            )
        )


    return {
        "expected_home":
            expected_home,

        "expected_away":
            expected_away,

        "expected_total":
            expected_total,

        "most_likely_total":
            most_likely,

        "top_totals":
            top_totals,

        "over_2_5": over(2.5),
        "over_3_5": over(3.5),
        "over_4_5": over(4.5),
        "over_5_5": over(5.5),
        "over_6_5": over(6.5)
    }


# =========================================================
# RUN ALL MODELS
# =========================================================

goals = predict_goals()
corners = predict_corners()
cards = predict_cards()


# =========================================================
# OUTPUT
# =========================================================

print()
print("=" * 60)
print(
    f"{HOME_TEAM} vs {AWAY_TEAM}"
)
print("=" * 60)


# ---------------- GOALS ----------------

print("\nGOALS")
print("-" * 60)

print(
    f"Expected goals: "
    f"{goals['home_xg']:.2f} - "
    f"{goals['away_xg']:.2f}"
)

print(
    f"{HOME_TEAM} win: "
    f"{goals['home_win'] * 100:.1f}%"
)

print(
    f"Draw: "
    f"{goals['draw'] * 100:.1f}%"
)

print(
    f"{AWAY_TEAM} win: "
    f"{goals['away_win'] * 100:.1f}%"
)

print(
    "Most likely score: "
    f"{goals['best_home_goals']}"
    "-"
    f"{goals['best_away_goals']}"
)


print("\nTop score predictions:")

for probability, home_goals, away_goals in (
    goals["top_scores"]
):

    print(
        f"{home_goals}-{away_goals}: "
        f"{probability * 100:.1f}%"
    )


# ---------------- CORNERS ----------------

print("\nCORNERS")
print("-" * 60)

print(
    f"{HOME_TEAM}: "
    f"{corners['expected_home']:.2f}"
)

print(
    f"{AWAY_TEAM}: "
    f"{corners['expected_away']:.2f}"
)

print(
    f"Expected total: "
    f"{corners['expected_total']:.2f}"
)

print(
    f"Most likely total: "
    f"{corners['most_likely_total']}"
)


print("\nOver / Under:")

for line, probability in [
    (8.5, corners["over_8_5"]),
    (9.5, corners["over_9_5"]),
    (10.5, corners["over_10_5"]),
    (11.5, corners["over_11_5"])
]:

    print(
        f"{line}: "
        f"Over {probability * 100:.1f}% | "
        f"Under {(1 - probability) * 100:.1f}%"
    )


# ---------------- CARDS ----------------

print("\nYELLOW CARDS")
print("-" * 60)

print(
    f"{HOME_TEAM}: "
    f"{cards['expected_home']:.2f}"
)

print(
    f"{AWAY_TEAM}: "
    f"{cards['expected_away']:.2f}"
)

print(
    f"Expected total: "
    f"{cards['expected_total']:.2f}"
)

print(
    f"Most likely total: "
    f"{cards['most_likely_total']}"
)


print("\nOver / Under:")

for line, probability in [
    (2.5, cards["over_2_5"]),
    (3.5, cards["over_3_5"]),
    (4.5, cards["over_4_5"]),
    (5.5, cards["over_5_5"]),
    (6.5, cards["over_6_5"])
]:

    print(
        f"{line}: "
        f"Over {probability * 100:.1f}% | "
        f"Under {(1 - probability) * 100:.1f}%"
    )


print()
print("=" * 60)