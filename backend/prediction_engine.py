from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "laliga_2025_full.csv"

RECENT_MATCHES = 5

GOAL_LONG_WEIGHT = 0.70
GOAL_RECENT_WEIGHT = 0.30

CORNER_LONG_WEIGHT = 0.80
CORNER_RECENT_WEIGHT = 0.20

CARD_LONG_WEIGHT = 0.70
CARD_RECENT_WEIGHT = 0.20
FOUL_RECENT_WEIGHT = 0.10


df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


TEAMS = sorted(
    set(df["home_team"])
    | set(df["away_team"])
)


# =========================================================
# GOALS
# =========================================================

def goal_long_strength(team):

    home_avg = df["home_goals"].mean()
    away_avg = df["away_goals"].mean()

    home = df[df["home_team"] == team]
    away = df[df["away_team"] == team]

    return {
        "home_attack":
            home["home_goals"].mean() / home_avg,

        "home_defense":
            home["away_goals"].mean() / away_avg,

        "away_attack":
            away["away_goals"].mean() / away_avg,

        "away_defense":
            away["home_goals"].mean() / home_avg
    }


def goal_recent_strength(team):

    matches = df[
        (df["home_team"] == team)
        | (df["away_team"] == team)
    ].tail(RECENT_MATCHES)

    scored = []
    conceded = []

    for _, match in matches.iterrows():

        if match["home_team"] == team:
            scored.append(match["home_goals"])
            conceded.append(match["away_goals"])

        else:
            scored.append(match["away_goals"])
            conceded.append(match["home_goals"])

    league_avg = (
        df["home_goals"].sum()
        + df["away_goals"].sum()
    ) / (2 * len(df))

    return {
        "attack":
            np.mean(scored) / league_avg,

        "defense":
            np.mean(conceded) / league_avg
    }


def predict_goals(home_team, away_team):

    home_long = goal_long_strength(home_team)
    away_long = goal_long_strength(away_team)

    home_recent = goal_recent_strength(home_team)
    away_recent = goal_recent_strength(away_team)

    home_attack = (
        GOAL_LONG_WEIGHT * home_long["home_attack"]
        + GOAL_RECENT_WEIGHT * home_recent["attack"]
    )

    home_defense = (
        GOAL_LONG_WEIGHT * home_long["home_defense"]
        + GOAL_RECENT_WEIGHT * home_recent["defense"]
    )

    away_attack = (
        GOAL_LONG_WEIGHT * away_long["away_attack"]
        + GOAL_RECENT_WEIGHT * away_recent["attack"]
    )

    away_defense = (
        GOAL_LONG_WEIGHT * away_long["away_defense"]
        + GOAL_RECENT_WEIGHT * away_recent["defense"]
    )

    home_lambda = (
        df["home_goals"].mean()
        * home_attack
        * away_defense
    )

    away_lambda = (
        df["away_goals"].mean()
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

    home_win = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, 1).sum()

    best_score = np.unravel_index(
        np.argmax(matrix),
        matrix.shape
    )

    scores = []

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):

            scores.append({
                "home_goals": h,
                "away_goals": a,
                "probability":
                    float(matrix[h, a] * 100)
            })

    scores.sort(
        key=lambda x: x["probability"],
        reverse=True
    )

    return {
        "expected_home_goals":
            round(float(home_lambda), 2),

        "expected_away_goals":
            round(float(away_lambda), 2),

        "home_win_probability":
            round(float(home_win * 100), 1),

        "draw_probability":
            round(float(draw * 100), 1),

        "away_win_probability":
            round(float(away_win * 100), 1),

        "most_likely_score": {
            "home": int(best_score[0]),
            "away": int(best_score[1])
        },

        "top_scores": scores[:5]
    }


# =========================================================
# CORNERS
# =========================================================

def corner_long_strength(team):

    home_avg = df["home_corners"].mean()
    away_avg = df["away_corners"].mean()

    home = df[df["home_team"] == team]
    away = df[df["away_team"] == team]

    return {
        "home_attack":
            home["home_corners"].mean() / home_avg,

        "home_defense":
            home["away_corners"].mean() / away_avg,

        "away_attack":
            away["away_corners"].mean() / away_avg,

        "away_defense":
            away["home_corners"].mean() / home_avg
    }


def corner_recent_strength(team):

    matches = df[
        (df["home_team"] == team)
        | (df["away_team"] == team)
    ].tail(RECENT_MATCHES)

    corners_for = []
    corners_against = []

    for _, match in matches.iterrows():

        if match["home_team"] == team:
            corners_for.append(match["home_corners"])
            corners_against.append(match["away_corners"])

        else:
            corners_for.append(match["away_corners"])
            corners_against.append(match["home_corners"])

    league_avg = (
        df["home_corners"].sum()
        + df["away_corners"].sum()
    ) / (2 * len(df))

    return {
        "attack":
            np.mean(corners_for) / league_avg,

        "defense":
            np.mean(corners_against) / league_avg
    }


def predict_corners(home_team, away_team):

    home_long = corner_long_strength(home_team)
    away_long = corner_long_strength(away_team)

    home_recent = corner_recent_strength(home_team)
    away_recent = corner_recent_strength(away_team)

    home_attack = (
        CORNER_LONG_WEIGHT * home_long["home_attack"]
        + CORNER_RECENT_WEIGHT * home_recent["attack"]
    )

    home_defense = (
        CORNER_LONG_WEIGHT * home_long["home_defense"]
        + CORNER_RECENT_WEIGHT * home_recent["defense"]
    )

    away_attack = (
        CORNER_LONG_WEIGHT * away_long["away_attack"]
        + CORNER_RECENT_WEIGHT * away_recent["attack"]
    )

    away_defense = (
        CORNER_LONG_WEIGHT * away_long["away_defense"]
        + CORNER_RECENT_WEIGHT * away_recent["defense"]
    )

    expected_home = (
        df["home_corners"].mean()
        * home_attack
        * away_defense
    )

    expected_away = (
        df["away_corners"].mean()
        * away_attack
        * home_defense
    )

    expected_total = (
        expected_home
        + expected_away
    )

    historical_total = (
        df["home_corners"]
        + df["away_corners"]
    )

    mean_total = historical_total.mean()
    variance_total = historical_total.var()

    alpha = (
        (variance_total - mean_total)
        / (mean_total ** 2)
    )

    alpha = max(alpha, 0.0001)

    n = 1 / alpha

    p = n / (
        n + expected_total
    )

    def over(line):

        cutoff = int(
            np.floor(line)
        )

        return float(
            1 - nbinom.cdf(
                cutoff,
                n,
                p
            )
        )

    values = np.arange(0, 26)

    probabilities = nbinom.pmf(
        values,
        n,
        p
    )

    most_likely = int(
        values[
            np.argmax(probabilities)
        ]
    )

    over_under = {}

    for line in [
        8.5,
        9.5,
        10.5,
        11.5
    ]:

        probability = over(line)

        over_under[str(line)] = {
            "over":
                round(probability * 100, 1),

            "under":
                round((1 - probability) * 100, 1)
        }

    return {
        "expected_home":
            round(float(expected_home), 2),

        "expected_away":
            round(float(expected_away), 2),

        "expected_total":
            round(float(expected_total), 2),

        "most_likely_total":
            most_likely,

        "over_under":
            over_under
    }


# =========================================================
# YELLOW CARDS
# =========================================================

def card_long_strength(team):

    home_avg = (
        df["home_yellow_cards"].mean()
    )

    away_avg = (
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
            / home_avg,

        "home_opponent_rate":
            home["away_yellow_cards"].mean()
            / away_avg,

        "away_card_rate":
            away["away_yellow_cards"].mean()
            / away_avg,

        "away_opponent_rate":
            away["home_yellow_cards"].mean()
            / home_avg
    }


def card_recent_form(team):

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


def predict_cards(home_team, away_team):

    home_long = card_long_strength(
        home_team
    )

    away_long = card_long_strength(
        away_team
    )

    home_recent = card_recent_form(
        home_team
    )

    away_recent = card_recent_form(
        away_team
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

    expected_home = (
        df["home_yellow_cards"].mean()
        * home_factor
        * away_long["away_opponent_rate"]
    )

    expected_away = (
        df["away_yellow_cards"].mean()
        * away_factor
        * home_long["home_opponent_rate"]
    )

    expected_total = (
        expected_home
        + expected_away
    )

    def over(line):

        cutoff = int(
            np.floor(line)
        )

        return float(
            1 - poisson.cdf(
                cutoff,
                expected_total
            )
        )

    values = np.arange(0, 16)

    probabilities = poisson.pmf(
        values,
        expected_total
    )

    most_likely = int(
        values[
            np.argmax(probabilities)
        ]
    )

    over_under = {}

    for line in [
        2.5,
        3.5,
        4.5,
        5.5,
        6.5
    ]:

        probability = over(line)

        over_under[str(line)] = {
            "over":
                round(probability * 100, 1),

            "under":
                round((1 - probability) * 100, 1)
        }

    return {
        "expected_home":
            round(float(expected_home), 2),

        "expected_away":
            round(float(expected_away), 2),

        "expected_total":
            round(float(expected_total), 2),

        "most_likely_total":
            most_likely,

        "over_under":
            over_under
    }


# =========================================================
# FULL MATCH
# =========================================================

def predict_full_match(
    home_team,
    away_team
):

    if home_team not in TEAMS:
        raise ValueError(
            f"Unknown home team: {home_team}"
        )

    if away_team not in TEAMS:
        raise ValueError(
            f"Unknown away team: {away_team}"
        )

    if home_team == away_team:
        raise ValueError(
            "Home and away teams must be different."
        )

    return {
        "match": {
            "home_team": home_team,
            "away_team": away_team
        },

        "goals":
            predict_goals(
                home_team,
                away_team
            ),

        "corners":
            predict_corners(
                home_team,
                away_team
            ),

        "yellow_cards":
            predict_cards(
                home_team,
                away_team
            )
    }