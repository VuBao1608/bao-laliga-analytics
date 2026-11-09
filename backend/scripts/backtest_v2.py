from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "laliga_2025.csv"

WARMUP_MATCHES = 100
MAX_GOALS = 10

RECENT_MATCHES = 5

LONG_TERM_WEIGHT = 0.75
RECENT_WEIGHT = 0.25


df = pd.read_csv(DATA_FILE)

df = df[
    df["home_goals"].notna()
    & df["away_goals"].notna()
].copy()

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


def get_long_term_strength(train, team):

    league_home_avg = train["home_goals"].mean()
    league_away_avg = train["away_goals"].mean()

    home_matches = train[
        train["home_team"] == team
    ]

    away_matches = train[
        train["away_team"] == team
    ]

    if len(home_matches) == 0 or len(away_matches) == 0:
        return None

    return {
        "home_attack":
            home_matches["home_goals"].mean()
            / league_home_avg,

        "home_defense":
            home_matches["away_goals"].mean()
            / league_away_avg,

        "away_attack":
            away_matches["away_goals"].mean()
            / league_away_avg,

        "away_defense":
            away_matches["home_goals"].mean()
            / league_home_avg
    }


def get_recent_strength(train, team):

    team_matches = train[
        (train["home_team"] == team)
        | (train["away_team"] == team)
    ].tail(RECENT_MATCHES)

    if len(team_matches) == 0:
        return None

    goals_scored = []
    goals_conceded = []

    for _, match in team_matches.iterrows():

        if match["home_team"] == team:

            goals_scored.append(
                match["home_goals"]
            )

            goals_conceded.append(
                match["away_goals"]
            )

        else:

            goals_scored.append(
                match["away_goals"]
            )

            goals_conceded.append(
                match["home_goals"]
            )


    league_goal_avg = (
        train["home_goals"].sum()
        + train["away_goals"].sum()
    ) / (2 * len(train))


    recent_attack = (
        np.mean(goals_scored)
        / league_goal_avg
    )

    recent_defense = (
        np.mean(goals_conceded)
        / league_goal_avg
    )


    return {
        "attack": recent_attack,
        "defense": recent_defense
    }


def combine_strength(
    long_value,
    recent_value
):

    return (
        LONG_TERM_WEIGHT * long_value
        + RECENT_WEIGHT * recent_value
    )


result_correct = 0
exact_score_correct = 0

home_goal_errors = []
away_goal_errors = []

predictions = []


for i in range(WARMUP_MATCHES, len(df)):

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


    home_attack = combine_strength(
        home_long["home_attack"],
        home_recent["attack"]
    )

    home_defense = combine_strength(
        home_long["home_defense"],
        home_recent["defense"]
    )

    away_attack = combine_strength(
        away_long["away_attack"],
        away_recent["attack"]
    )

    away_defense = combine_strength(
        away_long["away_defense"],
        away_recent["defense"]
    )


    league_home_avg = train["home_goals"].mean()
    league_away_avg = train["away_goals"].mean()


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


    home_probs = poisson.pmf(
        np.arange(MAX_GOALS + 1),
        home_lambda
    )

    away_probs = poisson.pmf(
        np.arange(MAX_GOALS + 1),
        away_lambda
    )


    score_matrix = np.outer(
        home_probs,
        away_probs
    )

    score_matrix /= score_matrix.sum()


    home_win_prob = np.tril(
        score_matrix,
        -1
    ).sum()

    draw_prob = np.trace(
        score_matrix
    )

    away_win_prob = np.triu(
        score_matrix,
        1
    ).sum()


    probabilities = [
        home_win_prob,
        draw_prob,
        away_win_prob
    ]


    predicted_result = np.argmax(
        probabilities
    )


    actual_home = int(
        match["home_goals"]
    )

    actual_away = int(
        match["away_goals"]
    )


    if actual_home > actual_away:
        actual_result = 0

    elif actual_home == actual_away:
        actual_result = 1

    else:
        actual_result = 2


    if predicted_result == actual_result:
        result_correct += 1


    predicted_score = np.unravel_index(
        np.argmax(score_matrix),
        score_matrix.shape
    )


    if (
        predicted_score[0] == actual_home
        and predicted_score[1] == actual_away
    ):
        exact_score_correct += 1


    home_goal_errors.append(
        abs(home_lambda - actual_home)
    )

    away_goal_errors.append(
        abs(away_lambda - actual_away)
    )


    predictions.append({
        "date": match["date"],

        "home_team": home_team,
        "away_team": away_team,

        "actual_home": actual_home,
        "actual_away": actual_away,

        "pred_home": predicted_score[0],
        "pred_away": predicted_score[1],

        "home_xg": home_lambda,
        "away_xg": away_lambda,

        "home_win_prob": home_win_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_win_prob
    })


n = len(predictions)


print("POISSON V2 - RECENT FORM")
print("-" * 35)

print("Backtest matches:", n)

print(
    "Result accuracy:",
    round(result_correct / n * 100, 2),
    "%"
)

print(
    "Exact score accuracy:",
    round(exact_score_correct / n * 100, 2),
    "%"
)

print(
    "Home goals MAE:",
    round(np.mean(home_goal_errors), 3)
)

print(
    "Away goals MAE:",
    round(np.mean(away_goal_errors), 3)
)


output = pd.DataFrame(predictions)

output.to_csv(
    BASE_DIR
    / "data"
    / "backtest_v2_predictions.csv",
    index=False
)

print(
    "\nSaved backtest_v2_predictions.csv"
)