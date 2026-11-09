from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "laliga_2025.csv"

WARMUP_MATCHES = 100
RECENT_MATCHES = 5
MAX_GOALS = 10

WEIGHTS = [
    0.00,
    0.10,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50
]


df = pd.read_csv(DATA_FILE)

df = df[
    df["home_goals"].notna()
    & df["away_goals"].notna()
].copy()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


def get_long_strength(train, team):

    league_home_avg = train["home_goals"].mean()
    league_away_avg = train["away_goals"].mean()

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


def get_recent_strength(train, team):

    matches = train[
        (train["home_team"] == team)
        | (train["away_team"] == team)
    ].tail(RECENT_MATCHES)

    if len(matches) == 0:
        return None

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
        train["home_goals"].sum()
        + train["away_goals"].sum()
    ) / (2 * len(train))

    return {
        "attack": np.mean(scored) / league_avg,
        "defense": np.mean(conceded) / league_avg
    }


def test_weight(recent_weight):

    long_weight = 1 - recent_weight

    correct = 0
    exact = 0

    home_errors = []
    away_errors = []

    total_predictions = 0


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
            long_weight
            * home_long["home_attack"]
            + recent_weight
            * home_recent["attack"]
        )

        home_defense = (
            long_weight
            * home_long["home_defense"]
            + recent_weight
            * home_recent["defense"]
        )

        away_attack = (
            long_weight
            * away_long["away_attack"]
            + recent_weight
            * away_recent["attack"]
        )

        away_defense = (
            long_weight
            * away_long["away_defense"]
            + recent_weight
            * away_recent["defense"]
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


        matrix = np.outer(
            home_probs,
            away_probs
        )

        matrix /= matrix.sum()


        result_probs = [
            np.tril(matrix, -1).sum(),
            np.trace(matrix),
            np.triu(matrix, 1).sum()
        ]

        predicted_result = np.argmax(
            result_probs
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
            correct += 1


        predicted_score = np.unravel_index(
            np.argmax(matrix),
            matrix.shape
        )


        if (
            predicted_score[0] == actual_home
            and predicted_score[1] == actual_away
        ):
            exact += 1


        home_errors.append(
            abs(home_lambda - actual_home)
        )

        away_errors.append(
            abs(away_lambda - actual_away)
        )

        total_predictions += 1


    return {
        "recent_weight": recent_weight,
        "result_accuracy":
            correct / total_predictions * 100,
        "exact_accuracy":
            exact / total_predictions * 100,
        "home_mae":
            np.mean(home_errors),
        "away_mae":
            np.mean(away_errors)
    }


results = []

for weight in WEIGHTS:

    result = test_weight(weight)

    results.append(result)

    print(
        f"Recent weight {weight:.2f} | "
        f"Result {result['result_accuracy']:.2f}% | "
        f"Exact {result['exact_accuracy']:.2f}% | "
        f"Home MAE {result['home_mae']:.3f} | "
        f"Away MAE {result['away_mae']:.3f}"
    )


results_df = pd.DataFrame(results)

results_df.to_csv(
    BASE_DIR / "data" / "recent_weight_results.csv",
    index=False
)
