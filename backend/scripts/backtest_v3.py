from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "laliga_2025.csv"


WARMUP_MATCHES = 100
RECENT_MATCHES = 5
MAX_GOALS = 10

RECENT_WEIGHT = 0.30
LONG_TERM_WEIGHT = 0.70

INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 60

# How strongly Elo changes expected goals
ELO_WEIGHT = 0.00

df = pd.read_csv(DATA_FILE)

df = df[
    df["home_goals"].notna()
    & df["away_goals"].notna()
].copy()

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    "date"
).reset_index(drop=True)


teams = sorted(
    set(df["home_team"])
    | set(df["away_team"])
)

ratings = {
    team: INITIAL_ELO
    for team in teams
}


def expected_elo_score(rating_a, rating_b):

    return 1 / (
        1
        + 10 ** (
            (rating_b - rating_a) / 400
        )
    )


def update_elo(match):

    home_team = match["home_team"]
    away_team = match["away_team"]

    home_rating = ratings[home_team]
    away_rating = ratings[away_team]

    expected_home = expected_elo_score(
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
        + K_FACTOR
        * (actual_home - expected_home)
    )

    ratings[away_team] = (
        away_rating
        + K_FACTOR
        * (actual_away - expected_away)
    )


def get_long_strength(train, team):

    league_home_avg = (
        train["home_goals"].mean()
    )

    league_away_avg = (
        train["away_goals"].mean()
    )

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
        train["home_goals"].sum()
        + train["away_goals"].sum()
    ) / (2 * len(train))


    return {
        "attack":
            np.mean(scored)
            / league_avg,

        "defense":
            np.mean(conceded)
            / league_avg
    }


# Build Elo only from warm-up games
for i in range(WARMUP_MATCHES):
    update_elo(df.iloc[i])


result_correct = 0
exact_correct = 0

home_errors = []
away_errors = []

predictions = []


for i in range(
    WARMUP_MATCHES,
    len(df)
):

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
        update_elo(match)
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


    league_home_avg = (
        train["home_goals"].mean()
    )

    league_away_avg = (
        train["away_goals"].mean()
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


    # -------------------------
    # ELO ADJUSTMENT
    # -------------------------

    home_elo = ratings[home_team]
    away_elo = ratings[away_team]

    elo_diff = (
        home_elo
        + HOME_ADVANTAGE
        - away_elo
    )


    home_elo_factor = (
        10 ** (
            ELO_WEIGHT
            * elo_diff
            / 400
        )
    )

    away_elo_factor = (
        10 ** (
            -ELO_WEIGHT
            * elo_diff
            / 400
        )
    )


    home_lambda *= home_elo_factor
    away_lambda *= away_elo_factor


    # -------------------------
    # POISSON
    # -------------------------

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


    home_win_prob = np.tril(
        matrix,
        -1
    ).sum()

    draw_prob = np.trace(
        matrix
    )

    away_win_prob = np.triu(
        matrix,
        1
    ).sum()


    result_probs = [
        home_win_prob,
        draw_prob,
        away_win_prob
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
        result_correct += 1


    predicted_score = np.unravel_index(
        np.argmax(matrix),
        matrix.shape
    )


    if (
        predicted_score[0] == actual_home
        and predicted_score[1] == actual_away
    ):
        exact_correct += 1


    home_errors.append(
        abs(
            home_lambda
            - actual_home
        )
    )

    away_errors.append(
        abs(
            away_lambda
            - actual_away
        )
    )


    predictions.append({
        "date": match["date"],

        "home_team": home_team,
        "away_team": away_team,

        "home_elo": home_elo,
        "away_elo": away_elo,

        "home_xg": home_lambda,
        "away_xg": away_lambda,

        "actual_home": actual_home,
        "actual_away": actual_away,

        "pred_home": predicted_score[0],
        "pred_away": predicted_score[1],

        "home_win_prob": home_win_prob,
        "draw_prob": draw_prob,
        "away_win_prob": away_win_prob
    })


    # IMPORTANT:
    # update Elo only AFTER prediction
    update_elo(match)


n = len(predictions)


print(
    "POISSON V3 - FORM + ELO"
)

print("-" * 40)

print(
    "Backtest matches:",
    n
)

print(
    "Result accuracy:",
    round(
        result_correct / n * 100,
        2
    ),
    "%"
)

print(
    "Exact score accuracy:",
    round(
        exact_correct / n * 100,
        2
    ),
    "%"
)

print(
    "Home goals MAE:",
    round(
        np.mean(home_errors),
        3
    )
)

print(
    "Away goals MAE:",
    round(
        np.mean(away_errors),
        3
    )
)


output = pd.DataFrame(
    predictions
)

output.to_csv(
    BASE_DIR
    / "data"
    / "backtest_v3_predictions.csv",
    index=False
)


print(
    "\nSaved backtest_v3_predictions.csv"
)