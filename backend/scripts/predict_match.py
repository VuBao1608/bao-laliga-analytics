from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "laliga_2025.csv"


HOME_TEAM = "Barcelona"
AWAY_TEAM = "Real Madrid"


df = pd.read_csv(DATA_FILE)

completed = df[
    df["home_goals"].notna()
    & df["away_goals"].notna()
].copy()


# League averages
league_home_avg = completed["home_goals"].mean()
league_away_avg = completed["away_goals"].mean()


def get_team_strength(team):
    home_matches = completed[
        completed["home_team"] == team
    ]

    away_matches = completed[
        completed["away_team"] == team
    ]

    home_goals_scored = home_matches["home_goals"].mean()
    home_goals_conceded = home_matches["away_goals"].mean()

    away_goals_scored = away_matches["away_goals"].mean()
    away_goals_conceded = away_matches["home_goals"].mean()

    return {
        "home_attack": home_goals_scored / league_home_avg,
        "home_defense": home_goals_conceded / league_away_avg,
        "away_attack": away_goals_scored / league_away_avg,
        "away_defense": away_goals_conceded / league_home_avg,
    }


home_strength = get_team_strength(HOME_TEAM)
away_strength = get_team_strength(AWAY_TEAM)


# Expected goals
home_lambda = (
    league_home_avg
    * home_strength["home_attack"]
    * away_strength["away_defense"]
)

away_lambda = (
    league_away_avg
    * away_strength["away_attack"]
    * home_strength["home_defense"]
)


print(f"\n{HOME_TEAM} vs {AWAY_TEAM}")
print("-" * 40)

print("\nExpected goals:")
print(f"{HOME_TEAM}: {home_lambda:.2f}")
print(f"{AWAY_TEAM}: {away_lambda:.2f}")


# Probability of 0, 1, 2, ... 10 goals
max_goals = 10

home_probs = poisson.pmf(
    np.arange(max_goals + 1),
    home_lambda
)

away_probs = poisson.pmf(
    np.arange(max_goals + 1),
    away_lambda
)


# Score probability matrix
score_matrix = np.outer(
    home_probs,
    away_probs
)

# Normalize because we stop at 10 goals
score_matrix = score_matrix / score_matrix.sum()


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


print("\nMatch result probabilities:")

print(
    f"{HOME_TEAM} win: "
    f"{home_win_prob * 100:.1f}%"
)

print(
    f"Draw: "
    f"{draw_prob * 100:.1f}%"
)

print(
    f"{AWAY_TEAM} win: "
    f"{away_win_prob * 100:.1f}%"
)


# Most likely exact score
best_score = np.unravel_index(
    np.argmax(score_matrix),
    score_matrix.shape
)

print("\nMost likely score:")

print(
    f"{HOME_TEAM} "
    f"{best_score[0]} - {best_score[1]} "
    f"{AWAY_TEAM}"
)


# Top 5 exact scores
scores = []

for home_goals in range(max_goals + 1):
    for away_goals in range(max_goals + 1):

        probability = score_matrix[
            home_goals,
            away_goals
        ]

        scores.append(
            (
                probability,
                home_goals,
                away_goals
            )
        )


scores.sort(reverse=True)


print("\nTop 5 score predictions:")

for probability, home_goals, away_goals in scores[:5]:

    print(
        f"{home_goals}-{away_goals}: "
        f"{probability * 100:.1f}%"
    )