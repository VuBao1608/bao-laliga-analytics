from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "laliga_2025_full.csv"

RECENT_MATCHES = 5
LONG_TERM_WEIGHT = 0.80
RECENT_WEIGHT = 0.20


# Temporary test match
HOME_TEAM = "Atletico Madrid"
AWAY_TEAM = "Villarreal"


df = pd.read_csv(DATA_FILE)

df = df[
    df["home_corners"].notna()
    & df["away_corners"].notna()
].copy()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


def get_long_strength(team):

    league_home_avg = df["home_corners"].mean()
    league_away_avg = df["away_corners"].mean()

    home = df[df["home_team"] == team]
    away = df[df["away_team"] == team]

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


def get_recent_strength(team):

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


home_long = get_long_strength(HOME_TEAM)
away_long = get_long_strength(AWAY_TEAM)

home_recent = get_recent_strength(HOME_TEAM)
away_recent = get_recent_strength(AWAY_TEAM)


home_attack = (
    LONG_TERM_WEIGHT * home_long["home_attack"]
    + RECENT_WEIGHT * home_recent["attack"]
)

home_defense = (
    LONG_TERM_WEIGHT * home_long["home_defense"]
    + RECENT_WEIGHT * home_recent["defense"]
)

away_attack = (
    LONG_TERM_WEIGHT * away_long["away_attack"]
    + RECENT_WEIGHT * away_recent["attack"]
)

away_defense = (
    LONG_TERM_WEIGHT * away_long["away_defense"]
    + RECENT_WEIGHT * away_recent["defense"]
)


league_home_avg = df["home_corners"].mean()
league_away_avg = df["away_corners"].mean()


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

expected_total = expected_home + expected_away


# ------------------------------------
# Negative Binomial dispersion
# ------------------------------------

total_corners = (
    df["home_corners"]
    + df["away_corners"]
)

mean_total = total_corners.mean()
var_total = total_corners.var()


# NB2:
# variance = mean + alpha * mean^2

alpha = (
    (var_total - mean_total)
    / (mean_total ** 2)
)

# Safety
alpha = max(alpha, 0.0001)


# scipy Negative Binomial parameters
n = 1 / alpha

p = n / (n + expected_total)


def over_probability(line):

    # Example:
    # Over 9.5 means total >= 10

    cutoff = int(np.floor(line))

    return 1 - nbinom.cdf(
        cutoff,
        n,
        p
    )

def under_probability(line):
    cutoff = int(np.floor(line))

    return nbinom.cdf(
        cutoff,
        n,
        p
    )

# ------------------------------------
# Most likely total corners
# ------------------------------------

max_total_corners = 25

corner_values = np.arange(
    0,
    max_total_corners + 1
)

corner_probabilities = nbinom.pmf(
    corner_values,
    n,
    p
)

most_likely_total = corner_values[
    np.argmax(corner_probabilities)
]


print(
    "\nMost likely total corners:",
    most_likely_total
)


top_indices = np.argsort(
    corner_probabilities
)[::-1][:5]


print("\nTop 5 total corner predictions:")

for index in top_indices:

    corners = corner_values[index]

    probability = corner_probabilities[index]

    print(
        f"{corners} corners: "
        f"{probability * 100:.1f}%"
    )

print()
print(
    f"{HOME_TEAM} vs {AWAY_TEAM}"
)

print("-" * 45)

print("\nExpected corners:")

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


print("\nTotal corner probabilities:")

for line in [8.5, 9.5, 10.5, 11.5]:

    over = over_probability(line)
    under = under_probability(line)

    print(
        f"Line {line}: "
        f"Over {over * 100:.1f}% | "
        f"Under {under * 100:.1f}%"
    )


print(
    "\nNegative Binomial alpha:",
    round(alpha, 4)
)