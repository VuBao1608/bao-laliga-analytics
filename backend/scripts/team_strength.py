from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

data_file = BASE_DIR / "data" / "laliga_2025.csv"

df = pd.read_csv(data_file)

completed = df[
    df["home_goals"].notna() &
    df["away_goals"].notna()
].copy()


league_home_avg = completed["home_goals"].mean()
league_away_avg = completed["away_goals"].mean()


teams = sorted(
    set(completed["home_team"]) |
    set(completed["away_team"])
)


rows = []


for team in teams:

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


    home_attack_strength = (
        home_goals_scored / league_home_avg
    )

    home_defense_strength = (
        home_goals_conceded / league_away_avg
    )

    away_attack_strength = (
        away_goals_scored / league_away_avg
    )

    away_defense_strength = (
        away_goals_conceded / league_home_avg
    )


    rows.append({
        "team": team,

        "home_attack": home_attack_strength,
        "home_defense": home_defense_strength,

        "away_attack": away_attack_strength,
        "away_defense": away_defense_strength
    })


strength_df = pd.DataFrame(rows)

strength_df = strength_df.sort_values(
    "home_attack",
    ascending=False
)


print(strength_df.round(3).to_string(index=False))