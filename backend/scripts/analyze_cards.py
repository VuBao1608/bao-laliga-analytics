from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "laliga_2025_full.csv"
)


df = pd.read_csv(DATA_FILE)

df = df[
    df["home_yellow_cards"].notna()
    & df["away_yellow_cards"].notna()
].copy()


df["total_yellow_cards"] = (
    df["home_yellow_cards"]
    + df["away_yellow_cards"]
)


home_mean = df["home_yellow_cards"].mean()
away_mean = df["away_yellow_cards"].mean()
total_mean = df["total_yellow_cards"].mean()

home_var = df["home_yellow_cards"].var()
away_var = df["away_yellow_cards"].var()
total_var = df["total_yellow_cards"].var()


print("YELLOW CARD ANALYSIS")
print("-" * 40)

print("Matches:", len(df))

print()

print(
    "Average home yellow cards:",
    round(home_mean, 3)
)

print(
    "Average away yellow cards:",
    round(away_mean, 3)
)

print(
    "Average total yellow cards:",
    round(total_mean, 3)
)

print()

print(
    "Home card variance:",
    round(home_var, 3)
)

print(
    "Away card variance:",
    round(away_var, 3)
)

print(
    "Total card variance:",
    round(total_var, 3)
)

print()

print(
    "Home variance / mean:",
    round(home_var / home_mean, 3)
)

print(
    "Away variance / mean:",
    round(away_var / away_mean, 3)
)

print(
    "Total variance / mean:",
    round(total_var / total_mean, 3)
)

print()


home_more = (
    df["home_yellow_cards"]
    > df["away_yellow_cards"]
).mean() * 100

same = (
    df["home_yellow_cards"]
    == df["away_yellow_cards"]
).mean() * 100

away_more = (
    df["home_yellow_cards"]
    < df["away_yellow_cards"]
).mean() * 100


print(
    "Home team had more cards:",
    round(home_more, 1),
    "%"
)

print(
    "Same cards:",
    round(same, 1),
    "%"
)

print(
    "Away team had more cards:",
    round(away_more, 1),
    "%"
)