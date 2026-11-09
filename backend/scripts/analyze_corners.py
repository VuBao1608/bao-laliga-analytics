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
    df["home_corners"].notna()
    & df["away_corners"].notna()
].copy()


df["total_corners"] = (
    df["home_corners"]
    + df["away_corners"]
)


home_mean = df["home_corners"].mean()
away_mean = df["away_corners"].mean()
total_mean = df["total_corners"].mean()

home_var = df["home_corners"].var()
away_var = df["away_corners"].var()
total_var = df["total_corners"].var()


print("CORNER ANALYSIS")
print("-" * 40)

print("Matches:", len(df))

print()

print(
    "Average home corners:",
    round(home_mean, 3)
)

print(
    "Average away corners:",
    round(away_mean, 3)
)

print(
    "Average total corners:",
    round(total_mean, 3)
)

print()

print(
    "Home corner variance:",
    round(home_var, 3)
)

print(
    "Away corner variance:",
    round(away_var, 3)
)

print(
    "Total corner variance:",
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
    df["home_corners"]
    > df["away_corners"]
).mean() * 100

same = (
    df["home_corners"]
    == df["away_corners"]
).mean() * 100

away_more = (
    df["home_corners"]
    < df["away_corners"]
).mean() * 100


print(
    "Home team had more corners:",
    round(home_more, 1),
    "%"
)

print(
    "Same corners:",
    round(same, 1),
    "%"
)

print(
    "Away team had more corners:",
    round(away_more, 1),
    "%"
)