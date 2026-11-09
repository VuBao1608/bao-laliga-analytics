from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

FIXTURES_FILE = BASE_DIR / "data" / "laliga_2025.csv"
STATS_FILE = BASE_DIR / "data" / "laliga_2025_statistics.csv"

OUTPUT_FILE = BASE_DIR / "data" / "laliga_2025_full.csv"


fixtures = pd.read_csv(FIXTURES_FILE)
stats = pd.read_csv(STATS_FILE)


print("Fixtures:", len(fixtures))
print("Statistics:", len(stats))


# Avoid duplicate team-name columns from the statistics file
stats = stats.drop(
    columns=["home_team", "away_team"],
    errors="ignore"
)


full = fixtures.merge(
    stats,
    on="fixture_id",
    how="left"
)


full = full.sort_values(
    "date"
).reset_index(drop=True)


print("Merged matches:", len(full))

print(
    "Matches missing statistics:",
    full["home_shots"].isna().sum()
)


duplicate_ids = full[
    "fixture_id"
].duplicated().sum()

print(
    "Duplicate fixture IDs:",
    duplicate_ids
)


full.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("Columns:")
print(full.columns.tolist())