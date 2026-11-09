import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

FIXTURES_FILE = (
    BASE_DIR
    / "data"
    / "laliga_2025.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "laliga_2025_statistics.csv"
)


load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("API_FOOTBALL_KEY")

URL = (
    "https://v3.football.api-sports.io/"
    "fixtures/statistics"
)

HEADERS = {
    "x-apisports-key": API_KEY
}


def clean_value(value):

    if value is None:
        return 0

    if (
        isinstance(value, str)
        and "%" in value
    ):
        return float(
            value.replace("%", "")
        )

    return value


def stats_to_dict(stats_list):

    result = {}

    for item in stats_list:

        result[item["type"]] = (
            clean_value(
                item["value"]
            )
        )

    return result


fixtures = pd.read_csv(
    FIXTURES_FILE
)


# --------------------------------
# Find already downloaded matches
# --------------------------------

if OUTPUT_FILE.exists():

    existing = pd.read_csv(
        OUTPUT_FILE
    )

    completed_ids = set(
        existing["fixture_id"]
        .astype(int)
        .tolist()
    )

    rows = existing.to_dict(
        "records"
    )

else:

    completed_ids = set()
    rows = []


print(
    "Already downloaded:",
    len(completed_ids)
)

print(
    "Total fixtures:",
    len(fixtures)
)

print()


# --------------------------------
# Download missing fixtures
# --------------------------------

for index, match in fixtures.iterrows():

    fixture_id = int(
        match["fixture_id"]
    )


    # Skip already downloaded
    if fixture_id in completed_ids:
        continue


    print(
        f"Fetching "
        f"{len(completed_ids) + 1}/"
        f"{len(fixtures)}"
        f" - "
        f"{match['home_team']} "
        f"vs "
        f"{match['away_team']}"
    )


    try:

        response = requests.get(
            URL,
            headers=HEADERS,
            params={
                "fixture": fixture_id
            },
            timeout=20
        )

        data = response.json()


    except Exception as error:

        print(
            "Request failed:",
            error
        )

        print(
            "Progress saved. "
            "Run the script again "
            "to continue."
        )

        break


    if data.get("errors"):

        print(
            "API error:",
            data["errors"]
        )

        continue


    teams_stats = data.get(
        "response",
        []
    )


    if len(teams_stats) != 2:

        print(
            "Statistics unavailable:",
            fixture_id
        )

        continue


    home_data = teams_stats[0]
    away_data = teams_stats[1]


    home_stats = stats_to_dict(
        home_data["statistics"]
    )

    away_stats = stats_to_dict(
        away_data["statistics"]
    )


    row = {

        "fixture_id": fixture_id,

        "home_team":
            match["home_team"],

        "away_team":
            match["away_team"],


        "home_shots":
            home_stats.get(
                "Total Shots",
                0
            ),

        "away_shots":
            away_stats.get(
                "Total Shots",
                0
            ),


        "home_shots_on_target":
            home_stats.get(
                "Shots on Goal",
                0
            ),

        "away_shots_on_target":
            away_stats.get(
                "Shots on Goal",
                0
            ),


        "home_corners":
            home_stats.get(
                "Corner Kicks",
                0
            ),

        "away_corners":
            away_stats.get(
                "Corner Kicks",
                0
            ),


        "home_yellow_cards":
            home_stats.get(
                "Yellow Cards",
                0
            ),

        "away_yellow_cards":
            away_stats.get(
                "Yellow Cards",
                0
            ),


        "home_red_cards":
            home_stats.get(
                "Red Cards",
                0
            ),

        "away_red_cards":
            away_stats.get(
                "Red Cards",
                0
            ),


        "home_possession":
            home_stats.get(
                "Ball Possession",
                0
            ),

        "away_possession":
            away_stats.get(
                "Ball Possession",
                0
            ),


        "home_fouls":
            home_stats.get(
                "Fouls",
                0
            ),

        "away_fouls":
            away_stats.get(
                "Fouls",
                0
            ),


        "home_offsides":
            home_stats.get(
                "Offsides",
                0
            ),

        "away_offsides":
            away_stats.get(
                "Offsides",
                0
            )
    }


    rows.append(row)

    completed_ids.add(
        fixture_id
    )


    # Save after EVERY fixture
    pd.DataFrame(
        rows
    ).to_csv(
        OUTPUT_FILE,
        index=False
    )


    time.sleep(0.15)


print()
print(
    "Downloaded statistics:",
    len(completed_ids)
)

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)