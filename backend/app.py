from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException

from prediction_engine import (
    TEAMS,
    predict_full_match
)


app = FastAPI(
    title="LaLiga Predictor API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():

    return {
        "message":
            "LaLiga Predictor API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.get("/teams")
def get_teams():

    return {
        "count": len(TEAMS),
        "teams": TEAMS
    }


@app.get("/predict")
def predict(
    home_team: str,
    away_team: str
):

    try:

        return predict_full_match(
            home_team,
            away_team
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )