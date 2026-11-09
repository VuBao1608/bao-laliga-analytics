"use client";

import { useEffect, useState } from "react";

const API_URL = "http://127.0.0.1:8000";

export default function Home() {
  const [teams, setTeams] = useState([]);

  const [homeTeam, setHomeTeam] = useState("");
  const [awayTeam, setAwayTeam] = useState("");

  const [prediction, setPrediction] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");


  // ------------------------------------
  // Load team list from FastAPI
  // ------------------------------------

  useEffect(() => {
    async function loadTeams() {
      try {
        const response = await fetch(
          `${API_URL}/teams`
        );

        if (!response.ok) {
          throw new Error(
            "Could not load teams."
          );
        }

        const data = await response.json();

        setTeams(data.teams);

        if (data.teams.length >= 2) {
          setHomeTeam(data.teams[0]);
          setAwayTeam(data.teams[1]);
        }
      } catch (err) {
        setError(
          "Could not connect to backend."
        );
      }
    }

    loadTeams();
  }, []);


  // ------------------------------------
  // Prediction
  // ------------------------------------

  async function handlePredict() {
    setError("");

    if (!homeTeam || !awayTeam) {
      setError(
        "Please select both teams."
      );

      return;
    }

    if (homeTeam === awayTeam) {
      setError(
        "Home and away teams must be different."
      );

      return;
    }

    setLoading(true);
    setPrediction(null);

    try {
      const params = new URLSearchParams({
        home_team: homeTeam,
        away_team: awayTeam,
      });

      const response = await fetch(
        `${API_URL}/predict?${params.toString()}`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Prediction failed."
        );
      }

      setPrediction(data);

    } catch (err) {
      setError(err.message);

    } finally {
      setLoading(false);
    }
  }


  return (
    <main className="min-h-screen bg-zinc-950 text-white">

      <div className="mx-auto max-w-6xl px-6 py-12">

        {/* Header */}

        <div className="mb-12">
          <p className="mb-2 text-sm font-semibold uppercase tracking-[0.25em] text-green-400">
            LaLiga Analytics
          </p>

          <h1 className="text-4xl font-bold md:text-6xl">
            Match Predictor
          </h1>

          <p className="mt-4 max-w-2xl text-zinc-400">
            Goals, match result, corners and
            yellow-card predictions powered by
            historical LaLiga data.
          </p>
        </div>


        {/* Team selector */}

        <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6">

          <div className="grid gap-6 md:grid-cols-[1fr_auto_1fr] md:items-end">

            {/* Home */}

            <div>
              <label className="mb-2 block text-sm text-zinc-400">
                Home Team
              </label>

              <select
                value={homeTeam}
                onChange={(e) =>
                  setHomeTeam(e.target.value)
                }
                className="w-full rounded-xl border border-zinc-700 bg-zinc-950 px-4 py-3 text-white"
              >
                {teams.map((team) => (
                  <option
                    key={team}
                    value={team}
                  >
                    {team}
                  </option>
                ))}
              </select>
            </div>


            <div className="hidden pb-3 text-xl font-bold text-zinc-500 md:block">
              VS
            </div>


            {/* Away */}

            <div>
              <label className="mb-2 block text-sm text-zinc-400">
                Away Team
              </label>

              <select
                value={awayTeam}
                onChange={(e) =>
                  setAwayTeam(e.target.value)
                }
                className="w-full rounded-xl border border-zinc-700 bg-zinc-950 px-4 py-3 text-white"
              >
                {teams.map((team) => (
                  <option
                    key={team}
                    value={team}
                  >
                    {team}
                  </option>
                ))}
              </select>
            </div>

          </div>


          <button
            onClick={handlePredict}
            disabled={loading}
            className="mt-6 w-full rounded-xl bg-green-500 px-6 py-3 font-bold text-black transition hover:bg-green-400 disabled:opacity-50"
          >
            {loading
              ? "Predicting..."
              : "Predict Match"}
          </button>


          {error && (
            <p className="mt-4 text-sm text-red-400">
              {error}
            </p>
          )}

        </section>


        {/* Prediction results */}

        {prediction && (

          <div className="mt-10">

            <div className="mb-8 text-center">

              <h2 className="text-3xl font-bold">
                {homeTeam}
                <span className="mx-4 text-zinc-600">
                  vs
                </span>
                {awayTeam}
              </h2>

            </div>


            {/* GOALS */}

            <section className="mb-6 rounded-2xl border border-zinc-800 bg-zinc-900 p-6">

              <h3 className="mb-6 text-xl font-bold">
                Goals
              </h3>


              <OutcomeCards
                homeTeam={homeTeam}
                awayTeam={awayTeam}
                goals={prediction.goals}
              />


              <div className="mt-6 grid gap-4 md:grid-cols-2">

                <StatCard
                  title="Expected Goals"
                  value={
                    `${prediction.goals.expected_home_goals} - ${prediction.goals.expected_away_goals}`
                  }
                />

                <StatCard
                  title="Most Likely Score"
                  value={
                    `${prediction.goals.most_likely_score.home} - ${prediction.goals.most_likely_score.away}`
                  }
                />

              </div>

              <TopScores scores={prediction.goals.top_scores} />

            </section>


            {/* 43 */}

            <section className="mb-6 rounded-2xl border border-zinc-800 bg-zinc-900 p-6">

              <h3 className="mb-6 text-xl font-bold">
                Corners
              </h3>

              <div className="grid gap-4 md:grid-cols-3">

                <StatCard
                  title={homeTeam}
                  value={
                    prediction.corners.expected_home
                  }
                />

                <StatCard
                  title="Expected Total"
                  value={
                    prediction.corners.expected_total
                  }
                />

                <StatCard
                  title={awayTeam}
                  value={
                    prediction.corners.expected_away
                  }
                />

              </div>

              <OverUnderTable
                data={
                  prediction.corners.over_under
                }
              />

              <div className="mt-4">
                <StatCard
                  title="Most Likely Total Corners"
                  value={prediction.corners.most_likely_total}
                />
              </div>

            </section>


            {/* CARDS */}

            <section className="rounded-2xl border border-zinc-800 bg-zinc-900 p-6">

              <h3 className="mb-6 text-xl font-bold">
                Yellow Cards
              </h3>

              <div className="grid gap-4 md:grid-cols-3">

                <StatCard
                  title={homeTeam}
                  value={
                    prediction.yellow_cards.expected_home
                  }
                />

                <StatCard
                  title="Expected Total"
                  value={
                    prediction.yellow_cards.expected_total
                  }
                />

                <StatCard
                  title={awayTeam}
                  value={
                    prediction.yellow_cards.expected_away
                  }
                />

              </div>

              <div className="mt-4">
                <StatCard
                  title="Most Likely Total Yellow Cards"
                  value={prediction.yellow_cards.most_likely_total}
                />
              </div>

              <OverUnderTable
                data={
                  prediction.yellow_cards.over_under
                }
              />

            </section>

          </div>

        )}

      </div>

    </main>
  );
}

function OutcomeCards({
  homeTeam,
  awayTeam,
  goals,
}) {
  const outcomes = [
    {
      key: "home",
      title: `${homeTeam} Win`,
      value: goals.home_win_probability,
    },
    {
      key: "draw",
      title: "Draw",
      value: goals.draw_probability,
    },
    {
      key: "away",
      title: `${awayTeam} Win`,
      value: goals.away_win_probability,
    },
  ];

  const highest = Math.max(
    ...outcomes.map((item) => item.value)
  );

  return (
    <div className="grid gap-4 md:grid-cols-3">
      {outcomes.map((item) => {
        const isHighest =
          item.value === highest;

        return (
          <div
            key={item.key}
            className={
              isHighest
                ? "rounded-xl border border-green-500 bg-green-500/10 p-5"
                : "rounded-xl border border-zinc-800 bg-zinc-950 p-5"
            }
          >
            <p className="text-sm text-zinc-500">
              {item.title}
            </p>

            <p className="mt-2 text-3xl font-bold">
              {item.value}%
            </p>

            {isHighest && (
              <p className="mt-2 text-xs font-semibold uppercase tracking-wider text-green-400">
                Most likely result
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}


function StatCard({ title, value }) {
  return (
    <div className="rounded-xl bg-zinc-950 p-5">

      <p className="text-sm text-zinc-500">
        {title}
      </p>

      <p className="mt-2 text-3xl font-bold">
        {value}
      </p>

    </div>
  );
}


function OverUnderTable({ data }) {
  return (
    <div className="mt-6 overflow-hidden rounded-xl border border-zinc-800">

      <div className="grid grid-cols-3 bg-zinc-950 px-4 py-3 text-sm text-zinc-500">

        <span>Line</span>
        <span>Over</span>
        <span>Under</span>

      </div>

      {Object.entries(data).map(
        ([line, probabilities]) => (

          <div
            key={line}
            className="grid grid-cols-3 border-t border-zinc-800 px-4 py-3"
          >
            <span>
              {line}
            </span>

            <span>
              {probabilities.over}%
            </span>

            <span>
              {probabilities.under}%
            </span>

          </div>

        )
      )}

    </div>
  );
}

function TopScores({ scores }) {
  if (!scores || scores.length === 0) {
    return null;
  }

  const maxProbability = Math.max(
    ...scores.map((score) => score.probability)
  );

  return (
    <div className="mt-6">
      <h4 className="mb-4 font-semibold">
        Top Scorelines
      </h4>

      <div className="space-y-3">
        {scores.map((score, index) => {
          const width =
            (score.probability / maxProbability) * 100;

          return (
            <div
              key={`${score.home_goals}-${score.away_goals}-${index}`}
              className="rounded-xl bg-zinc-950 p-4"
            >
              <div className="mb-2 flex items-center justify-between">
                <span className="font-semibold">
                  {score.home_goals} - {score.away_goals}
                </span>

                <span className="text-sm text-zinc-400">
                  {score.probability.toFixed(1)}%
                </span>
              </div>

              <div className="h-2 overflow-hidden rounded-full bg-zinc-800">
                <div
                  className="h-full rounded-full bg-green-500"
                  style={{
                    width: `${width}%`,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}