
import pandas as pd

from production.services.model_loader import (
    load_model
)

assets = load_model(
    tour="atp",
    surface="hard",
    engine="xgboost",
    version="v1"
)

df = pd.read_parquet(
    "data/parquet/training_matches_with_elo.parquet"
)

features = assets["features"]

for col in features:

    if col not in df.columns:

        df[col] = None

X = df[features]

X = assets["imputer"].transform(X)

probabilities = (
    assets["model"]
    .predict_proba(X)[:, 1]
)

results = pd.DataFrame({

    "probability":
        probabilities,

    "target":
        df["target"]

})

# Cuota impl?cita del modelo
results["fair_odds"] = (
    1 / results["probability"]
)

print()
print("=" * 90)
print("ROI SIMULATION BY THRESHOLD")
print("=" * 90)

thresholds = [

    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95

]

rows = []

for threshold in thresholds:

    subset = results[
        results["probability"]
        >= threshold
    ]

    matches = len(subset)

    if matches == 0:

        continue

    wins = subset["target"].sum()

    losses = matches - wins

    strike_rate = wins / matches

    avg_probability = (
        subset["probability"]
        .mean()
    )

    avg_fair_odds = (
        subset["fair_odds"]
        .mean()
    )

    rows.append({

        "threshold":
            threshold,

        "matches":
            matches,

        "wins":
            int(wins),

        "losses":
            int(losses),

        "strike_rate":
            round(
                strike_rate,
                4
            ),

        "avg_probability":
            round(
                avg_probability,
                4
            ),

        "avg_fair_odds":
            round(
                avg_fair_odds,
                3
            )
    })

summary = pd.DataFrame(
    rows
)

print()
print(summary)

print()
print("=" * 90)
print("BEST THRESHOLDS")
print("=" * 90)

print()

print(
    summary.sort_values(
        "strike_rate",
        ascending=False
    )
)
