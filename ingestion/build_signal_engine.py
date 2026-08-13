import pandas as pd
import numpy as np

INPUT = "data/parquet/rolling_metrics.parquet"
OUTPUT = "data/parquet/signal_engine.parquet"

df = pd.read_parquet(INPUT)

# Surface Form
df["surface_form"] = np.where(
    df["win_pct"] >= 75,
    "HIGH",
    np.where(
        df["win_pct"] >= 55,
        "MEDIUM",
        "LOW"
    )
)

# Fatigue
df["fatigue_risk"] = np.where(
    df["avg_minutes"] >= 150,
    "HIGH",
    np.where(
        df["avg_minutes"] >= 110,
        "MEDIUM",
        "LOW"
    )
)

# Break Risk
df["break_risk"] = np.where(
    df["second_serve_won"] < 45,
    "HIGH",
    np.where(
        df["second_serve_won"] < 52,
        "MEDIUM",
        "LOW"
    )
)

# Tie Break Probability
df["tiebreak_probability"] = np.where(
    df["first_serve_won"] >= 70,
    "HIGH",
    np.where(
        df["first_serve_won"] >= 65,
        "MEDIUM",
        "LOW"
    )
)

score = []

for _, row in df.iterrows():

    s = 0

    s += row["win_pct"] * 0.5

    if pd.notna(row["first_serve_won"]):
        s += row["first_serve_won"] * 0.3

    if pd.notna(row["second_serve_won"]):
        s += row["second_serve_won"] * 0.2

    score.append(round(s, 2))

df["confidence_score"] = score

df.to_parquet(
    OUTPUT,
    index=False
)

print()
print("DONE")
print(df.shape)
print(f"Saved: {OUTPUT}")
