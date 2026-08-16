
import pandas as pd

df = pd.read_parquet(
    "tests/reports/top5_predictions_unique.parquet"
)

bets = df[
    df["top5_probability"] >= 0.90
]

print()

print("=" * 80)
print("TOP5 EXTREME CONFIDENCE")
print("=" * 80)

print()

print(
    "BETS:",
    len(bets)
)

print()

print(
    "HIT RATE:",
    round(
        bets["target"].mean() * 100,
        2
    ),
    "%"
)

print()

print(
    bets[
        [
            "delta_surface_inactivity_days",
            "delta_elo",
            "delta_inactivity_days"
        ]
    ]
    .describe()
)
