
import pandas as pd

df = pd.read_parquet(
    "tests/reports/top5_predictions_unique.parquet"
)

for threshold in [
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]:

    bets = df[
        df["top5_probability"]
        > threshold
    ]

    hit = bets["target"].mean()

    print()

    print(
        f"THRESHOLD {threshold}"
    )

    print(
        "BETS:",
        len(bets)
    )

    print(
        "HIT:",
        round(
            hit * 100,
            2
        ),
        "%"
    )
