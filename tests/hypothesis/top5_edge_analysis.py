
import pandas as pd

df = pd.read_parquet(
    "tests/reports/top5_predictions_unique.parquet"
)

df["edge"] = (
    df["top5_probability"]
    - 0.50
)

print()

print("=" * 80)
print("TOP5 EDGE ANALYSIS")
print("=" * 80)

for threshold in [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40
]:

    bets = df[
        df["edge"] > threshold
    ]

    hit_rate = (
        bets["target"]
        .mean()
    )

    roi = (
        (
            bets["target"] == 1
        ).sum()
        -
        (
            bets["target"] == 0
        ).sum()
    ) / len(bets)

    print()

    print(
        f"EDGE > {threshold:.2f}"
    )

    print(
        "BETS:",
        len(bets)
    )

    print(
        "HIT:",
        round(
            hit_rate * 100,
            2
        ),
        "%"
    )

    print(
        "ROI:",
        round(
            roi * 100,
            2
        ),
        "%"
    )
