
import pandas as pd

df = pd.read_parquet(
    "tests/reports/top5_predictions_unique.parquet"
)

print()
print("=" * 80)
print("TOP5 ROI REPORT")
print("=" * 80)

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
        df["top5_probability"] > threshold
    ].copy()

    wins = int(
        bets["target"].sum()
    )

    losses = (
        len(bets)
        -
        wins
    )

    hit_rate = (
        bets["target"]
        .mean()
    )

    roi = (
        wins - losses
    ) / len(bets)

    print()
    print("-" * 80)

    print(
        f"THRESHOLD {threshold}"
    )

    print(
        "BETS:",
        len(bets)
    )

    print(
        "WINS:",
        wins
    )

    print(
        "LOSSES:",
        losses
    )

    print(
        "HIT RATE:",
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
