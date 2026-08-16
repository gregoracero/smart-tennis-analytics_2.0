
import pandas as pd

df = pd.read_parquet(
    "tests/reports/top5_temporal_predictions.parquet"
)

print()
print("=" * 80)
print("TOP5 ROI BY EDGE")
print("=" * 80)

for threshold in [
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30
]:

    bets = df[
        df["top5_edge"] > threshold
    ].copy()

    if len(bets) == 0:
        continue

    bets["profit"] = bets.apply(
        lambda r:
        r["odds_a"] - 1
        if r["target"] == 1
        else -1,
        axis=1
    )

    roi = bets["profit"].mean()

    print()

    print(
        f"EDGE > {threshold:.2f}"
    )

    print(
        "BETS:",
        len(bets)
    )

    print(
        "HIT RATE:",
        round(
            bets["target"].mean() * 100,
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
