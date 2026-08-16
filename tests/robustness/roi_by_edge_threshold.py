
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

thresholds = [
    0.02,
    0.05,
    0.10,
    0.15,
    0.20
]

print()
print("=" * 80)
print("ROI BY EDGE THRESHOLD")
print("=" * 80)

for edge in thresholds:

    bets = df[
        df["edge"] > edge
    ].copy()

    bets["profit"] = bets.apply(
        lambda r:
        r["odds_a"] - 1
        if r["target"] == 1
        else -1,
        axis=1
    )

    roi = (
        bets["profit"].sum()
        / len(bets)
    )

    hit = (
        bets["target"].mean()
    )

    print()
    print("EDGE >", edge)
    print("-" * 40)
    print("BETS :", len(bets))
    print("HIT  :", round(hit*100,2), "%")
    print("ROI  :", round(roi*100,2), "%")
