
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

thresholds = [
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40
]

print()
print("=" * 100)
print("EXTREME EDGE TEST")
print("=" * 100)

for edge in thresholds:

    bets = df[
        df["edge"] > edge
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

    roi = (
        bets["profit"].sum()
        /
        len(bets)
    )

    hit = (
        bets["target"].mean()
    )

    print()

    print(
        f"EDGE > {edge:.2f}"
    )

    print(
        f"BETS = {len(bets)}"
    )

    print(
        f"HIT = {hit:.2%}"
    )

    print(
        f"ROI = {roi:.2%}"
    )
