
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

bets = df[
    (df["edge"] > 0.05)
    &
    (df["market_probability_a"] < 0.50)
].copy()

bets["profit"] = bets.apply(
    lambda r:
    r["odds_a"] - 1
    if r["target"] == 1
    else -1,
    axis=1
)

bins = [
    1.5,
    2.0,
    3.0,
    5.0,
    100
]

labels = [
    "1.5-2.0",
    "2.0-3.0",
    "3.0-5.0",
    "5.0+"
]

bets["bucket"] = pd.cut(
    bets["odds_a"],
    bins=bins,
    labels=labels
)

print()
print("=" * 80)
print("UNDERDOG BREAKDOWN")
print("=" * 80)

for bucket, tmp in bets.groupby(
    "bucket"
):

    roi = (
        tmp["profit"].sum()
        /
        len(tmp)
    )

    hit = (
        tmp["target"].mean()
    )

    print()
    print(bucket)
    print("-" * 40)
    print("BETS :", len(tmp))
    print("HIT  :", round(hit*100,2), "%")
    print("ROI  :", round(roi*100,2), "%")
