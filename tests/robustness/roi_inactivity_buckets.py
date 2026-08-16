
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

bets = df[
    df["edge"] > 0.05
].copy()

bets["profit"] = bets.apply(
    lambda r:
    r["odds_a"] - 1
    if r["target"] == 1
    else -1,
    axis=1
)

bins = [
    -1000,
    -30,
    -14,
    -7,
    0,
    30,
    1000
]

labels = [
    "< -30",
    "-30 to -14",
    "-14 to -7",
    "-7 to 0",
    "0 to 30",
    "> 30"
]

bets["bucket"] = pd.cut(
    bets["delta_inactivity_days"],
    bins=bins,
    labels=labels
)

for bucket, tmp in bets.groupby("bucket"):

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
    print("BETS:", len(tmp))
    print("HIT :", round(hit*100,2))
    print("ROI :", round(roi*100,2))
