import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

EDGE = 0.05

bets = df[
    df["edge"] > EDGE
].copy()

bets["profit"] = bets.apply(
    lambda r:
    r["odds_a"] - 1
    if r["target"] == 1
    else -1,
    axis=1
)

print()
print("=" * 80)
print("ROI BY SURFACE")
print("=" * 80)

for surface, tmp in bets.groupby(
    "surface"
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
    print(surface)
    print("-" * 40)
    print("BETS :", len(tmp))
    print("HIT  :", round(hit * 100,2), "%")
    print("ROI  :", round(roi * 100,2), "%")
