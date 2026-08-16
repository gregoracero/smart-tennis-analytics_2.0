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

groups = {
    "Favorites":
        bets[
            bets[
                "market_probability_a"
            ] > 0.5
        ],

    "Underdogs":
        bets[
            bets[
                "market_probability_a"
            ] < 0.5
        ]
}

for name, tmp in groups.items():

    roi = (
        tmp["profit"].sum()
        /
        len(tmp)
    )

    hit = (
        tmp["target"].mean()
    )

    print()
    print(name)
    print("-" * 40)
    print("BETS :", len(tmp))
    print("HIT  :", round(hit*100,2), "%")
    print("ROI  :", round(roi*100,2), "%")