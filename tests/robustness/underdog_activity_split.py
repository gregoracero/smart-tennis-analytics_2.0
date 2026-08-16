
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

groups = {

    "DOG + MUCH MORE ACTIVE (< -14)":
        bets[
            bets["delta_inactivity_days"] < -14
        ],

    "DOG + MORE ACTIVE (-14 to 0)":
        bets[
            (bets["delta_inactivity_days"] >= -14)
            &
            (bets["delta_inactivity_days"] < 0)
        ],

    "DOG + LESS ACTIVE (> 0)":
        bets[
            bets["delta_inactivity_days"] > 0
        ]
}

print()
print("=" * 100)
print("UNDERDOG ACTIVITY SPLIT")
print("=" * 100)

for name, tmp in groups.items():

    roi = (
        tmp["profit"].sum()
        /
        len(tmp)
    )

    hit = tmp["target"].mean()

    print()
    print(name)
    print("-" * 40)
    print("BETS :", len(tmp))
    print("HIT  :", round(hit*100,2), "%")
    print("ROI  :", round(roi*100,2), "%")
