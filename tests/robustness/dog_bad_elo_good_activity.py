
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

bets = df[
    (df["edge"] > 0.05)
    &
    (df["market_probability_a"] < 0.50)
    &
    (df["delta_inactivity_days"] < -14)
    &
    (df["delta_elo"] < 0)
].copy()

bets["profit"] = bets.apply(
    lambda r:
    r["odds_a"] - 1
    if r["target"] == 1
    else -1,
    axis=1
)

if len(bets) == 0:

    print()
    print("NO BETS FOUND")

else:

    roi = (
        bets["profit"].sum()
        /
        len(bets)
    )

    hit = (
        bets["target"].mean()
    )

    print()
    print("=" * 100)
    print("UNDERDOG + WORSE ELO + BETTER ACTIVITY")
    print("=" * 100)

    print()
    print("BETS :", len(bets))
    print(
        "HIT  :",
        round(hit * 100, 2),
        "%"
    )
    print(
        "ROI  :",
        round(roi * 100, 2),
        "%"
    )

    print()
    print("AVG DELTA ELO")
    print(
        round(
            bets["delta_elo"].mean(),
            2
        )
    )

    print()
    print("AVG DELTA INACTIVITY")
    print(
        round(
            bets["delta_inactivity_days"].mean(),
            2
        )
    )

    print()
    print("AVG ODDS")
    print(
        round(
            bets["odds_a"].mean(),
            2
        )
    )
