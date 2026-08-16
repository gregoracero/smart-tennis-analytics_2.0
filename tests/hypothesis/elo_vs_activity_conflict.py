
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

bets = df[
    (df["edge"] > 0.05)
    &
    (df["delta_elo"] < 0)
    &
    (df["delta_inactivity_days"] < -14)
].copy()

bets["profit"] = bets.apply(
    lambda r:
    r["odds_a"] - 1
    if r["target"] == 1
    else -1,
    axis=1
)

print()

print("BETS")
print(len(bets))

print()

print("WIN RATE")
print(
    round(
        bets["target"].mean() * 100,
        2
    )
)

print()

print("ROI")
print(
    round(
        bets["profit"].mean() * 100,
        2
    )
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
