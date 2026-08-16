
import pandas as pd

df = pd.read_parquet(
    "tests/reports/top5_temporal_predictions.parquet"
)

bets = df[

    (df["top5_edge"] > 0.05)

    &

    (df["delta_elo"] < 0)

    &

    (
        df["delta_surface_inactivity_days"]
        < -14
    )

].copy()

bets["profit"] = bets.apply(
    lambda r:
    r["odds_a"] - 1
    if r["target"] == 1
    else -1,
    axis=1
)

print()

print("PREMIUM ROI ZONE")

print()

print("BETS")
print(len(bets))

print()

print("HIT RATE")
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

print("AVG ODDS")
print(
    round(
        bets["odds_a"].mean(),
        2
    )
)
