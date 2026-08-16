
import pandas as pd

df = pd.read_parquet(
    "tests/reports/top5_temporal_predictions.parquet"
)

bets = df[

    (df["top5_edge"] > 0.10)

    &

    (df["delta_elo"] < 0)

    &

    (
        df["delta_surface_inactivity_days"]
        < -14
    )

].copy()

bets = bets.sort_values(
    "match_date"
)

bets["profit"] = bets.apply(
    lambda r:
    r["odds_a"] - 1
    if r["target"] == 1
    else -1,
    axis=1
)

bets["cum_profit"] = (
    bets["profit"]
    .cumsum()
)

bets["peak"] = (
    bets["cum_profit"]
    .cummax()
)

bets["drawdown"] = (
    bets["cum_profit"]
    -
    bets["peak"]
)

print()

print("BETS")
print(len(bets))

print()

print("TOTAL PROFIT")
print(
    round(
        bets["profit"].sum(),
        2
    )
)

print()

print("MAX DRAWDOWN")
print(
    round(
        bets["drawdown"].min(),
        2
    )
)
