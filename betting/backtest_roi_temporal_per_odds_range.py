import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

EDGE_THRESHOLD = 0.05

bets = df[
    df["edge"] > EDGE_THRESHOLD
].copy()

bets["profit"] = bets.apply(
    lambda r:
    r["odds_a"] - 1
    if r["target"] == 1
    else -1,
    axis=1
)

bins = [
    1.00,
    1.50,
    2.00,
    3.00,
    5.00,
    100.00
]

labels = [
    "1.00-1.50",
    "1.50-2.00",
    "2.00-3.00",
    "3.00-5.00",
    "5.00+"
]

bets["odds_bucket"] = pd.cut(
    bets["odds_a"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print()
print("=" * 80)
print(f"ROI BY ODDS RANGE (EDGE > {EDGE_THRESHOLD:.0%})")
print("=" * 80)

for bucket, tmp in bets.groupby(
    "odds_bucket",
    observed=False
):

    if len(tmp) == 0:
        continue

    total_bets = len(tmp)

    wins = (
        tmp["target"] == 1
    ).sum()

    hit_rate = wins / total_bets

    total_profit = tmp["profit"].sum()

    roi = total_profit / total_bets

    avg_odds = tmp["odds_a"].mean()

    print()
    print(bucket)
    print("-" * 40)
    print("BETS      :", total_bets)
    print("AVG ODDS  :", round(avg_odds, 3))
    print("WINS      :", wins)
    print("HIT RATE  :", f"{hit_rate:.2%}")
    print("PROFIT    :", round(total_profit, 2))
    print("ROI       :", f"{roi:.2%}")

print()