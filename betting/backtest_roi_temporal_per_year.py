import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

df["match_date"] = pd.to_datetime(
    df["match_date"]
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

print()
print("=" * 80)
print(f"ROI BY YEAR (EDGE > {EDGE_THRESHOLD:.0%})")
print("=" * 80)

for year, tmp in bets.groupby(
    bets["match_date"].dt.year
):

    total_bets = len(tmp)

    wins = (
        tmp["target"] == 1
    ).sum()

    hit_rate = wins / total_bets

    total_profit = tmp["profit"].sum()

    roi = total_profit / total_bets

    print()
    print(year)
    print("-" * 40)
    print("BETS     :", total_bets)
    print("WINS     :", wins)
    print("HIT RATE :", f"{hit_rate:.2%}")
    print("PROFIT   :", round(total_profit, 2))
    print("ROI      :", f"{roi:.2%}")

print()