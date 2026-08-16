
import pandas as pd

df = pd.read_parquet(
    "data/parquet/atp_matches_with_predictions_nearest.parquet"
)

thresholds = [
    0.03,
    0.05,
    0.08,
    0.10
]

print()
print("=" * 80)
print("ROI BACKTEST")
print("=" * 80)

for threshold in thresholds:

    bets = df[
        df["edge"] > threshold
    ].copy()

    if len(bets) == 0:
        continue

    bets["stake"] = 1.0

    bets["profit"] = bets.apply(
        lambda r:
        r["odds_a"] - 1
        if r["target"] == 1
        else -1,
        axis=1
    )

    total_bets = len(bets)

    wins = (bets["target"] == 1).sum()

    hit_rate = wins / total_bets

    total_profit = bets["profit"].sum()

    total_stake = bets["stake"].sum()

    roi = total_profit / total_stake

    print()
    print(f"EDGE > {threshold:.0%}")
    print("-" * 40)
    print("BETS       :", total_bets)
    print("WINS       :", wins)
    print("HIT RATE   :", f"{hit_rate:.2%}")
    print("PROFIT     :", round(total_profit,2))
    print("ROI        :", f"{roi:.2%}")

print()
