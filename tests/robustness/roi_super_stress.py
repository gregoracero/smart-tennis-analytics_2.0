
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

BASE_BETS = df[
    df["edge"] > 0.05
].copy()

print()
print("=" * 100)
print("ODDS STRESS TEST")
print("=" * 100)

for factor in [
    1.00,
    0.98,
    0.95,
    0.90,
    0.85,
    0.80
]:

    bets = BASE_BETS.copy()

    bets["adj_odds"] = (
        bets["odds_a"] * factor
    )

    bets["profit"] = bets.apply(
        lambda r:
        r["adj_odds"] - 1
        if r["target"] == 1
        else -1,
        axis=1
    )

    roi = (
        bets["profit"].sum()
        /
        len(bets)
    )

    print()

    print(
        f"ODDS x {factor:.2f}"
    )

    print(
        f"ROI = {roi:.2%}"
    )
