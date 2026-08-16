
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

print()
print("=" * 100)
print("MODEL PENALTY TEST")
print("=" * 100)

for penalty in [
    1.00,
    0.98,
    0.95,
    0.90,
    0.85
]:

    tmp = df.copy()

    tmp["adj_model"] = (
        tmp["model_probability"]
        * penalty
    )

    tmp["adj_edge"] = (
        tmp["adj_model"]
        -
        tmp["market_probability_a"]
    )

    bets = tmp[
        tmp["adj_edge"] > 0.05
    ].copy()

    if len(bets) == 0:
        continue

    bets["profit"] = bets.apply(
        lambda r:
        r["odds_a"] - 1
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
        f"MODEL x {penalty:.2f}"
    )

    print(
        f"BETS = {len(bets)}"
    )

    print(
        f"ROI = {roi:.2%}"
    )
