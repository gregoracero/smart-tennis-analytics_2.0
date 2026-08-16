
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

bets = df[
    df["edge"] > 0.05
].copy()

for reduction in [0.98, 0.95, 0.90]:

    tmp = bets.copy()

    tmp["adj_odds"] = (
        tmp["odds_a"] * reduction
    )

    tmp["profit"] = tmp.apply(
        lambda r:
        r["adj_odds"] - 1
        if r["target"] == 1
        else -1,
        axis=1
    )

    roi = (
        tmp["profit"].sum()
        / len(tmp)
    )

    print()
    print(
        f"ODDS x {reduction}"
    )
    print(
        f"ROI = {roi:.2%}"
    )
