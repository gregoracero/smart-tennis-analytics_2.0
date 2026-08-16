
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

bets["tour"] = bets[
    "tourney_level"
].apply(
    lambda x:
    "CHALLENGER"
    if x == "C"
    else "ATP"
)

bets["profit"] = bets.apply(
    lambda r:
    r["odds_a"] - 1
    if r["target"] == 1
    else -1,
    axis=1
)

summary = (

    bets

    .groupby("tour")

    .agg(
        bets=("target","size"),
        hit_rate=("target","mean"),
        roi=("profit","mean")
    )

)

print()
print(summary)
