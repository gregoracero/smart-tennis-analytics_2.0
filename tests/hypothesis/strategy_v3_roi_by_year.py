
import pandas as pd

df = pd.read_parquet(
    "tests/reports/top5_temporal_predictions.parquet"
)

df["match_date"] = pd.to_datetime(
    df["match_date"]
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

bets["year"] = (
    bets["match_date"]
    .dt.year
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

    .groupby("year")

    .agg(
        bets=("target","size"),
        wins=("target","sum"),
        hit_rate=("target","mean"),
        roi=("profit","mean"),
        total_profit=("profit","sum"),
        avg_odds=("odds_a","mean")
    )

)

summary["hit_rate"] = (
    summary["hit_rate"] * 100
)

summary["roi"] = (
    summary["roi"] * 100
)

print()
print("=" * 80)
print("ROI BY YEAR")
print("=" * 80)
print()
print(summary)

summary.to_csv(
    "tests/reports/strategy_v3_roi_by_year.csv"
)

print()
print(
    "Saved: tests/reports/strategy_v3_roi_by_year.csv"
)
