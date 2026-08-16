
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

dogs = df[
    (df["edge"] > 0.05)
    &
    (df["market_probability_a"] < 0.50)
].copy()

print()
print("=" * 80)
print("UNDERDOG FEATURE PROFILE")
print("=" * 80)

cols = [

    "delta_elo",
    "delta_surface_elo",

    "delta_rank",
    "delta_rank_points",

    "delta_inactivity_days",

    "delta_win_pct_20",

    "delta_service_points_won_pct_20",

    "delta_return_points_won_pct_20"

]

print()

print(
    dogs[cols]
    .describe()
)
