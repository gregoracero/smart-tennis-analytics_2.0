
import pandas as pd

pd.set_option(
    "display.max_columns",
    None
)

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

top = (
    df[
        (df["edge"] > 0.05)
        &
        (df["market_probability_a"] < 0.50)
    ]
    .sort_values(
        "edge",
        ascending=False
    )
)

output = (
    "tests/reports/"
    "top_underdogs.csv"
)

top.to_csv(
    output,
    index=False
)

print()

print("Saved:")
print(output)

print()

print(
    top[
        [
            "match_date",
            "player_a",
            "player_b",
            "odds_a",
            "market_probability_a",
            "model_probability",
            "edge",
            "delta_elo",
            "delta_surface_elo",
            "delta_rank_points",
            "delta_inactivity_days",
            "target"
        ]
    ]
    .head(50)
)
