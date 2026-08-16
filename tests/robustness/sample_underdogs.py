
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

sample = (
    df[
        (df["edge"] > 0.05)
        &
        (df["market_probability_a"] < 0.50)
    ]
    .sample(
        n=100,
        random_state=42
    )
)

output = (
    "tests/reports/"
    "underdog_sample.csv"
)

sample.to_csv(
    output,
    index=False
)

print()
print("Saved:")
print(output)

print()
print(sample[
    [
        "match_date",
        "player_a",
        "player_b",
        "odds_a",
        "market_probability_a",
        "model_probability",
        "edge",
        "target"
    ]
].head(20))
