
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

bins = [
    0.00,
    0.50,
    0.70,
    0.80,
    0.90,
    0.95,
    0.98,
    0.99,
    1.00
]

df["prob_bin"] = pd.cut(
    df["model_probability"],
    bins=bins,
    include_lowest=True
)

summary = (
    df
    .groupby("prob_bin")
    .agg(
        matches=("target","size"),
        predicted=("model_probability","mean"),
        actual=("target","mean")
    )
)

print(summary)
