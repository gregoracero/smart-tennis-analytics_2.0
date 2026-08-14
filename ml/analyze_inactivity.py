import pandas as pd

df = pd.read_parquet(
    "data/parquet/training_matches_with_elo.parquet"
)

for col in [
    "days_inactive_a",
    "days_inactive_b",
    "surface_days_inactive_a",
    "surface_days_inactive_b"
]:

    print()
    print("=" * 60)
    print(col)

    print(
        df[col].describe(
            percentiles=[
                0.5,
                0.75,
                0.90,
                0.95,
                0.99
            ]
        )
    )