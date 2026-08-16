
import pandas as pd

full = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

top5 = pd.read_parquet(
    "tests/reports/top5_temporal_predictions.parquet"
)

print()
print("=" * 80)
print("FULL MODEL")
print("=" * 80)

print()

print(
    full["edge"]
    .describe()
)

print()

print("=" * 80)
print("TOP5 MODEL")
print("=" * 80)

print()

print(
    top5["top5_edge"]
    .describe()
)
