
import pandas as pd

df = pd.read_parquet(
    "tests/reports/top5_predictions.parquet"
)

df["pair"] = df.apply(
    lambda r:
    "|".join(
        sorted(
            [
                str(r["player_a"]),
                str(r["player_b"])
            ]
        )
    ),
    axis=1
)

unique = (
    df
    .sort_values(
        "top5_probability",
        ascending=False
    )
    .drop_duplicates(
        subset=[
            "match_date",
            "pair"
        ]
    )
)

output = (
    "tests/reports/"
    "top5_predictions_unique.parquet"
)

unique.to_parquet(
    output,
    index=False
)

print()
print("ROWS ORIGINAL")
print(len(df))

print()
print("ROWS UNIQUE")
print(len(unique))

print()
print("SAVED")
print(output)
