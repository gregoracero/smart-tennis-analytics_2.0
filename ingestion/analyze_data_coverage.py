import pandas as pd

df = pd.read_parquet(
    "data/parquet/player_matches.parquet"
)

df["year"] = (
    df["tourney_date"]
    .astype(str)
    .str[:4]
)

cols = [
    "minutes",
    "ace",
    "df",
    "svpt",
    "first_in",
    "first_won",
    "second_won",
    "bp_saved",
    "bp_faced"
]

rows = []

for year, g in df.groupby("year"):

    row = {
        "year": year,
        "rows": len(g)
    }

    for col in cols:

        row[col] = round(
            g[col]
            .notna()
            .mean()
            * 100,
            2
        )

    rows.append(row)

result = (
    pd.DataFrame(rows)
    .sort_values("year")
)

print(result)

result.to_parquet(
    "data/parquet/data_coverage_by_year.parquet",
    index=False
)
