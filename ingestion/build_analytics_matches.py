import pandas as pd

INPUT = "data/parquet/master_matches.parquet"
OUTPUT = "data/parquet/analytics_matches.parquet"

print("Loading all_matches...")

df = pd.read_parquet(INPUT)

df["tourney_date"] = pd.to_numeric(
    df["tourney_date"],
    errors="coerce"
)

analytics = df[
    df["tourney_date"] >= 20100101
].copy()

analytics.to_parquet(
    OUTPUT,
    index=False
)

print()
print("DONE")
print("Rows :", len(analytics))
print("Cols :", len(analytics.columns))
print("From :", analytics["tourney_date"].min())
print("To   :", analytics["tourney_date"].max())
print()
print(f"Saved: {OUTPUT}")
