
import pandas as pd
from pathlib import Path

INPUT_DIR = (
    Path("data/raw/tennis_data")
)

files = sorted(
    INPUT_DIR.glob("*.xlsx")
)

dfs = []

for file in files:

    print(
        f"Loading {file.name}"
    )

    df = pd.read_excel(file)

    keep = [

        "Date",

        "Winner",
        "Loser",

        "Surface",

        "Series",

        "AvgW",
        "AvgL",

        "MaxW",
        "MaxL",

        "PSW",
        "PSL",

        "B365W",
        "B365L"
    ]

    keep = [
        c
        for c in keep
        if c in df.columns
    ]

    df = df[keep]

    dfs.append(df)

master = pd.concat(
    dfs,
    ignore_index=True
)

master["Date"] = pd.to_datetime(
    master["Date"]
)

master = master.sort_values(
    "Date"
)

odds_cols = [

    "AvgW",
    "AvgL",

    "MaxW",
    "MaxL",

    "PSW",
    "PSL",

    "B365W",
    "B365L",

    "BFEW",
    "BFEL"
]

for col in odds_cols:

    if col in master.columns:

        master[col] = pd.to_numeric(
            master[col],
            errors="coerce"
        )

master["market_prob_winner"] = (
    1 / master["AvgW"]
)

master["market_prob_loser"] = (
    1 / master["AvgL"]
)

print()
print("NULL ODDS")

for col in odds_cols:

    if col in master.columns:

        print(
            col,
            master[col]
            .isna()
            .mean()
        )

master.to_parquet(
    "data/parquet/atp_odds_2020_2026.parquet"
)

print()
print("SHAPE")
print(master.shape)

print()
print(
    "Saved: data/parquet/atp_odds_2020_2026.parquet"
)
