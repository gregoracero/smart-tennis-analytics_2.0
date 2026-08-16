
import pandas as pd

matches = pd.read_parquet(
    "data/parquet/training_matches_with_elo.parquet"
)

odds = pd.read_parquet(
    "data/parquet/atp_odds_2020_2026.parquet"
)

matches["match_date"] = pd.to_datetime(
    matches["match_date"]
)

odds["Date"] = pd.to_datetime(
    odds["Date"]
)

print()
print("MATCHES")
print(matches.shape)

print()
print("ODDS")
print(odds.shape)

print()
print("MATCH RANGE")
print(matches["match_date"].min())
print(matches["match_date"].max())

print()
print("ODDS RANGE")
print(odds["Date"].min())
print(odds["Date"].max())

common_dates = set(

    matches["match_date"]
    .dt.date

).intersection(

    set(
        odds["Date"]
        .dt.date
    )

)

print()
print("COMMON DATES")

print(
    len(common_dates)
)

print()

coverage = (

    matches["match_date"]
    .dt.date
    .isin(common_dates)

).mean()

print(
    "DATE COVERAGE:",
    round(
        coverage * 100,
        2
    ),
    "%"
)
