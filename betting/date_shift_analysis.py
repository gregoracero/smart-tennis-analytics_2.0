
import pandas as pd

matches = pd.read_parquet(
    "data/parquet/training_matches_with_elo.parquet"
)

odds = pd.read_parquet(
    "data/parquet/atp_odds_2020_2026.parquet"
)

matches = matches[
    matches["match_date"] >= "2020-01-01"
]

matches["match_date"] = pd.to_datetime(
    matches["match_date"]
)

odds["Date"] = pd.to_datetime(
    odds["Date"]
)

match_dates = set(
    matches["match_date"].dt.date
)

for shift in [-2,-1,0,1,2]:

    shifted = (
        odds["Date"]
        +
        pd.Timedelta(days=shift)
    )

    overlap = len(
        set(
            shifted.dt.date
        )
        &
        match_dates
    )

    print()
    print(
        f"SHIFT {shift:+d}"
    )

    print(
        overlap
    )
