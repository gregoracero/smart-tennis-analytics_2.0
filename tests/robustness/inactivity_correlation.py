
import pandas as pd

df = pd.read_parquet(
    "data/parquet/training_matches_with_elo.parquet"
)

print()
print("=" * 80)
print("INACTIVITY CORRELATIONS")
print("=" * 80)

corr_inactivity = (
    df["delta_inactivity_days"]
    .corr(
        df["target"]
    )
)

corr_surface_inactivity = (
    df["delta_surface_inactivity_days"]
    .corr(
        df["target"]
    )
)

print()
print("DELTA_INACTIVITY_DAYS")
print(
    round(
        corr_inactivity,
        6
    )
)

print()
print("DELTA_SURFACE_INACTIVITY_DAYS")
print(
    round(
        corr_surface_inactivity,
        6
    )
)

print()
print("ABSOLUTE VALUES")
print(
    "delta_inactivity_days:",
    round(
        abs(corr_inactivity),
        6
    )
)

print(
    "delta_surface_inactivity_days:",
    round(
        abs(
            corr_surface_inactivity
        ),
        6
    )
)
