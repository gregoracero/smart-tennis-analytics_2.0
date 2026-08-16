
import pandas as pd

df = pd.read_parquet(
    "tests/reports/top5_temporal_predictions.parquet"
)

all_bets = df[
    df["top5_edge"] > 0.05
].copy()

zone = df[

    (df["top5_edge"] > 0.05)

    &

    (df["delta_elo"] < 0)

    &

    (
        df["delta_surface_inactivity_days"]
        < -14
    )

].copy()

for name,data in [

    ("ALL",all_bets),

    ("ZONE",zone)

]:

    data["profit"] = data.apply(
        lambda r:
        r["odds_a"] - 1
        if r["target"] == 1
        else -1,
        axis=1
    )

    print()

    print("=" * 60)

    print(name)

    print("=" * 60)

    print()

    print("BETS")
    print(len(data))

    print()

    print("HIT RATE")
    print(
        round(
            data["target"].mean() * 100,
            2
        )
    )

    print()

    print("ROI")
    print(
        round(
            data["profit"].mean() * 100,
            2
        )
    )
