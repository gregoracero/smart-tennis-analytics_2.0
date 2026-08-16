
import pandas as pd

df = pd.read_parquet(
    "tests/reports/top5_predictions_unique.parquet"
)

for threshold in [
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]:

    bets = df[
        df["top5_probability"] > threshold
    ].copy()

    bets["profit"] = bets["target"].apply(
        lambda x: 1 if x == 1 else -1
    )

    roi = bets["profit"].mean()

    print()
    print("=" * 70)
    print(f"THRESHOLD {threshold}")
    print("=" * 70)

    print(
        "BETS:",
        len(bets)
    )

    print(
        "HIT RATE:",
        round(
            bets["target"].mean() * 100,
            2
        ),
        "%"
    )

    print(
        "ROI:",
        round(
            roi * 100,
            2
        ),
        "%"
    )

    if len(bets) > 0:

        print(
            "WINS:",
            int(
                bets["target"].sum()
            )
        )

        print(
            "LOSSES:",
            int(
                len(bets)
                -
                bets["target"].sum()
            )
        )

        print(
            "AVG PROBABILITY:",
            round(
                bets["top5_probability"].mean(),
                4
            )
        )
