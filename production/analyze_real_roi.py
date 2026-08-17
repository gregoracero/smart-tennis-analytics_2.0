
import pandas as pd

df = pd.read_parquet(
    "data/parquet/atp_matches_with_predictions.parquet"
)

print()
print("=" * 90)
print("REAL ROI BY MODEL CONFIDENCE")
print("=" * 90)

thresholds = [

    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90

]

rows = []

for threshold in thresholds:

    bets = df[
        df["model_probability"]
        >= threshold
    ].copy()

    if len(bets) == 0:

        continue

    bets["profit"] = bets.apply(

        lambda r:

        r["odds_a"] - 1

        if r["target"] == 1

        else -1,

        axis=1
    )

    total_profit = (
        bets["profit"]
        .sum()
    )

    roi = (
        total_profit
        /
        len(bets)
    )

    rows.append({

        "threshold":
            threshold,

        "bets":
            len(bets),

        "wins":
            int(
                bets["target"]
                .sum()
            ),

        "win_rate":
            round(
                bets["target"]
                .mean(),
                4
            ),

        "avg_odds":
            round(
                bets["odds_a"]
                .mean(),
                3
            ),

        "profit_units":
            round(
                total_profit,
                2
            ),

        "roi":
            round(
                roi * 100,
                2
            )
    })

summary = pd.DataFrame(
    rows
)

print()
print(summary)

print()
print("=" * 90)
print("BEST ROI")
print("=" * 90)
print()

print(
    summary.sort_values(
        "roi",
        ascending=False
    )
)
