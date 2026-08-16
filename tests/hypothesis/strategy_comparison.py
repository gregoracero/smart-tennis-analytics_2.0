
import pandas as pd

df = pd.read_parquet(
    "tests/reports/top5_temporal_predictions.parquet"
)

strategies = {

    "A_ALL": (

        (df["top5_edge"] > 0.05)

    ),

    "B_PREMIUM": (

        (df["top5_edge"] > 0.05)

        &

        (df["delta_elo"] < 0)

        &

        (
            df["delta_surface_inactivity_days"]
            < -14
        )

    ),

    "C_PREMIUM_EDGE10": (

        (df["top5_edge"] > 0.10)

        &

        (df["delta_elo"] < 0)

        &

        (
            df["delta_surface_inactivity_days"]
            < -14
        )

    ),

    "D_PREMIUM_EDGE15": (

        (df["top5_edge"] > 0.15)

        &

        (df["delta_elo"] < 0)

        &

        (
            df["delta_surface_inactivity_days"]
            < -14
        )

    )

}

print()
print("=" * 80)
print("STRATEGY COMPARISON")
print("=" * 80)

results = []

for name,mask in strategies.items():

    bets = df[
        mask
    ].copy()

    bets["profit"] = bets.apply(
        lambda r:
        r["odds_a"] - 1
        if r["target"] == 1
        else -1,
        axis=1
    )

    roi = bets["profit"].mean()

    hit_rate = (
        bets["target"]
        .mean()
    )

    avg_odds = (
        bets["odds_a"]
        .mean()
    )

    results.append({

        "strategy": name,

        "bets": len(bets),

        "hit_rate":
            round(
                hit_rate * 100,
                2
            ),

        "roi":
            round(
                roi * 100,
                2
            ),

        "avg_odds":
            round(
                avg_odds,
                2
            )
    })

summary = pd.DataFrame(
    results
)

print()
print(summary)

summary.to_csv(
    "tests/reports/strategy_comparison.csv",
    index=False
)

print()
print(
    "Saved:"
)

print(
    "tests/reports/strategy_comparison.csv"
)
