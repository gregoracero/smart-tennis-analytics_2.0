
import pandas as pd

df = pd.read_parquet(
    "data/parquet/atp_matches_with_predictions_nearest.parquet"
)

print()
print("=" * 100)
print("MODEL VS MARKET")
print("=" * 100)

# -----------------------------
# MODEL
# -----------------------------

model_pick = (
    df["model_probability"] >= 0.5
).astype(int)

model_accuracy = (
    model_pick
    ==
    df["target"]
).mean()

# -----------------------------
# MARKET
# -----------------------------

market_pick = (
    df["market_probability_a"] >= 0.5
).astype(int)

market_accuracy = (
    market_pick
    ==
    df["target"]
).mean()

print()
print(
    "MODEL ACCURACY:",
    round(
        model_accuracy,
        4
    )
)

print(
    "MARKET ACCURACY:",
    round(
        market_accuracy,
        4
    )
)

# -----------------------------
# ROI MODEL
# -----------------------------

model_bets = df.copy()

model_bets["prediction"] = (
    model_bets["model_probability"] >= 0.5
).astype(int)

model_bets = model_bets[
    model_bets["prediction"] == 1
]

model_bets["profit"] = model_bets.apply(

    lambda r:

    r["odds_a"] - 1

    if r["target"] == 1

    else -1,

    axis=1
)

model_roi = (

    model_bets["profit"]
    .sum()

    /

    len(model_bets)

)

print()

print(
    "MODEL ROI:",
    round(
        model_roi * 100,
        2
    ),
    "%"
)

# -----------------------------
# ROI MARKET
# -----------------------------

market_bets = df.copy()

market_bets["prediction"] = (
    market_bets["market_probability_a"] >= 0.5
).astype(int)

market_bets = market_bets[
    market_bets["prediction"] == 1
]

market_bets["profit"] = market_bets.apply(

    lambda r:

    r["odds_a"] - 1

    if r["target"] == 1

    else -1,

    axis=1
)

market_roi = (

    market_bets["profit"]
    .sum()

    /

    len(market_bets)

)

print(
    "MARKET ROI:",
    round(
        market_roi * 100,
        2
    ),
    "%"
)

# -----------------------------
# EDGE ANALYSIS
# -----------------------------

print()
print("=" * 100)
print("ROI BY EDGE")
print("=" * 100)

thresholds = [

    0.00,
    0.05,
    0.10,
    0.15,
    0.20

]

rows = []

for threshold in thresholds:

    bets = df[
        df["edge"]
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

    roi = (

        bets["profit"]
        .sum()

        /

        len(bets)

    )

    rows.append({

        "edge_threshold":
            threshold,

        "bets":
            len(bets),

        "win_rate":
            round(
                bets["target"]
                .mean(),
                4
            ),

        "avg_edge":
            round(
                bets["edge"]
                .mean(),
                4
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
print("=" * 100)
print("BEST EDGE")
print("=" * 100)

print()

print(
    summary.sort_values(
        "roi",
        ascending=False
    )
)

