
import pandas as pd

df = pd.read_parquet(
    "data/parquet/atp_matches_with_predictions_nearest.parquet"
)

df["match_key"] = (
    df["pair_key_y"].astype(str)
    + "|"
    + df["match_date"].astype(str)
)

# Nos quedamos con la orientaci?n favorita del modelo
picks = (

    df

    .sort_values(
        "model_probability",
        ascending=False
    )

    .drop_duplicates(
        "match_key"
    )

    .copy()
)

print()
print("=" * 100)
print("UNIQUE MATCH ANALYSIS")
print("=" * 100)

print()
print(
    "ROWS:",
    len(df)
)

print(
    "UNIQUE MATCHES:",
    len(picks)
)

print()

accuracy = (
    picks["target"]
    .mean()
)

print(
    "MODEL ACCURACY:",
    round(
        accuracy,
        4
    )
)

# ROI del modelo

picks["profit"] = picks.apply(

    lambda r:

    r["odds_a"] - 1

    if r["target"] == 1

    else -1,

    axis=1
)

roi = (

    picks["profit"]
    .sum()

    /

    len(picks)

)

print(
    "MODEL ROI:",
    round(
        roi * 100,
        2
    ),
    "%"
)

print()

print("=" * 100)
print("ROI BY PROBABILITY THRESHOLD")
print("=" * 100)

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

    bets = picks[
        picks["model_probability"]
        >= threshold
    ].copy()

    if len(bets) == 0:

        continue

    roi = (

        bets["profit"]
        .sum()

        /

        len(bets)

    )

    rows.append({

        "threshold":
            threshold,

        "bets":
            len(bets),

        "win_rate":
            round(
                bets["target"].mean(),
                4
            ),

        "avg_probability":
            round(
                bets["model_probability"].mean(),
                4
            ),

        "avg_odds":
            round(
                bets["odds_a"].mean(),
                3
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
print("ROI BY EDGE")
print("=" * 100)

rows = []

for threshold in [

    0.00,
    0.05,
    0.10,
    0.15,
    0.20

]:

    bets = picks[
        picks["edge"]
        >= threshold
    ].copy()

    if len(bets) == 0:

        continue

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
                bets["target"].mean(),
                4
            ),

        "avg_edge":
            round(
                bets["edge"].mean(),
                4
            ),

        "avg_odds":
            round(
                bets["odds_a"].mean(),
                3
            ),

        "roi":
            round(
                roi * 100,
                2
            )
    })

edge_summary = pd.DataFrame(
    rows
)

print()
print(edge_summary)

