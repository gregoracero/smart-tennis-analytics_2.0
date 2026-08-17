
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

df["match_key"] = (
    df["pair_key"].astype(str)
    + "|"
    + df["match_date"].astype(str)
)

# ------------------------------------------------
# MODELO
# ------------------------------------------------

model_picks = (

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

model_accuracy = (
    model_picks["target"]
    .mean()
)

# ------------------------------------------------
# MERCADO
# ------------------------------------------------

market_picks = (

    df

    .sort_values(
        "market_probability_a",
        ascending=False
    )

    .drop_duplicates(
        "match_key"
    )

    .copy()
)

market_accuracy = (
    market_picks["target"]
    .mean()
)

# ------------------------------------------------
# ROI MODELO
# ------------------------------------------------

model_picks["profit"] = model_picks.apply(

    lambda r:

    r["odds_a"] - 1

    if r["target"] == 1

    else -1,

    axis=1
)

model_roi = (

    model_picks["profit"]
    .sum()

    /

    len(model_picks)

)

# ------------------------------------------------
# ROI MERCADO
# ------------------------------------------------

market_picks["profit"] = market_picks.apply(

    lambda r:

    r["odds_a"] - 1

    if r["target"] == 1

    else -1,

    axis=1
)

market_roi = (

    market_picks["profit"]
    .sum()

    /

    len(market_picks)

)

print()
print("=" * 80)
print("UNIQUE MATCH COMPARISON")
print("=" * 80)

print()

print(
    "UNIQUE MATCHES:",
    len(model_picks)
)

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

print()

print(
    "MODEL ROI:",
    round(
        model_roi * 100,
        2
    ),
    "%"
)

print(
    "MARKET ROI:",
    round(
        market_roi * 100,
        2
    ),
    "%"
)

print()

print(
    "MODEL WINS:",
    int(
        model_picks["target"]
        .sum()
    )
)

print(
    "MODEL LOSSES:",
    len(model_picks)
    -
    int(
        model_picks["target"]
        .sum()
    )
)

print()

print(
    "MARKET WINS:",
    int(
        market_picks["target"]
        .sum()
    )
)

print(
    "MARKET LOSSES:",
    len(market_picks)
    -
    int(
        market_picks["target"]
        .sum()
    )
)
