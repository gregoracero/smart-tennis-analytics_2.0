import pandas as pd

SIGNALS = "data/parquet/signal_engine.parquet"

df = pd.read_parquet(SIGNALS)

score = []

for _, row in df.iterrows():

    s = 0

    if pd.notna(row.get("win_pct")):
        s += row["win_pct"] * 0.50

    if pd.notna(row.get("first_serve_won")):
        s += row["first_serve_won"] * 0.20

    if pd.notna(row.get("second_serve_won")):
        s += row["second_serve_won"] * 0.15

    if pd.notna(row.get("bp_saved")):
        s += row["bp_saved"] * 0.15

    score.append(round(s, 2))

df["prediction_score"] = score

df.to_parquet(
    "data/parquet/prediction_features.parquet",
    index=False
)

print(df.shape)

print(
    df[
        [
            "player",
            "surface",
            "prediction_score"
        ]
    ]
    .sort_values(
        "prediction_score",
        ascending=False
    )
    .head(20)
)
