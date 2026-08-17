
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

df["match_key"] = (
    df["pair_key"].astype(str)
    + "|"
    + df["match_date"].astype(str)
)

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

picks["profit"] = picks.apply(

    lambda r:

    r["odds_a"] - 1

    if r["target"] == 1

    else -1,

    axis=1
)

print()
print("=" * 100)
print("FULL MODEL")
print("=" * 100)

roi = (
    picks["profit"].sum()
    / len(picks)
)

print()
print("MATCHES :", len(picks))
print("WINS    :", int(picks["target"].sum()))
print("LOSSES  :", len(picks) - int(picks["target"].sum()))
print("ACC     :", round(picks["target"],4))
print("ROI     :", round(roi * 100, 2), "%")

print()
print("=" * 100)
print("ROI BY EDGE")
print("=" * 100)

rows = []

for threshold in [

    0.00,
    0.03,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30

]:

    bets = picks[
        picks["edge"] >= threshold
    ].copy()

    if len(bets) == 0:
        continue

    roi = (
        bets["profit"].sum()
        / len(bets)
    )

    rows.append({

        "edge":
            threshold,

        "bets":
            len(bets),

        "wins":
            int(
                bets["target"].sum()
            ),

        "losses":
            len(bets)
            -
            int(
                bets["target"].sum()
            ),

        "hit_rate":
            round(
                bets["target"].mean(),
                4
            ),

        "avg_probability":
            round(
                bets["model_probability"].mean(),
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

summary = pd.DataFrame(rows)

print()
print(summary)

print()
print("=" * 100)
print("BEST ROI")
print("=" * 100)

print()

print(
    summary.sort_values(
        "roi",
        ascending=False
    )
)
