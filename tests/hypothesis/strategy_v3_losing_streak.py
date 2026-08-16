
import pandas as pd

df = pd.read_parquet(
    "tests/reports/top5_temporal_predictions.parquet"
)

bets = df[

    (df["top5_edge"] > 0.10)

    &

    (df["delta_elo"] < 0)

    &

    (
        df["delta_surface_inactivity_days"]
        < -14
    )

].copy()

bets = bets.sort_values(
    "match_date"
)

results = bets["target"].tolist()

max_losing = 0
current_losing = 0

max_winning = 0
current_winning = 0

for r in results:

    if r == 1:

        current_winning += 1
        max_winning = max(
            max_winning,
            current_winning
        )

        current_losing = 0

    else:

        current_losing += 1

        max_losing = max(
            max_losing,
            current_losing
        )

        current_winning = 0

print()

print("TOTAL BETS")
print(len(results))

print()

print("LONGEST WINNING STREAK")
print(max_winning)

print()

print("LONGEST LOSING STREAK")
print(max_losing)
