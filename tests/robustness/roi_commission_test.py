
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

bets = df[
    df["edge"] > 0.05
].copy()

print()
print("=" * 100)
print("COMMISSION TEST")
print("=" * 100)

for commission in [
    0.00,
    0.02,
    0.05,
    0.10,
    0.15
]:

    tmp = bets.copy()

    tmp["profit"] = tmp.apply(
        lambda r:
        (r["odds_a"] - 1)
        * (1 - commission)
        if r["target"] == 1
        else -1,
        axis=1
    )

    roi = (
        tmp["profit"].sum()
        /
        len(tmp)
    )

    print()

    print(
        f"COMMISSION {commission:.0%}"
    )

    print(
        f"ROI = {roi:.2%}"
    )
