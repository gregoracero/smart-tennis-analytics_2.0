import pandas as pd

HIST = "data/parquet/all_matches.parquet"
ONGOING = "data/raw/tml/ongoing_tourneys.csv"
OUTPUT = "data/parquet/master_matches.parquet"

print("Loading historical...")

hist = pd.read_parquet(HIST)

print("Loading ongoing...")

ongoing = pd.read_csv(ONGOING)

historical_rows = len(hist)
ongoing_rows = len(ongoing)

# --------------------------------------------------
# Normalización
# --------------------------------------------------

for df in (hist, ongoing):

    if "tourney_date" in df.columns:
        df["tourney_date"] = (
            df["tourney_date"]
            .astype(str)
        )

# --------------------------------------------------
# Detectar qué partidos de ongoing
# ya existen en histórico
# --------------------------------------------------

keys = [
    "tourney_date",
    "winner_name",
    "loser_name",
    "score"
]

hist_keys = set(
    hist[keys]
    .astype(str)
    .apply(tuple, axis=1)
)

ongoing["exists_in_hist"] = (
    ongoing[keys]
    .astype(str)
    .apply(tuple, axis=1)
    .isin(hist_keys)
)

new_matches = ongoing[
    ~ongoing["exists_in_hist"]
].copy()

new_matches_added = len(new_matches)

print()
print(f"New matches detected: {new_matches_added}")

# --------------------------------------------------
# Construcción incremental
# --------------------------------------------------

current = pd.concat(
    [
        hist,
        new_matches.drop(
            columns=["exists_in_hist"]
        )
    ],
    ignore_index=True,
    sort=False
)

# --------------------------------------------------
# Normalización de tipos para parquet
# --------------------------------------------------

for col in current.columns:

    if current[col].dtype == "object":

        current[col] = (
            current[col]
            .fillna("")
            .astype(str)
        )

# --------------------------------------------------
# Guardar
# --------------------------------------------------

current.to_parquet(
    OUTPUT,
    index=False
)

print()
print("=" * 60)
print("MATCH CONSOLIDATION")
print("=" * 60)

print(f"Historical rows    : {historical_rows}")
print(f"Ongoing rows       : {ongoing_rows}")
print(f"New matches added  : {new_matches_added}")
print(f"Final rows         : {len(current)}")

print()
print(f"Saved: {OUTPUT}")

