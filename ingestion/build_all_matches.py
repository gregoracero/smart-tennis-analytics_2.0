from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/tml")
OUTPUT = Path("data/parquet/all_matches.parquet")

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

files = []

for f in RAW_DIR.glob("*.csv"):

    name = f.name.lower()

    if "_wta" in name:
        continue

    if "ongoing" in name:
        continue

    if "amateur" in name:
        continue

    if name == "atp_database.csv":
        continue

    files.append(f)

files = sorted(files)

print("=" * 60)
print("FILES SELECTED")
print("=" * 60)

for f in files[-10:]:
    print(f.name)

print()
print(f"Total files: {len(files)}")
print()

dfs = []

for file in files:

    print(f"Loading {file.name}")

    try:

        df = pd.read_csv(
            file,
            low_memory=False
        )

        df["source_file"] = file.name

        dfs.append(df)

    except Exception as e:

        print(f"ERROR {file.name}")
        print(e)

if not dfs:

    raise Exception(
        "No valid CSV files loaded"
    )

result = pd.concat(
    dfs,
    ignore_index=True,
    sort=False
)

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print("Rows :", len(result))
print("Cols :", len(result.columns))

# --------------------------------------------------
# NORMALIZAR TODAS LAS COLUMNAS OBJECT
# --------------------------------------------------

object_cols = result.select_dtypes(
    include=["object"]
).columns

print()
print(f"Object columns detected: {len(object_cols)}")

for col in object_cols:

    result[col] = (
        result[col]
        .fillna("")
        .astype(str)
    )

# --------------------------------------------------
# GUARDAR
# --------------------------------------------------

result.to_parquet(
    OUTPUT,
    index=False
)

print()
print(f"Saved -> {OUTPUT}")