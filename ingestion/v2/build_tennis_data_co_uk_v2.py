from pathlib import Path
import pandas as pd

ROOT = Path(
    "data/raw/tennis_data_co_uk_v2"
)

OUT = Path(
    "data/parquet/v2/tennis_data_co_uk_v2.parquet"
)

frames = []

files = list(
    ROOT.rglob("*.xlsx")
)

files += list(
    ROOT.rglob("*.xls")
)

print()
print("FILES FOUND:", len(files))

for file in files:

    print(file)

    try:

        df = pd.read_excel(
            file
        )

        df["source_file"] = file.name

        try:

            df["season"] = int(
                file.stem[:4]
            )
        except:
            df["season"] = None

        frames.append(df)

    except Exception as e:

        print()
        print("ERROR")
        print(file)
        print(e)

result = pd.concat(
    frames,
    ignore_index=True
)

print()
print("ROWS:", len(result))
print("COLS:", len(result.columns))

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

print()
print("NORMALIZING TYPES...")

for col in result.columns:

    if result[col].dtype == "object":

        result[col] = (
            result[col]
            .astype(str)
            .str.strip()
        )

        result[col] = result[col].replace(
            {
                "": pd.NA,
                "\t": pd.NA,
                "nan": pd.NA,
                "None": pd.NA
            }
        )

result.to_parquet(
    OUT,
    index=False
)

print()
print("SAVED:", OUT)