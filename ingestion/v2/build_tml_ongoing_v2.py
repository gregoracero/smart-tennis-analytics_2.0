from pathlib import Path
import pandas as pd

ROOT = Path(
    "data/raw/ongoing_v2"
)

OUT = Path(
    "data/parquet/v2/tml_ongoing_v2.parquet"
)

frames = []

files = []

files.extend(
    ROOT.rglob("*.csv")
)

files.extend(
    ROOT.rglob("*.parquet")
)

print()
print("FILES FOUND:", len(files))

for file in files:

    print(file)

    try:

        if file.suffix == ".parquet":

            df = pd.read_parquet(file)

        else:

            df = pd.read_csv(
                file,
                low_memory=False
            )
            
            # ------------------------------------------------
            # NORMALIZE DATES
            # ------------------------------------------------

            if "tourney_date" in df.columns:

                df["tourney_date"] = pd.to_datetime(
                    df["tourney_date"]
                    .astype(str)
                    .str.extract(r"(\d{8})")[0],
                    format="%Y%m%d",
                    errors="coerce"
                )
            
            

        df["source_file"] = file.name

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