from pathlib import Path
import pandas as pd

ROOT = Path(
    "data/raw/tml_v2"
)

OUT = Path(
    "data/parquet/v2/tml_v2.parquet"
)

# ==========================================================
# DATA QUALITY REMEDIATION
# ==========================================================

SURFACE_OVERRIDES = {

    ("Basel", 2007): "Carpet",
    ("Basel", 2008): "Carpet",

    ("Paris Masters", 2007): "Carpet",
    ("Paris Masters", 2008): "Carpet",

    ("Zagreb", 2008): "Carpet",
    ("Zagreb", 2009): "Carpet",

    ("Moscow", 2007): "Carpet",
    ("Moscow", 2008): "Carpet",

    ("St. Petersburg", 2008): "Carpet",
    
    ("Santiago", 2025): "Clay",

}

# ==========================================================
# PLAYER NAME OVERRIDES
# ==========================================================

PLAYER_NAME_OVERRIDES = {

    # Miami Masters 2026
    "otic van de Zandschulp":
        "Botic van de Zandschulp",

    "Adrian Boitan":
        "Gabi Adrian Boitan",
}

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

# ==========================================================
# DERIVED FIELDS
# ==========================================================

if "tourney_date" in result.columns:

    tourney_dates = pd.to_datetime(
        result["tourney_date"],
        errors="coerce"
    )

    # ------------------------------------------------------
    # CALENDAR YEAR
    # ------------------------------------------------------

    result["match_year"] = (
        tourney_dates.dt.year
    )

    # ------------------------------------------------------
    # ATP SEASON
    # ------------------------------------------------------

    result["season"] = (
        result["match_year"]
    )

    december_rollover = (

        tourney_dates.dt.month.eq(12)

        &

        tourney_dates.dt.day.ge(29)

    )

    result.loc[
        december_rollover,
        "season"
    ] += 1

    print()
    print(
        "ATP SEASON ADJUSTMENTS:",
        int(december_rollover.sum())
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
        
        
# ==========================================================
# PLAYER NAME NORMALIZATION
# ==========================================================

print()
print("APPLYING PLAYER NAME FIXES...")

total_player_fixes = 0

for col in ["winner_name", "loser_name"]:

    affected = (
        result[col]
        .isin(PLAYER_NAME_OVERRIDES.keys())
    )

    affected_rows = int(affected.sum())

    if affected_rows > 0:

        result[col] = result[col].replace(
            PLAYER_NAME_OVERRIDES
        )

        total_player_fixes += affected_rows

        print(
            f"[PLAYER] {col} "
            f"rows={affected_rows}"
        )

print()
print(
    "TOTAL PLAYER FIXES:",
    total_player_fixes
)

# ==========================================================
# APPLY DATA QUALITY FIXES
# ==========================================================

print()
print("APPLYING DATA QUALITY FIXES...")

total_fixes = 0

for (
    tournament,
    year
), corrected_surface in (
    SURFACE_OVERRIDES.items()
):

    mask = (

        (result["tourney_name"] == tournament)

        &

        (result["match_year"] == year)

    )

    affected_rows = int(
        mask.sum()
    )

    if affected_rows > 0:

        original_surfaces = (
            result.loc[
                mask,
                "surface"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        result.loc[
            mask,
            "surface"
        ] = corrected_surface

        total_fixes += affected_rows

        print(
            f"[SURFACE] "
            f"{tournament} | "
            f"{year} | "
            f"{original_surfaces} -> "
            f"{corrected_surface} | "
            f"rows={affected_rows}"
        )

print()
print(
    "TOTAL DATA QUALITY FIXES:",
    total_fixes
)

# ==========================================================
# TOURNAMENT NORMALIZATION
# ==========================================================

print()
print("APPLYING TOURNAMENT NORMALIZATION...")

total_tournament_fixes = 0

# Winston Salem -> Winston-Salem
winston_mask = (
    result["tourney_name"] == "Winston Salem"
)

affected_rows = int(
    winston_mask.sum()
)

if affected_rows > 0:

    result.loc[
        winston_mask,
        "tourney_name"
    ] = "Winston-Salem"

    total_tournament_fixes += affected_rows

    print(
        "[TOURNAMENT] "
        "Winston Salem -> Winston-Salem "
        f"({affected_rows} rows)"
    )

# ''s-Hertogenbosch -> 's-Hertogenbosch
hertogenbosch_mask = (
    result["tourney_name"] == "''s-Hertogenbosch"
)

affected_rows = int(
    hertogenbosch_mask.sum()
)

if affected_rows > 0:

    result.loc[
        hertogenbosch_mask,
        "tourney_name"
    ] = "'s-Hertogenbosch"

    total_tournament_fixes += affected_rows

    print(
        "[TOURNAMENT] "
        "''s-Hertogenbosch -> 's-Hertogenbosch "
        f"({affected_rows} rows)"
    )

print()
print(
    "TOTAL TOURNAMENT FIXES:",
    total_tournament_fixes
)

# ==========================================================
# SAVE
# ==========================================================

result.to_parquet(
    OUT,
    index=False
)

print()
print("ROWS:", len(result))
print()

print("SAVED:", OUT)