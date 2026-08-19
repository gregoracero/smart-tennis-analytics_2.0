from pathlib import Path
import pandas as pd

ROOT = Path(
    "data/raw/tennis_data_co_uk_v2"
)

OUT = Path(
    "data/parquet/v2/tennis_data_co_uk_v2.parquet"
)

# ==========================================================
# DATA QUALITY REMEDIATION
# ==========================================================

SURFACE_OVERRIDES = {

    # ATP Bogotá se jugó en Hard
    ("Claro Open Colombia", 2013): "Hard",
    ("Claro Open Colombia", 2014): "Hard",
    ("Claro Open Colombia", 2015): "Hard",
    
    ("Brasil Open", 2003): "Hard",
    
    ("Legg Mason Classic", 2004): "Hard",
    ("Vietnam Open", 2005): "Carpet",
    ("Sardegna Open", 2021): "Clay",
    ("AnyTech365 Andalucia Open", 2021): "Clay",
    ("Masters Cup", 2005): "Carpet",
}

# ==========================================================
# MATCH RESULT OVERRIDES
# ==========================================================

MATCH_RESULT_OVERRIDES = [

    {
        "season": 2026,
        "tournament": "ABN AMRO World Tennis Tournament",
        "winner": "Bergs Z.",
        "loser": "Medjedovic H.",
        "correct_winner": "Medjedovic H.",
        "correct_loser": "Bergs Z."
    },
    {
        "season": 2008,
        "tournament": "BNP Paribas",
        "winner": "Nalbandian D.",
        "loser": "Tsonga J.W.",
        "correct_winner": "Tsonga J.W.",
        "correct_loser": "Nalbandian D."
    },

    {
        "season": 2021,
        "tournament": "Masters Cup",
        "winner": "Berrettini M.",
        "loser": "Zverev A.",
        "correct_winner": "Zverev A.",
        "correct_loser": "Berrettini M."
    },

    {
        "season": 2023,
        "tournament": "Grand Prix Hassan II",
        "winner": "Kuzmanov D.",
        "loser": "Carballes Baena R.",
        "correct_winner": "Carballes Baena R.",
        "correct_loser": "Kuzmanov D."
    },

    {
        "season": 2023,
        "tournament": "Shanghai Masters",
        "winner": "Kecmanovic M.",
        "loser": "Bu Y.",
        "correct_winner": "Bu Y.",
        "correct_loser": "Kecmanovic M."
    },

    {
        "season": 2024,
        "tournament": "Masters Cup",
        "winner": "De Minaur A.",
        "loser": "Fritz T.",
        "correct_winner": "Fritz T.",
        "correct_loser": "De Minaur A."
    },

    {
        "season": 2026,
        "tournament": "ABN AMRO World Tennis Tournament",
        "winner": "Bergs Z.",
        "loser": "Medjedovic H.",
        "correct_winner": "Medjedovic H.",
        "correct_loser": "Bergs Z."
    },
    {
        "season": 2003,
        "tournament": "Bellsouth Open",
        "winner": "Rios M.",
        "loser": "Sanchez D.",
        "correct_winner": "Sanchez D.",
        "correct_loser": "Rios M."
    },
    {
        "season": 2005,
        "tournament": "ABN AMRO World Tennis Tournament",
        "winner": "Johansson T.",
        "loser": "Ljubicic I.",
        "correct_winner": "Ljubicic I.",
        "correct_loser": "Johansson T."
    },
    {
        "season": 2005,
        "tournament": "Regions Morgan Keegan Championships",
        "winner": "Mirnyi M.",
        "loser": "Carlsen K.",
        "correct_winner": "Carlsen K.",
        "correct_loser": "Mirnyi M."
    },
    {
        "season": 2009,
        "tournament": "Chennai Open",
        "winner": "Schuettler R.",
        "loser": "Devvarman S.",
        "correct_winner": "Devvarman S.",
        "correct_loser": "Schuettler R."
    },
    {
        "season": 2009,
        "tournament": "Hall of Fame Championships",
        "winner": "Querrey S.",
        "loser": "Ram R.",
        "correct_winner": "Ram R.",
        "correct_loser": "Querrey S."
    },
    {
        "season": 2001,
        "tournament": "Generali Open",
        "winner": "Golmard J.",
        "loser": "Diaz J.",
        "correct_winner": "Diaz J.",
        "correct_loser": "Golmard J."
    },
    {
        "season": 2005,
        "tournament": "ABN AMRO World Tennis Tournament",
        "winner": "Federer R.",
        "loser": "Johansson T.",

        "correct_winner": "Federer R.",
        "correct_loser": "Ljubicic I."
    },
    {
        "season": 2007,
        "tournament": "BMW Open",
        "winner": "Bachinger M.",
        "loser": "Beck K.",
        "correct_winner": "Bachinger M.",
        "correct_loser": "Beck A."
    },
    {
        "season": 2007,
        "tournament": "Hall of Fame Championships",
        "winner": "Healey N.",
        "loser": "Levine I.",
        "correct_winner": "Healey N.",
        "correct_loser": "Levine J."
    },
    {
        "season": 2004,
        "tournament": "TD Waterhouse Cup",
        "winner": "Johansson J.",
        "loser": "Elseneer G.",
        "correct_winner": "Johansson J.",
        "correct_loser": "Elsner D."
    },
    {
        "season": 2003,
        "tournament": "Copenhagen Open",
        "winner": "Larsson M.",
        "loser": "Nielsen M.",
        "correct_winner": "Larsson M.",
        "correct_loser": "Nielsen F."
    },
    {
        "season": 2005,
        "tournament": "International Championships",
        "winner": "Gabashvili T.",
        "loser": "Blake T.",
        "correct_winner": "Gabashvili T.",
        "correct_loser": "Blake J."
    },
    
    
    
    
    


    
    
]

# ==========================================================
# TOURNAMENT OVERRIDES
# ==========================================================

TOURNAMENT_OVERRIDES = [

    {
        "from_tournament": "European Open",
        "surface": "Clay",
        "to_tournament": "Hamburg"
    }

]

# ==========================================================
# PLAYER OVERRIDES
# ==========================================================

PLAYER_OVERRIDES = {

    "Stebe C-M.": "Stebe C.M.",

    "Barrios Vera M.T.": "Barrios M.",
    
    "Struff J-L.": "Struff J.L.",
    
    "Herbert P-H.": "Herbert P.H.",
    
    "Schuttler P.": "Schuettler P.",
    
    "Nedovyesov O.": "Nedovyesov A.",
    
    "Sultan-Khalfan A.": "Khalfan S.",
    
    "Benneteau A.": "Benneteau J.",
    
    "Monteiro J.": "Monteiro T.",
    
    "Vicente M.": "Vicente F.",
    
    "Rascon T.": "Rascon J.L.",
    
    "March O.": "Marach O.",
    
    "Marin L.": "Morejon L.A.",
    
    "Rascon T.": "Rascon-Lope J.L.",   
}

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

        except Exception:

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

for col in ["Winner", "Loser"]:

    result[col] = (
        result[col]
        .astype(str)
        .str.strip()
    )

    result[col] = (
        result[col]
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
    )
    
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
# APPLY PLAYER OVERRIDES
# ==========================================================

print()
print("APPLYING PLAYER OVERRIDES...")

for col in ["Winner", "Loser"]:

    before = result[col].copy()

    result[col] = result[col].replace(
        PLAYER_OVERRIDES
    )

    affected_rows = int(
        (before != result[col])
        .sum()
    )

    if affected_rows > 0:

        print(
            f"[PLAYER] {col} "
            f"rows={affected_rows}"
        )
        

# ==========================================================
# APPLY MATCH RESULT OVERRIDES
# ==========================================================

print()
print("APPLYING MATCH RESULT OVERRIDES...")

for fix in MATCH_RESULT_OVERRIDES:

    mask = (
        (result["season"] == fix["season"])
        &
        (result["Tournament"] == fix["tournament"])
        &
        (result["Winner"] == fix["winner"])
        &
        (result["Loser"] == fix["loser"])
    )

    affected_rows = int(mask.sum())

    if affected_rows:

        result.loc[
            mask,
            "Winner"
        ] = fix["correct_winner"]

        result.loc[
            mask,
            "Loser"
        ] = fix["correct_loser"]

        print(
            f"[RESULT] "
            f"{fix['tournament']} "
            f"{fix['season']} "
            f"rows={affected_rows}"
        )        

    
# ==========================================================
# DATE DERIVED FIELDS
# ==========================================================

if "Date" in result.columns:

    date_dt = pd.to_datetime(
        result["Date"],
        errors="coerce"
    )

    result["match_year"] = (
        date_dt.dt.year
    )

    result["match_month"] = (
        date_dt.dt.month
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

        (result["Tournament"] == tournament)

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
                "Surface"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        result.loc[
            mask,
            "Surface"
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
# APPLY TOURNAMENT OVERRIDES
# ==========================================================

print()
print("APPLYING TOURNAMENT OVERRIDES...")

for override in TOURNAMENT_OVERRIDES:

    mask = (

        (result["Tournament"] == override["from_tournament"])

        &

        (result["Surface"] == override["surface"])

    )

    affected_rows = int(
        mask.sum()
    )

    if affected_rows > 0:

        result.loc[
            mask,
            "Tournament"
        ] = override["to_tournament"]

        print(
            f"[TOURNAMENT] "
            f"{override['from_tournament']} -> "
            f"{override['to_tournament']} "
            f"({affected_rows} rows)"
        )
        
# ==========================================================
# AUCKLAND / ADELAIDE 2020 SWAP
# ==========================================================

print()
print("APPLYING AUCKLAND / ADELAIDE 2020 SWAP...")

mask_asb = (
    (result["Tournament"] == "ASB Classic")
    &
    (result["season"] == 2020)
)

mask_adelaide = (
    (result["Tournament"] == "Adelaide International")
    &
    (result["season"] == 2020)
)

asb_rows = int(mask_asb.sum())
adelaide_rows = int(mask_adelaide.sum())

if asb_rows > 0 or adelaide_rows > 0:

    result.loc[
        mask_asb,
        "Tournament"
    ] = "__TMP_AUCKLAND_2020__"

    result.loc[
        mask_adelaide,
        "Tournament"
    ] = "ASB Classic"

    result.loc[
        result["Tournament"]
        ==
        "__TMP_AUCKLAND_2020__",
        "Tournament"
    ] = "Adelaide International"

    print(
        f"[TOURNAMENT_SWAP] "
        f"ASB Classic <-> Adelaide International "
        f"(ASB={asb_rows}, Adelaide={adelaide_rows})"
    )

# ----------------------------------------------------------
# HEINEKEN OPEN
#
# Enero      -> Auckland
# Septiembre -> Shanghai
# Octubre    -> Shanghai
# ----------------------------------------------------------

heineken_shanghai_mask = (

    (result["Tournament"] == "Heineken Open")

    &

    (result["match_month"] >= 9)

)

affected_rows = int(
    heineken_shanghai_mask.sum()
)

if affected_rows > 0:

    result.loc[
        heineken_shanghai_mask,
        "Tournament"
    ] = "Shanghai"

    print(
        f"[TOURNAMENT] "
        f"Heineken Open -> Shanghai "
        f"({affected_rows} rows)"
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
