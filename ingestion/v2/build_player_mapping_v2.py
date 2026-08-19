from pathlib import Path
import pandas as pd
import re

INPUT_TD = (
    "data/parquet/v2/"
    "tennis_data_co_uk_v2.parquet"
)

INPUT_TML = (
    "data/parquet/v2/"
    "tml_v2.parquet"
)

OUTPUT = (
    "data/parquet/v2/"
    "player_mapping_v2.parquet"
)

# ==========================================================
# MANUAL ALIASES
# ==========================================================

TENNIS_KEY_ALIASES = {

    "dolgopolov o": "dolgopolov a",

    "barrios m": "barrios t",
    #"barrios vera mt": "barrios t",
    
    "viloca ja": "viloca puig ja",
    
    "querry s": "querrey s",
    
    "bogomolov jr a": "bogomolov a",
    "bogomolov jra": "bogomolov a",

    "wang yt": "wang j",
    
    "van assche l": "assche l",
    
    "vallejo d": "vallejo ad",
    
    "van gemerden m": "gemerden m",
    
    "el aarej m": "aarej m",
    
    "lopez jaen ma": "jaen ma",
    
    #"de heart r": "heart r",
    "de heart r": "deheart r",
    #"de heart r": "ryler deheart",
    
    "statham j": "statham r",
    
    "youzhny a": "youzhny m",
    
    "robredo r": "robredo t",
    
    "schuettler p": "schuettler r",
    
    "matsukevitch d": "matsukevich d",
    
    "ancic i": "ancic m",
    
    "guccione a": "guccione c",
    
    "mukund s": "sasikumar m",
    
    "stebe cm": "stebe c",
    
    "verdasco m": "verdasco f",
    
    "stepanek m": "stepanek r",
    
    "kucera v": "kucera k",
    
    "hantschek m": "hantschk m",
    
    "ascione a": "ascione t",
    
    "prpic a": "prpic f",
    
    "struff jl": "struff j",
    
    "herbert ph": "herbert p",
    
    "kunitcin i": "kunitsyn i",
    
    "fish a": "fish m",
    
    "rascon jl": "rascon_lope jl",
    
   
}

# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize_text(text):

    text = str(text)

    text = text.lower()
    
    text = re.sub(
        r"([a-z])\.\s+([a-z])\.",
        r"\1.\2.",
        text
    )

    replacements = {

        "'": "",
        "-": " ",
        ".": "",
        ",": ""

    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    text = " ".join(
        text.split()
    )

    # O'Connell / O Connell
    text = text.replace(
        "o connell",
        "oconnell"
    )
    text = text.replace(
        "del bonis",
        "delbonis"
    )
    text = text.replace(
        "dev varman",
        "devvarman"
    )
    

    return text

# ==========================================================
# TENNIS DATA KEYS
# ==========================================================

def tennis_keys(name):

    name = normalize_text(name)

    name = TENNIS_KEY_ALIASES.get(
        name,
        name
    )

    m = re.match(
        r"^(.*)\s([a-z]+)$",
        name
    )

    if not m:
        return set()

    surname = (
        m.group(1)
        .strip()
        .replace(" ", "_")
    )

    initials = (
        m.group(2)
        .strip()
    )

    keys = set()

    keys.add(
        f"{surname}_{initials}"
    )

    keys.add(
        f"{surname}_{initials[0]}"
    )

    # -----------------------------------------
    # Compound surnames
    # -----------------------------------------

    parts = surname.split("_")

    if len(parts) > 1:

        keys.add(
            f"{parts[0]}_{initials[0]}"
        )

        keys.add(
            f"{parts[-1]}_{initials[0]}"
        )

        keys.add(
            f"{parts[0]}_{initials}"
        )

        keys.add(
            f"{parts[-1]}_{initials}"
        )

    return keys

# ==========================================================
# TML KEYS
# ==========================================================

def tml_keys(name):

    parts = normalize_text(
        name
    ).split()

    if len(parts) < 2:
        return set()

    first_name = parts[0]

    surnames = parts[1:]

    first_initial = (
        first_name[0]
    )

    keys = set()

    # -------------------------------------------------
    # Roberto Bautista Agut
    # -------------------------------------------------

    first_surname = surnames[0]

    keys.add(
        f"{first_surname}_{first_initial}"
    )

    if len(surnames) > 1:

        compound_surname = "_".join(
            surnames
        )

        keys.add(
            f"{compound_surname}_{first_initial}"
        )

        last_surname = surnames[-1]

        keys.add(
            f"{last_surname}_{first_initial}"
        )

    # -------------------------------------------------
    # Pierre Hugues Herbert
    # -> herbert_ph
    # -------------------------------------------------

    if len(parts) >= 3:

        initials = (
            first_name[0]
            +
            parts[1][0]
        )

        last_name = parts[-1]

        keys.add(
            f"{last_name}_{initials}"
        )

        keys.add(
            f"{first_surname}_{initials}"
        )

    # -------------------------------------------------
    # Juan Martin del Potro
    # -> del_potro_j
    # -> del_potro_jm
    # -------------------------------------------------

    if len(parts) >= 4:

        last_two = "_".join(
            parts[-2:]
        )

        keys.add(
            f"{last_two}_{first_initial}"
        )

        initials = (
            first_name[0]
            +
            parts[1][0]
        )

        keys.add(
            f"{last_two}_{initials}"
        )

    return keys

# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading datasets...")

td = pd.read_parquet(INPUT_TD)

tml = pd.read_parquet(INPUT_TML)

td_players = sorted(
    pd.concat(
        [
            td["Winner"],
            td["Loser"]
        ]
    )
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

tml_players = sorted(
    pd.concat(
        [
            tml["winner_name"],
            tml["loser_name"]
        ]
    )
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

# ==========================================================
# BUILD LOOKUPS
# ==========================================================

td_lookup = {}

for player in td_players:

    for key in tennis_keys(player):

        td_lookup.setdefault(
            key,
            set()
        ).add(player)

tml_lookup = {}

for player in tml_players:

    for key in tml_keys(player):

        tml_lookup.setdefault(
            key,
            set()
        ).add(player)

# ==========================================================
# BUILD MAPPING
# ==========================================================

rows = []

common_keys = (
    set(td_lookup.keys())
    &
    set(tml_lookup.keys())
)

for key in sorted(common_keys):

    for td_name in sorted(
        td_lookup[key]
    ):

        for tml_name in sorted(
            tml_lookup[key]
        ):

            rows.append({

                "player_key":
                    key,

                "tennis_data_name":
                    td_name,

                "tml_name":
                    tml_name,

                "mapping_method":
                    "AUTO_KEY",

                "confidence":
                    1.0

            })
            
result = pd.DataFrame(rows)

# ==========================================================
# MANUAL PLAYER OVERRIDES
# ==========================================================

MANUAL_PLAYER_OVERRIDES = [

    {
        "player_key": "zhang_ze",
        "tennis_data_name": "Zhang Ze",
        "tml_name": "Ze Zhang"
    },

    {
        "player_key": "zhang_zhizhen",
        "tennis_data_name": "Zhang Zh.",
        "tml_name": "Zhizhen Zhang"
    },

    {
        "player_key": "srichaphan_p",
        "tennis_data_name": "Srichaphan N.",
        "tml_name": "Paradorn Srichaphan"
    },

    {
        "player_key": "wang_y",
        "tennis_data_name": "Wang Y. Jr.",
        "tml_name": "Yu Jr. Wang"
    }
]

override_df = pd.DataFrame(
    MANUAL_PLAYER_OVERRIDES
)

override_df["mapping_method"] = "MANUAL_OVERRIDE"
override_df["confidence"] = 1.0

result = pd.concat(
    [result, override_df],
    ignore_index=True
)



# ==========================================================
# CANONICAL PLAYER KEY PER TML PLAYER
# ==========================================================

key_priority = (
    result
    .groupby(
        ["tml_name", "player_key"]
    )
    .size()
    .reset_index(name="matches")
)

canonical_keys = (
    key_priority
    .sort_values(
        ["tml_name", "matches"],
        ascending=[True, False]
    )
    .drop_duplicates(
        subset=["tml_name"],
        keep="first"
    )
    [
        [
            "tml_name",
            "player_key"
        ]
    ]
    .rename(
        columns={
            "player_key":
            "canonical_player_key"
        }
    )
)

result = result.merge(
    canonical_keys,
    on="tml_name",
    how="left"
)

result["player_key"] = (
    result["canonical_player_key"]
)

result = result.drop(
    columns=[
        "canonical_player_key"
    ]
)

result = result.drop_duplicates(
    subset=[
        "player_key",
        "tennis_data_name",
        "tml_name"
    ]
)


result = result.sort_values(
    [
        "player_key",
        "tennis_data_name",
        "tml_name"
    ]
)

# ==========================================================
# MANUAL REJECTS
# ==========================================================

MANUAL_REJECTS = {

    ("Zhang Ze", "Zhizhen Zhang"),
    ("Zhang Ze.", "Zhizhen Zhang"),
    ("Zhang Zh.", "Ze Zhang"),

}

result = result[
    ~result.apply(
        lambda r: (
            r["tennis_data_name"],
            r["tml_name"]
        ) in MANUAL_REJECTS,
        axis=1
    )
]

# ==========================================================
# SAVE
# ==========================================================

Path(
    "data/parquet/v2"
).mkdir(
    parents=True,
    exist_ok=True
)

print()

print(
    "TD KEYS:",
    len(td_lookup)
)

print(
    "TML KEYS:",
    len(tml_lookup)
)

print(
    "COMMON KEYS:",
    len(common_keys)
)

duplicates = (
    result
    .groupby("tml_name")
    ["player_key"]
    .nunique()
)

print()
print(
    "TML PLAYERS WITH >1 KEY:",
    int((duplicates > 1).sum())
)

result.to_parquet(
    OUTPUT,
    index=False
)

print()
print("=" * 80)
print("PLAYER MAPPING V2")
print("=" * 80)

print()

print(
    "ROWS:",
    len(result)
)

print(
    "UNIQUE KEYS:",
    result["player_key"].nunique()
)

print()

print(
    result.head(50)
)

print()

print(
    "SAVED:",
    OUTPUT
)