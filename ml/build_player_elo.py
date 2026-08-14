import pandas as pd
import numpy as np
import time

MASTER_PATH = "data/parquet/master_matches.parquet"

INITIAL_ELO = 1500
BASE_ELO = 1500
HALF_LIFE_DAYS = 730
SURFACE_HALF_LIFE_DAYS = 3650

def apply_decay(
    elo,
    days_inactive
):

    if days_inactive <= 0:
        return elo

    decay_factor = np.exp(
        -days_inactive /
        HALF_LIFE_DAYS
    )

    return (
        BASE_ELO
        +
        (
            elo - BASE_ELO
        )
        * decay_factor
    )
    
def apply_surface_decay(
    elo,
    days_inactive
):

    if days_inactive <= 0:
        return elo

    decay_factor = np.exp(
        -days_inactive /
        SURFACE_HALF_LIFE_DAYS
    )

    return (
        BASE_ELO
        +
        (
            elo - BASE_ELO
        )
        * decay_factor
    )


def expected_score(
    elo_a,
    elo_b
):

    return (
        1
        /
        (
            1
            +
            10 ** (
                (elo_b - elo_a)
                / 400
            )
        )
    )

def get_k_factor(
    level
):

    if level == "G":
        return 40

    if level == "M":
        return 32

    if level == "500":
        return 28

    return 24

def get_days_inactive(
    player_id,
    current_date,
    last_match_date
):

    if player_id not in last_match_date:
        return -1

    
    return (
        current_date
        -
        last_match_date[player_id]
    ).days

def get_surface_days_inactive(
    player_id,
    surface,
    current_date,
    last_surface_match_date
):

    if (
        surface not in last_surface_match_date
        or
        player_id not in last_surface_match_date[surface]
    ):
        return -1

    return (
        current_date
        -
        last_surface_match_date[surface][player_id]
    ).days

print("Loading master dataset...")

master = pd.read_parquet(
    MASTER_PATH
)

master["match_date"] = pd.to_datetime(
    master["tourney_date"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

master = master.dropna(
    subset=["match_date"]
)

master = master.drop_duplicates(
    subset=[
        "tourney_date",
        "winner_id",
        "loser_id",
        "tourney_name"
    ]
)

master = master.sort_values(
    "match_date"
)

#master = master.head(100000)

overall_elo = {}

surface_elo = {
    "Hard": {},
    "Clay": {},
    "Grass": {},
    "Carpet": {}
}

last_match_date = {}
last_surface_match_date = {
    "Hard": {},
    "Clay": {},
    "Grass": {},
    "Carpet": {}
}

matches_played = {}

surface_matches_played = {
    "Hard": {},
    "Clay": {},
    "Grass": {},
    "Carpet": {}
}

rows = []

total_matches = len(master)

start_time = time.time()

for idx, (_, match) in enumerate(master.iterrows(), start=1):
    
    if idx % 25000 == 0:
        
        elapsed = time.time() - start_time
        
        rate = idx / elapsed
        
        remaining = total_matches - idx
        
        eta_minutes = (
        remaining / rate / 60
        )
        
        print()
        
        print(
        f"Processed {idx:,} / {total_matches:,} matches"
        )
        
        print(
        f"Rate: {rate:.1f} matches/sec"
        )
        
        print(
        f"ETA: {eta_minutes:.1f} minutes"
        )

    winner_id = match["winner_id"]
    loser_id = match["loser_id"]

    current_date = match["match_date"]

    surface = str(
        match["surface"]
    ).strip()

    winner_days = get_days_inactive(
        winner_id,
        current_date,
        last_match_date
    )

    loser_days = get_days_inactive(
        loser_id,
        current_date,
        last_match_date
    )
    
    winner_surface_days = -1
    loser_surface_days = -1

    if surface in surface_elo:

        winner_surface_days = (
            get_surface_days_inactive(
                winner_id,
                surface,
                current_date,
                last_surface_match_date
            )
        )

        loser_surface_days = (
            get_surface_days_inactive(
                loser_id,
                surface,
                current_date,
                last_surface_match_date
            )
        )
    
    
    winner_is_new_player = int(
        winner_days == -1
    )

    loser_is_new_player = int(
        loser_days == -1
    )

    winner_stored_elo = (
        overall_elo.get(
            winner_id,
            INITIAL_ELO
        )
    )

    loser_stored_elo = (
        overall_elo.get(
            loser_id,
            INITIAL_ELO
        )
    )

    winner_elo = apply_decay(
        winner_stored_elo,
        winner_days
    )

    loser_elo = apply_decay(
        loser_stored_elo,
        loser_days
    )

    winner_matches = matches_played.get(
        winner_id,
        0
    )

    loser_matches = matches_played.get(
        loser_id,
        0
    )

    winner_surface_elo = INITIAL_ELO
    loser_surface_elo = INITIAL_ELO

    winner_surface_stored_elo = INITIAL_ELO
    loser_surface_stored_elo = INITIAL_ELO

    winner_surface_matches = 0
    loser_surface_matches = 0

    if surface in surface_elo:

        winner_surface_stored_elo = (
            surface_elo[surface]
            .get(
                winner_id,
                INITIAL_ELO
            )
        )

        loser_surface_stored_elo = (
            surface_elo[surface]
            .get(
                loser_id,
                INITIAL_ELO
            )
        )
        
        winner_surface_elo = (
            apply_surface_decay(
                winner_surface_stored_elo,
                winner_surface_days
            )
        )

        loser_surface_elo = (
            apply_surface_decay(
                loser_surface_stored_elo,
                loser_surface_days
            )
        )

        winner_surface_matches = (
            surface_matches_played[surface]
            .get(
                winner_id,
                0
            )
        )

        loser_surface_matches = (
            surface_matches_played[surface]
            .get(
                loser_id,
                0
            )
        )
    
    level = str(
        match["tourney_level"]
    )

    k = get_k_factor(
        level
    )
    
    max_days = max(
        winner_days,
        loser_days
    )
    
    if max_days > 180:
    
        k *= 2
    
    elif max_days > 70:
    
        k *= 1.5

    rows.append({

        "match_date": current_date,

        "surface": surface,
        
        "k_factor": k,

        "winner_id": winner_id,
        "loser_id": loser_id,

        "winner_elo_before": winner_elo,
        "loser_elo_before": loser_elo,
        
        "winner_stored_elo":
            winner_stored_elo,

        "loser_stored_elo":
            loser_stored_elo,

        "winner_surface_elo_before":
            winner_surface_elo,

        "loser_surface_elo_before":
            loser_surface_elo,
        "winner_surface_stored_elo":
            winner_surface_stored_elo,

        "loser_surface_stored_elo":
            loser_surface_stored_elo,
            
        "winner_is_new_player":
            winner_is_new_player,

        "loser_is_new_player":
            loser_is_new_player,

        "winner_days_inactive":
            winner_days,

        "loser_days_inactive":
            loser_days,
        
        "winner_surface_days_inactive":
            winner_surface_days,

        "loser_surface_days_inactive":
            loser_surface_days,


        "winner_matches_played":
            winner_matches,

        "loser_matches_played":
            loser_matches,

        "winner_surface_matches_played":
            winner_surface_matches,

        "loser_surface_matches_played":
            loser_surface_matches

    })

    expected_winner = expected_score(
        winner_elo,
        loser_elo
    )
    
    overall_elo[winner_id] = (
        winner_stored_elo
        +
        k
        *
        (
            1
            -
            expected_winner
        )
    )

    overall_elo[loser_id] = (
        loser_stored_elo
        -
        k
        *
        (
            1
            -
            expected_winner
        )
    )
    
    if surface in surface_elo:

        expected_surface = expected_score(
            winner_surface_elo,
            loser_surface_elo
        )

        surface_elo[surface][winner_id] = (
            winner_surface_stored_elo
            +
            k
            *
            (
                1
                -
                expected_surface
            )
        )

        surface_elo[surface][loser_id] = (
            loser_surface_stored_elo
            -
            k
            *
            (
                1
                -
                expected_surface
            )
        )

        surface_matches_played[surface][winner_id] = (
            winner_surface_matches
            + 1
        )

        surface_matches_played[surface][loser_id] = (
            loser_surface_matches
            + 1
        )

    matches_played[winner_id] = (
        winner_matches + 1
    )

    matches_played[loser_id] = (
        loser_matches + 1
    )

    last_match_date[winner_id] = (
        current_date
    )

    last_match_date[loser_id] = (
        current_date
    )
    if surface in surface_elo:

        last_surface_match_date[surface][winner_id] = (
            current_date
        )

        last_surface_match_date[surface][loser_id] = (
            current_date
        )

df = pd.DataFrame(
    rows
)

print()
print("ROWS")

print(len(df))

print()

print("OVERALL ELO")

print(
    df[
        [
            "winner_elo_before",
            "loser_elo_before"
        ]
    ]
    .describe()
)

print()

print("SURFACE ELO")

print(
    df[
        [
            "winner_surface_elo_before",
            "loser_surface_elo_before"
        ]
    ]
    .describe()
)

all_elos = pd.Series(
    list(
        overall_elo.values()
    )
)

print()

print("FINAL ELO DISTRIBUTION")

print(
    all_elos.describe()
)
print()

print("TOP 20 ELO")

print(
    all_elos
    .sort_values(
        ascending=False
    )
    .head(20)
)

print()

print("BOTTOM 20 ELO")

print(
    all_elos
    .sort_values()
    .head(20)
)

elo_history = pd.DataFrame(
    rows
)

print()

print("NEW PLAYERS")

print(
    (
        elo_history["winner_is_new_player"]
        +
        elo_history["loser_is_new_player"]
    ).sum()
)

print()

print("AVG WINNER ELO")

print(
    round(
        elo_history[
            "winner_elo_before"
        ].mean(),
        2
    )
)

print()

print("AVG LOSER ELO")

print(
    round(
        elo_history[
            "loser_elo_before"
        ].mean(),
        2
    )
)

output_path = (
    "data/parquet/player_elo_history.parquet"
)


for surface_name in surface_elo:

    values = pd.Series(
        list(
            surface_elo[surface_name].values()
        )
    )

    print()
    print(surface_name)

    print(
        values.describe()
    )

print()

print(
    elo_history[
        "winner_surface_matches_played"
    ]
    .describe()
)

print()

print(
    elo_history[
        "loser_surface_matches_played"
    ]
    .describe()
)

elo_history.to_parquet(
    output_path,
    index=False
)

print()

print("MAX ELO")

print(
    round(
        elo_history[
            "winner_elo_before"
        ].max(),
        2
    )
)

print()

print("MAX SURFACE ELO")

print(
    round(
        elo_history[
            "winner_surface_elo_before"
        ].max(),
        2
    )
)

print()

print("MAX INACTIVITY DAYS")

print(
    max(
        elo_history[
            "winner_days_inactive"
        ].max(),

        elo_history[
            "loser_days_inactive"
        ].max()
    )
)

print()

print()

print("SURFACE COUNTS")

for s in surface_elo:

    print()

    print(s)

    print(
        len(
            surface_elo[s]
        )
    )
    
print()

print("SURFACE MATCH COUNTS")

print(
    master["surface"]
    .value_counts(
        dropna=False
    )
)



print("SAVED")

print(output_path)

print()

print("FINAL SHAPE")

print(
    elo_history.shape
)
print()

print("COLUMNS")

print(

elo_history.columns.tolist()

)