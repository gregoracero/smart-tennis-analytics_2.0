from functools import lru_cache
import numpy as np
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

from streamlit_app.config.data_paths import MATCH_HISTORY_PATH

# Backward-compatible name retained for the existing loader functions.
TENNIS_MATCHES_WITH_PLAYER_STATS = MATCH_HISTORY_PATH


# ==========================================================
# CONFIGURATION
# ==========================================================

PLAYER_COLUMNS = [
    "player_1_id",
    "player_1_name",
    "player_1_hand",
    "player_1_height",
    "player_1_ioc",
    "player_1_age",
    "player_1_rank",
    "player_1_rank_points",
    "player_2_id",
    "player_2_name",
    "player_2_hand",
    "player_2_height",
    "player_2_ioc",
    "player_2_age",
    "player_2_rank",
    "player_2_rank_points",
]

MATCH_COLUMNS = [
    "tourney_id",
    "tourney_name",
    "tourney_level",
    "match_num",
    "match_date",
    "tourney_date",
    "surface",
    "round",
    "best_of",
    "target_player_1_win",
]

ODDS_COLUMNS = [
    "player_1_b365_odds",
    "player_2_b365_odds",
    "player_1_ps_odds",
    "player_2_ps_odds",
    "player_1_max_odds",
    "player_2_max_odds",
    "player_1_avg_odds",
    "player_2_avg_odds",
    "player_1_bfe_odds",
    "player_2_bfe_odds",
]

PREMATCH_FEATURES = [
    "career_matches_before",
    "career_wins_before",
    "career_win_rate_before",
    "surface_matches_before",
    "surface_wins_before",
    "surface_win_rate_before",
    "elo_before",
    "surface_elo_before",
    "form_last_5_before",
    "form_last_10_before",
    "avg_minutes_last_5_before",
    "days_since_last_match",
    "stat_matches_before",
    "aces_per_service_game_before",
    "double_faults_per_service_game_before",
    "first_serve_in_pct_before",
    "first_serve_win_pct_before",
    "second_serve_win_pct_before",
    "service_points_won_pct_before",
    "break_points_saved_pct_before",
    "return_points_won_pct_before",
]


# ==========================================================
# INTERNAL HELPERS
# ==========================================================

def _validate_source_file():
    if not TENNIS_MATCHES_WITH_PLAYER_STATS.exists():
        raise FileNotFoundError(
            "No se encuentra el Parquet principal:\n"
            f"{TENNIS_MATCHES_WITH_PLAYER_STATS.resolve()}"
        )


def _available_columns():
    """
    Devuelve las columnas disponibles sin cargar todo el Parquet.
    """

    _validate_source_file()

    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(
        TENNIS_MATCHES_WITH_PLAYER_STATS
    )

    return set(
        parquet_file.schema_arrow.names
    )


def _first_existing_column(
    frame,
    column_names,
):
    """
    Combina varias columnas utilizando la primera disponible
    y no nula de cada fila.
    """

    result = pd.Series(
        np.nan,
        index=frame.index,
        dtype="float64",
    )

    for column in column_names:
        if column not in frame.columns:
            continue

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        result = result.fillna(values)

    return result


def _safe_divide(
    numerator,
    denominator,
):
    denominator = denominator.replace(
        0,
        np.nan,
    )

    return numerator / denominator


def _normalize_player_id(series):
    """
    Mantiene IDs numéricos cuando es posible.

    Los IDs ausentes permanecen como valores nulos.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    ).astype("Int64")


def _create_player_side(
    matches,
    side,
):
    """
    Genera una fila por partido para player_1 o player_2.
    """

    other_side = 2 if side == 1 else 1

    result = pd.DataFrame(
        index=matches.index
    )

    # ------------------------------------------------------
    # Match context
    # ------------------------------------------------------

    context_columns = [
        "tourney_id",
        "tourney_name",
        "tourney_level",
        "match_num",
        "match_date",
        "surface",
        "round",
        "best_of",
    ]

    for column in context_columns:
        if column in matches.columns:
            result[column] = matches[column]

    # ------------------------------------------------------
    # Player identity
    # ------------------------------------------------------

    result["player_id"] = (
        matches[f"player_{side}_id"]
        if f"player_{side}_id" in matches.columns
        else pd.NA
    )

    result["player_name"] = (
        matches[f"player_{side}_name"]
        if f"player_{side}_name" in matches.columns
        else pd.NA
    )

    for target_column, source_suffix in [
        ("hand", "hand"),
        ("height", "height"),
        ("ioc", "ioc"),
        ("age", "age"),
        ("rank", "rank"),
        ("rank_points", "rank_points"),
    ]:
        source_column = (
            f"player_{side}_{source_suffix}"
        )

        result[target_column] = (
            matches[source_column]
            if source_column in matches.columns
            else pd.NA
        )

    result["opponent_id"] = (
        matches[f"player_{other_side}_id"]
        if f"player_{other_side}_id"
        in matches.columns
        else pd.NA
    )

    result["opponent_name"] = (
        matches[f"player_{other_side}_name"]
        if f"player_{other_side}_name"
        in matches.columns
        else pd.NA
    )

    result["opponent_rank"] = (
        matches[f"player_{other_side}_rank"]
        if f"player_{other_side}_rank"
        in matches.columns
        else pd.NA
    )

    # ------------------------------------------------------
    # Result
    # ------------------------------------------------------

    target = pd.to_numeric(
        matches["target_player_1_win"],
        errors="coerce",
    )

    if side == 1:
        result["won"] = target
    else:
        result["won"] = 1 - target

    result["won"] = (
        result["won"]
        .astype("Int8")
    )

    result["result"] = np.where(
        result["won"].eq(1),
        "W",
        "L",
    )

    # ------------------------------------------------------
    # Prematch player features
    # ------------------------------------------------------

    for feature in PREMATCH_FEATURES:
        source_column = (
            f"player_{side}_{feature}"
        )

        if source_column in matches.columns:
            result[feature] = matches[
                source_column
            ]

    # ------------------------------------------------------
    # Odds
    # ------------------------------------------------------

    bookmaker_mapping = {
        "b365_odds": (
            f"player_{side}_b365_odds"
        ),
        "ps_odds": (
            f"player_{side}_ps_odds"
        ),
        "max_odds": (
            f"player_{side}_max_odds"
        ),
        "avg_odds": (
            f"player_{side}_avg_odds"
        ),
        "bfe_odds": (
            f"player_{side}_bfe_odds"
        ),
    }

    for target_column, source_column in (
        bookmaker_mapping.items()
    ):
        result[target_column] = (
            pd.to_numeric(
                matches[source_column],
                errors="coerce",
            )
            if source_column in matches.columns
            else np.nan
        )

    # Cuota representativa para estadísticas históricas.
    # Prioridad:
    # 1. Average odds
    # 2. Pinnacle/PS
    # 3. Bet365
    # 4. Betfair Exchange
    #
    # No se utiliza MaxW/MaxL como primera opción porque
    # puede producir una estimación demasiado optimista.
    result["odds"] = _first_existing_column(
        result,
        [
            "avg_odds",
            "ps_odds",
            "b365_odds",
            "bfe_odds",
        ],
    )

    result["has_odds"] = (
        result["odds"]
        .gt(1.0)
    )

    # ------------------------------------------------------
    # Market probabilities
    # ------------------------------------------------------

    opponent_odds = _first_existing_column(
        matches,
        [
            f"player_{other_side}_avg_odds",
            f"player_{other_side}_ps_odds",
            f"player_{other_side}_b365_odds",
            f"player_{other_side}_bfe_odds",
        ],
    )

    valid_odds = (
        result["odds"].gt(1.0)
        & opponent_odds.gt(1.0)
    )

    raw_player_probability = (
        1.0 / result["odds"]
    )

    raw_opponent_probability = (
        1.0 / opponent_odds
    )

    overround = (
        raw_player_probability
        + raw_opponent_probability
    )

    result["market_probability_raw"] = (
        raw_player_probability
        .where(valid_odds)
    )

    result["market_probability_no_vig"] = (
        (
            raw_player_probability
            / overround
        )
        .where(valid_odds)
    )

    result["market_overround"] = (
        (overround - 1.0)
        .where(valid_odds)
    )

    # ------------------------------------------------------
    # Historical flat-stake simulation
    # ------------------------------------------------------

    # Beneficio histórico con una unidad por partido:
    #
    # Victoria: cuota - 1
    # Derrota: -1
    #
    # Solo se calcula cuando existe una cuota válida.
    result["flat_stake_profit"] = np.where(
        result["has_odds"]
        & result["won"].eq(1),
        result["odds"] - 1.0,
        np.where(
            result["has_odds"],
            -1.0,
            np.nan,
        ),
    )

    result["flat_stake_return"] = np.where(
        result["has_odds"],
        result["flat_stake_profit"],
        np.nan,
    )

    result["player_side"] = side

    return result


# ==========================================================
# LOADERS
# ==========================================================

@lru_cache(maxsize=1)
def load_matches():
    """
    Carga únicamente las columnas necesarias.

    El resultado se mantiene en caché durante la ejecución
    del dashboard.
    """

    available = _available_columns()

    requested = (
        MATCH_COLUMNS
        + PLAYER_COLUMNS
        + ODDS_COLUMNS
        + [
            f"player_{side}_{feature}"
            for side in (1, 2)
            for feature in PREMATCH_FEATURES
        ]
    )

    selected_columns = [
        column
        for column in dict.fromkeys(
            requested
        )
        if column in available
    ]

    required = {
        "player_1_id",
        "player_1_name",
        "player_2_id",
        "player_2_name",
        "target_player_1_win",
    }

    missing = required - set(
        selected_columns
    )

    if missing:
        raise ValueError(
            "Faltan columnas obligatorias en "
            f"{TENNIS_MATCHES_WITH_PLAYER_STATS.name}: "
            f"{sorted(missing)}"
        )

    df = pd.read_parquet(
        TENNIS_MATCHES_WITH_PLAYER_STATS,
        columns=selected_columns,
        engine="pyarrow",
    )

    # ------------------------------------------------------
    # Date
    # ------------------------------------------------------

    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(
            df["match_date"],
            errors="coerce",
        )

    elif "tourney_date" in df.columns:
        df["match_date"] = pd.to_datetime(
            pd.to_numeric(
                df["tourney_date"],
                errors="coerce",
            )
            .astype("Int64")
            .astype(str),
            format="%Y%m%d",
            errors="coerce",
        )

    else:
        raise ValueError(
            "No existe match_date ni tourney_date "
            "en el Parquet."
        )

    df["season"] = (
        df["match_date"]
        .dt.year
        .astype("Int16")
    )

    return df


@lru_cache(maxsize=1)
def load_player_betting_detail():
    """
    Devuelve un registro por jugador y partido.
    """

    matches = load_matches()

    player_1 = _create_player_side(
        matches,
        side=1,
    )

    player_2 = _create_player_side(
        matches,
        side=2,
    )

    detail = pd.concat(
        [
            player_1,
            player_2,
        ],
        ignore_index=True,
    )

    detail["player_id"] = (
        _normalize_player_id(
            detail["player_id"]
        )
    )

    detail["opponent_id"] = (
        _normalize_player_id(
            detail["opponent_id"]
        )
    )

    detail["season"] = (
        pd.to_datetime(
            detail["match_date"],
            errors="coerce",
        )
        .dt.year
        .astype("Int16")
    )

    detail = (
        detail
        .dropna(
            subset=[
                "player_name",
                "match_date",
            ]
        )
        .sort_values(
            [
                "match_date",
                "tourney_id",
                "match_num",
            ],
            kind="mergesort",
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    return detail


@lru_cache(maxsize=1)
def load_player_betting():
    """
    Genera una fila resumen por jugador.
    """

    detail = load_player_betting_detail()

    summary = (
        detail
        .groupby(
            [
                "player_id",
                "player_name",
            ],
            dropna=False,
            observed=True,
        )
        .agg(
            matches=(
                "won",
                "size",
            ),
            wins=(
                "won",
                "sum",
            ),
            matches_with_odds=(
                "has_odds",
                "sum",
            ),
            total_flat_stake_profit=(
                "flat_stake_profit",
                "sum",
            ),
            average_odds=(
                "odds",
                "mean",
            ),
            average_market_probability=(
                "market_probability_no_vig",
                "mean",
            ),
            first_match_date=(
                "match_date",
                "min",
            ),
            last_match_date=(
                "match_date",
                "max",
            ),
            latest_rank=(
                "rank",
                "last",
            ),
            latest_rank_points=(
                "rank_points",
                "last",
            ),
            latest_elo=(
                "elo_before",
                "last",
            ),
            latest_surface_elo=(
                "surface_elo_before",
                "last",
            ),
        )
        .reset_index()
    )

    summary["losses"] = (
        summary["matches"]
        - summary["wins"]
    )

    summary["win_rate"] = _safe_divide(
        summary["wins"],
        summary["matches"],
    )

    summary["odds_coverage"] = (
        _safe_divide(
            summary["matches_with_odds"],
            summary["matches"],
        )
    )

    summary["flat_stake_roi"] = (
        _safe_divide(
            summary[
                "total_flat_stake_profit"
            ],
            summary[
                "matches_with_odds"
            ],
        )
    )

    summary["profit_per_match_with_odds"] = (
        summary["flat_stake_roi"]
    )

    return (
        summary
        .sort_values(
            [
                "matches",
                "player_name",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ==========================================================
# PLAYERS
# ==========================================================

def get_betting_players():
    df = load_player_betting()

    return (
        df[
            [
                "player_id",
                "player_name",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "player_name",
        )
        .reset_index(
            drop=True,
        )
    )


# ==========================================================
# SUMMARY
# ==========================================================

def get_player_betting_summary(
    player_id,
):
    df = load_player_betting()

    normalized_player_id = pd.to_numeric(
        pd.Series([player_id]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(normalized_player_id):
        return None

    result = df[
        df["player_id"].eq(
            int(normalized_player_id)
        )
    ]

    if result.empty:
        return None

    return result.iloc[0]


# ==========================================================
# DETAIL
# ==========================================================

def get_player_betting_detail(
    player_id,
):
    df = load_player_betting_detail()

    normalized_player_id = pd.to_numeric(
        pd.Series([player_id]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(normalized_player_id):
        return df.iloc[0:0].copy()

    return (
        df[
            df["player_id"].eq(
                int(normalized_player_id)
            )
        ]
        .copy()
        .reset_index(
            drop=True,
        )
    )


# ==========================================================
# FILTERS
# ==========================================================

def get_surfaces():
    df = load_player_betting_detail()

    values = sorted(
        df["surface"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return [
        "ALL",
        *values,
    ]


def get_years():
    df = load_player_betting_detail()

    seasons = sorted(
        df["season"]
        .dropna()
        .astype(int)
        .astype(str)
        .unique()
        .tolist(),
        reverse=True,
    )

    return [
        "ALL",
        *seasons,
    ]


def get_tourney_levels():
    df = load_player_betting_detail()

    if "tourney_level" not in df.columns:
        return ["ALL"]

    values = sorted(
        df["tourney_level"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return [
        "ALL",
        *values,
    ]


def get_rounds():
    df = load_player_betting_detail()

    if "round" not in df.columns:
        return ["ALL"]

    values = sorted(
        df["round"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return [
        "ALL",
        *values,
    ]


# ==========================================================
# FILTERED DETAIL
# ==========================================================

def get_filtered_player_betting_detail(
    player_id,
    surface="ALL",
    season="ALL",
    tourney_level="ALL",
    round_name="ALL",
):
    df = get_player_betting_detail(
        player_id
    )

    if surface != "ALL":
        df = df[
            df["surface"]
            .astype(str)
            .eq(str(surface))
        ]

    if season != "ALL":
        df = df[
            df["season"]
            .astype(str)
            .eq(str(season))
        ]

    if (
        tourney_level != "ALL"
        and "tourney_level" in df.columns
    ):
        df = df[
            df["tourney_level"]
            .astype(str)
            .eq(str(tourney_level))
        ]

    if (
        round_name != "ALL"
        and "round" in df.columns
    ):
        df = df[
            df["round"]
            .astype(str)
            .eq(str(round_name))
        ]

    return df.reset_index(
        drop=True
    )


# ==========================================================
# FILTERED AGGREGATION
# ==========================================================

def calculate_betting_metrics(
    detail,
):
    """
    Calcula las métricas del dashboard sobre cualquier
    subconjunto filtrado.
    """

    if detail is None or detail.empty:
        return {
            "matches": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": np.nan,
            "matches_with_odds": 0,
            "odds_coverage": np.nan,
            "average_odds": np.nan,
            "flat_stake_profit": 0.0,
            "flat_stake_roi": np.nan,
            "average_market_probability": np.nan,
        }

    matches = len(detail)

    wins = int(
        detail["won"]
        .fillna(0)
        .sum()
    )

    matches_with_odds = int(
        detail["has_odds"]
        .fillna(False)
        .sum()
    )

    flat_stake_profit = (
        detail["flat_stake_profit"]
        .sum(
            min_count=1
        )
    )

    if pd.isna(flat_stake_profit):
        flat_stake_profit = 0.0

    return {
        "matches": matches,
        "wins": wins,
        "losses": matches - wins,
        "win_rate": (
            wins / matches
            if matches
            else np.nan
        ),
        "matches_with_odds": (
            matches_with_odds
        ),
        "odds_coverage": (
            matches_with_odds / matches
            if matches
            else np.nan
        ),
        "average_odds": (
            detail["odds"].mean()
        ),
        "flat_stake_profit": float(
            flat_stake_profit
        ),
        "flat_stake_roi": (
            flat_stake_profit
            / matches_with_odds
            if matches_with_odds
            else np.nan
        ),
        "average_market_probability": (
            detail[
                "market_probability_no_vig"
            ]
            .mean()
        ),
    }


# ==========================================================
# CACHE CONTROL
# ==========================================================

def clear_betting_cache():
    """
    Utilizar después de regenerar el Parquet mientras
    el dashboard continúa ejecutándose.
    """

    load_matches.cache_clear()
    load_player_betting_detail.cache_clear()
    load_player_betting.cache_clear()