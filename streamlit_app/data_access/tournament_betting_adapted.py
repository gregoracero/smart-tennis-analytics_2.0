from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ==========================================================
# PATHS
# ==========================================================

from streamlit_app.config.data_paths import MATCH_HISTORY_PATH

# Backward-compatible name retained for the existing loader functions.
TENNIS_MATCHES_WITH_PLAYER_STATS = MATCH_HISTORY_PATH


# ==========================================================
# CONFIGURATION
# ==========================================================

CONTEXT_COLUMNS = [
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

PLAYER_COLUMNS = [
    "player_1_id",
    "player_1_name",
    "player_1_rank",
    "player_1_rank_points",
    "player_2_id",
    "player_2_name",
    "player_2_rank",
    "player_2_rank_points",
]


# ==========================================================
# INTERNAL HELPERS
# ==========================================================

def _validate_source():
    if not TENNIS_MATCHES_WITH_PLAYER_STATS.exists():
        raise FileNotFoundError(
            "No se encuentra el Parquet principal:\n"
            f"{TENNIS_MATCHES_WITH_PLAYER_STATS.resolve()}"
        )


@lru_cache(maxsize=1)
def _available_columns():
    """
    Consulta el esquema sin cargar el Parquet completo.
    """

    _validate_source()

    parquet_file = pq.ParquetFile(
        TENNIS_MATCHES_WITH_PLAYER_STATS
    )

    return set(
        parquet_file.schema_arrow.names
    )


def _safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _safe_divide(
    numerator,
    denominator,
):
    """
    División segura para series o valores escalares.
    """

    if isinstance(
        denominator,
        pd.Series,
    ):
        denominator = denominator.replace(
            0,
            np.nan,
        )

        return numerator / denominator

    if (
        denominator is None
        or pd.isna(denominator)
        or denominator == 0
    ):
        return np.nan

    return numerator / denominator


def _first_valid_pair(
    frame,
    candidate_pairs,
):
    """
    Selecciona la primera pareja válida de cuotas para cada
    partido.

    La pareja debe proceder del mismo proveedor.
    """

    player_1_odds = pd.Series(
        np.nan,
        index=frame.index,
        dtype="float64",
    )

    player_2_odds = pd.Series(
        np.nan,
        index=frame.index,
        dtype="float64",
    )

    bookmaker = pd.Series(
        pd.NA,
        index=frame.index,
        dtype="string",
    )

    for (
        provider_name,
        player_1_column,
        player_2_column,
    ) in candidate_pairs:

        if (
            player_1_column not in frame.columns
            or player_2_column not in frame.columns
        ):
            continue

        odds_1 = _safe_numeric(
            frame[player_1_column]
        )

        odds_2 = _safe_numeric(
            frame[player_2_column]
        )

        valid = (
            player_1_odds.isna()
            & odds_1.gt(1.0)
            & odds_2.gt(1.0)
        )

        player_1_odds = (
            player_1_odds.where(
                ~valid,
                odds_1,
            )
        )

        player_2_odds = (
            player_2_odds.where(
                ~valid,
                odds_2,
            )
        )

        bookmaker = bookmaker.where(
            ~valid,
            provider_name,
        )

    return (
        player_1_odds,
        player_2_odds,
        bookmaker,
    )


def _prepare_match_odds(
    matches,
):
    """
    Añade cuotas, probabilidades de mercado, favorito,
    ganador y resultados flat-stake.

    La prioridad utilizada es:
    1. Average
    2. Pinnacle / PS
    3. Bet365
    4. Betfair Exchange

    Max no se utiliza para la simulación principal porque
    puede producir una estimación histórica optimista.
    """

    matches = matches.copy()

    (
        matches["player_1_odds"],
        matches["player_2_odds"],
        matches["odds_source"],
    ) = _first_valid_pair(
        matches,
        [
            (
                "AVG",
                "player_1_avg_odds",
                "player_2_avg_odds",
            ),
            (
                "PS",
                "player_1_ps_odds",
                "player_2_ps_odds",
            ),
            (
                "B365",
                "player_1_b365_odds",
                "player_2_b365_odds",
            ),
            (
                "BFE",
                "player_1_bfe_odds",
                "player_2_bfe_odds",
            ),
        ],
    )

    matches["has_odds"] = (
        matches["player_1_odds"].gt(1.0)
        & matches["player_2_odds"].gt(1.0)
    )

    # ------------------------------------------------------
    # Market probabilities
    # ------------------------------------------------------

    raw_probability_1 = (
        1.0 / matches["player_1_odds"]
    )

    raw_probability_2 = (
        1.0 / matches["player_2_odds"]
    )

    probability_total = (
        raw_probability_1
        + raw_probability_2
    )

    matches[
        "player_1_market_probability"
    ] = (
        raw_probability_1
        / probability_total
    ).where(
        matches["has_odds"]
    )

    matches[
        "player_2_market_probability"
    ] = (
        raw_probability_2
        / probability_total
    ).where(
        matches["has_odds"]
    )

    matches["market_overround"] = (
        probability_total - 1.0
    ).where(
        matches["has_odds"]
    )

    # ------------------------------------------------------
    # Winner
    # ------------------------------------------------------

    player_1_won = _safe_numeric(
        matches["target_player_1_win"]
    ).eq(1)

    matches["winner_side"] = np.where(
        player_1_won,
        1,
        2,
    )

    matches["winner_id"] = np.where(
        player_1_won,
        matches.get(
            "player_1_id",
            pd.NA,
        ),
        matches.get(
            "player_2_id",
            pd.NA,
        ),
    )

    matches["winner_name"] = np.where(
        player_1_won,
        matches.get(
            "player_1_name",
            pd.NA,
        ),
        matches.get(
            "player_2_name",
            pd.NA,
        ),
    )

    matches["winner_odds"] = np.where(
        player_1_won,
        matches["player_1_odds"],
        matches["player_2_odds"],
    )

    # ------------------------------------------------------
    # Favourite and underdog
    # ------------------------------------------------------

    player_1_is_favorite = (
        matches["player_1_odds"]
        .lt(matches["player_2_odds"])
    )

    player_2_is_favorite = (
        matches["player_2_odds"]
        .lt(matches["player_1_odds"])
    )

    matches["favorite_side"] = pd.Series(
        np.select(
            [
                player_1_is_favorite,
                player_2_is_favorite,
            ],
            [
                1,
                2,
            ],
            default=0,
        ),
        index=matches.index,
        dtype="int8",
    )

    matches["favorite_odds"] = pd.Series(
        np.select(
            [
                matches["favorite_side"].eq(1),
                matches["favorite_side"].eq(2),
            ],
            [
                matches["player_1_odds"],
                matches["player_2_odds"],
            ],
            default=np.nan,
        ),
        index=matches.index,
        dtype="float64",
    )

    matches["underdog_odds"] = pd.Series(
        np.select(
            [
                matches["favorite_side"].eq(1),
                matches["favorite_side"].eq(2),
            ],
            [
                matches["player_2_odds"],
                matches["player_1_odds"],
            ],
            default=np.nan,
        ),
        index=matches.index,
        dtype="float64",
    )

    valid_favorite = (
        matches["has_odds"]
        .fillna(False)
        .astype(bool)
        & matches["favorite_side"].ne(0)
    )

    favorite_won_values = (
        matches["favorite_side"]
        .eq(matches["winner_side"])
    )

    matches["favorite_won"] = pd.Series(
        pd.NA,
        index=matches.index,
        dtype="boolean",
    )

    matches.loc[
        valid_favorite,
        "favorite_won",
    ] = favorite_won_values.loc[
        valid_favorite
    ].astype(bool)

    matches["underdog_won"] = pd.Series(
        pd.NA,
        index=matches.index,
        dtype="boolean",
    )

    matches.loc[
        matches["favorite_won"].eq(True),
        "underdog_won",
    ] = False

    matches.loc[
        matches["favorite_won"].eq(False),
        "underdog_won",
    ] = True


    # ------------------------------------------------------
    # Favourite and underdog result
    # ------------------------------------------------------

    valid_favorite = (
        matches["has_odds"]
        .fillna(False)
        .astype(bool)
        & matches["favorite_side"]
        .fillna(0)
        .ne(0)
    )

    favorite_result = (
        matches["favorite_side"]
        .eq(matches["winner_side"])
    )

    matches["favorite_won"] = pd.Series(
        pd.NA,
        index=matches.index,
        dtype="boolean",
    )

    matches.loc[
        valid_favorite,
        "favorite_won",
    ] = (
        favorite_result.loc[
            valid_favorite
        ]
        .fillna(False)
        .astype(bool)
    )


    matches["underdog_won"] = pd.Series(
        pd.NA,
        index=matches.index,
        dtype="boolean",
    )

    favorite_win_mask = (
        matches["favorite_won"]
        .fillna(False)
        .astype(bool)
    )

    favorite_loss_mask = (
        matches["favorite_won"]
        .fillna(True)
        .eq(False)
        .astype(bool)
    )

    # Limitar las máscaras a partidos elegibles.
    favorite_win_mask = (
        valid_favorite
        & favorite_win_mask
    )

    favorite_loss_mask = (
        valid_favorite
        & favorite_loss_mask
    )

    matches.loc[
        favorite_win_mask,
        "underdog_won",
    ] = False

    matches.loc[
        favorite_loss_mask,
        "underdog_won",
    ] = True


    # ------------------------------------------------------
    # Flat stake on favourite
    # ------------------------------------------------------

    matches[
        "favorite_flat_stake_profit"
    ] = pd.Series(
        np.nan,
        index=matches.index,
        dtype="float64",
    )

    matches.loc[
        favorite_win_mask,
        "favorite_flat_stake_profit",
    ] = (
        pd.to_numeric(
            matches.loc[
                favorite_win_mask,
                "favorite_odds",
            ],
            errors="coerce",
        )
        - 1.0
    )

    matches.loc[
        favorite_loss_mask,
        "favorite_flat_stake_profit",
    ] = -1.0


    # ------------------------------------------------------
    # Flat stake on underdog
    # ------------------------------------------------------

    underdog_win_mask = (
        matches["underdog_won"]
        .fillna(False)
        .astype(bool)
    )

    underdog_loss_mask = (
        matches["underdog_won"]
        .fillna(True)
        .eq(False)
        .astype(bool)
    )

    underdog_win_mask = (
        valid_favorite
        & underdog_win_mask
    )

    underdog_loss_mask = (
        valid_favorite
        & underdog_loss_mask
    )

    matches[
        "underdog_flat_stake_profit"
    ] = pd.Series(
        np.nan,
        index=matches.index,
        dtype="float64",
    )

    matches.loc[
        underdog_win_mask,
        "underdog_flat_stake_profit",
    ] = (
        pd.to_numeric(
            matches.loc[
                underdog_win_mask,
                "underdog_odds",
            ],
            errors="coerce",
        )
        - 1.0
    )

    matches.loc[
        underdog_loss_mask,
        "underdog_flat_stake_profit",
    ] = -1.0

    return matches


# ==========================================================
# BASE MATCH LOADER
# ==========================================================

@lru_cache(maxsize=1)
def load_tournament_matches():
    """
    Carga una fila por partido con las cuotas orientadas.

    Esta tabla sirve como detalle para filtros y gráficos.
    """

    available = _available_columns()

    requested = (
        CONTEXT_COLUMNS
        + PLAYER_COLUMNS
        + ODDS_COLUMNS
    )

    selected_columns = [
        column
        for column in dict.fromkeys(
            requested
        )
        if column in available
    ]

    required = {
        "tourney_name",
        "target_player_1_win",
    }

    missing = (
        required
        - set(selected_columns)
    )

    if missing:
        raise ValueError(
            "Faltan columnas obligatorias "
            "en el Parquet:\n"
            f"{sorted(missing)}"
        )

    matches = pd.read_parquet(
        TENNIS_MATCHES_WITH_PLAYER_STATS,
        columns=selected_columns,
        engine="pyarrow",
    )

    # ------------------------------------------------------
    # Tournament name compatibility
    # ------------------------------------------------------

    matches["tournament_name"] = (
        matches["tourney_name"]
        .astype("string")
    )

    # ------------------------------------------------------
    # Date
    # ------------------------------------------------------

    if "match_date" in matches.columns:
        matches["match_date"] = (
            pd.to_datetime(
                matches["match_date"],
                errors="coerce",
            )
        )

    elif "tourney_date" in matches.columns:
        matches["match_date"] = (
            pd.to_datetime(
                pd.to_numeric(
                    matches["tourney_date"],
                    errors="coerce",
                )
                .astype("Int64")
                .astype(str),
                format="%Y%m%d",
                errors="coerce",
            )
        )

    else:
        raise ValueError(
            "El Parquet no contiene "
            "match_date ni tourney_date."
        )

    matches["season"] = (
        matches["match_date"]
        .dt.year
        .astype("Int16")
    )

    matches = _prepare_match_odds(
        matches
    )

    sort_columns = [
        column
        for column in [
            "match_date",
            "tourney_id",
            "match_num",
        ]
        if column in matches.columns
    ]

    return (
        matches
        .dropna(
            subset=[
                "tournament_name",
                "match_date",
            ]
        )
        .sort_values(
            sort_columns,
            kind="mergesort",
            na_position="last",
        )
        .reset_index(
            drop=True,
        )
    )


# ==========================================================
# TOURNAMENT AGGREGATION
# ==========================================================

@lru_cache(maxsize=1)
def load_tournament_betting():
    """
    Genera estadísticas por torneo, temporada y superficie.

    Una misma sede o nombre de torneo puede aparecer en
    distintas temporadas y superficies.
    """

    matches = (
        load_tournament_matches()
    )

    if matches.empty:
        return pd.DataFrame()

    grouping_columns = [
        "tournament_name",
        "season",
        "surface",
        "tourney_level",
    ]

    grouping_columns = [
        column
        for column in grouping_columns
        if column in matches.columns
    ]

    tournament_stats = (
        matches
        .groupby(
            grouping_columns,
            dropna=False,
            observed=True,
        )
        .agg(
            matches=(
                "target_player_1_win",
                "size",
            ),
            matches_with_odds=(
                "has_odds",
                "sum",
            ),
            first_match_date=(
                "match_date",
                "min",
            ),
            last_match_date=(
                "match_date",
                "max",
            ),
            average_player_1_odds=(
                "player_1_odds",
                "mean",
            ),
            average_player_2_odds=(
                "player_2_odds",
                "mean",
            ),
            average_winner_odds=(
                "winner_odds",
                "mean",
            ),
            average_favorite_odds=(
                "favorite_odds",
                "mean",
            ),
            average_underdog_odds=(
                "underdog_odds",
                "mean",
            ),
            favorite_wins=(
                "favorite_won",
                "sum",
            ),
            valid_favorite_matches=(
                "favorite_won",
                "count",
            ),
            favorite_flat_stake_profit=(
                "favorite_flat_stake_profit",
                "sum",
            ),
            underdog_flat_stake_profit=(
                "underdog_flat_stake_profit",
                "sum",
            ),
            average_market_overround=(
                "market_overround",
                "mean",
            ),
        )
        .reset_index()
    )

    tournament_stats[
        "matches_without_odds"
    ] = (
        tournament_stats["matches"]
        - tournament_stats[
            "matches_with_odds"
        ]
    )

    tournament_stats[
        "odds_coverage"
    ] = _safe_divide(
        tournament_stats[
            "matches_with_odds"
        ],
        tournament_stats["matches"],
    )

    tournament_stats[
        "favorite_losses"
    ] = (
        tournament_stats[
            "valid_favorite_matches"
        ]
        - tournament_stats[
            "favorite_wins"
        ]
    )

    tournament_stats[
        "favorite_win_rate"
    ] = _safe_divide(
        tournament_stats[
            "favorite_wins"
        ],
        tournament_stats[
            "valid_favorite_matches"
        ],
    )

    tournament_stats[
        "underdog_wins"
    ] = (
        tournament_stats[
            "valid_favorite_matches"
        ]
        - tournament_stats[
            "favorite_wins"
        ]
    )

    tournament_stats[
        "underdog_win_rate"
    ] = (
        1.0
        - tournament_stats[
            "favorite_win_rate"
        ]
    )

    tournament_stats[
        "favorite_flat_stake_roi"
    ] = _safe_divide(
        tournament_stats[
            "favorite_flat_stake_profit"
        ],
        tournament_stats[
            "valid_favorite_matches"
        ],
    )

    tournament_stats[
        "underdog_flat_stake_roi"
    ] = _safe_divide(
        tournament_stats[
            "underdog_flat_stake_profit"
        ],
        tournament_stats[
            "valid_favorite_matches"
        ],
    )

    return (
        tournament_stats
        .sort_values(
            [
                "tournament_name",
                "season",
            ],
            ascending=[
                True,
                False,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True,
        )
    )


# ==========================================================
# TOURNAMENT LOOKUP
# ==========================================================

def get_tournaments():
    df = load_tournament_betting()

    if df.empty:
        return []

    return sorted(
        df["tournament_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


# ==========================================================
# TOURNAMENT DETAIL
# ==========================================================

def get_tournament_detail(
    tournament_name,
    season=None,
    surface="ALL",
    tourney_level="ALL",
):
    """
    Devuelve estadísticas agregadas del torneo.

    Compatible con la llamada original:

        get_tournament_detail("Wimbledon")

    También admite filtros opcionales.
    """

    df = load_tournament_betting()

    result = df[
        df["tournament_name"]
        .astype(str)
        .eq(str(tournament_name))
    ].copy()

    if season not in [
        None,
        "ALL",
        "CAREER",
    ]:
        result = result[
            result["season"]
            .astype(str)
            .eq(str(season))
        ]

    if (
        surface not in [
            None,
            "ALL",
        ]
        and "surface" in result.columns
    ):
        result = result[
            result["surface"]
            .astype(str)
            .eq(str(surface))
        ]

    if (
        tourney_level not in [
            None,
            "ALL",
        ]
        and "tourney_level"
        in result.columns
    ):
        result = result[
            result["tourney_level"]
            .astype(str)
            .eq(str(tourney_level))
        ]

    return (
        result
        .sort_values(
            [
                "season",
                "surface",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True,
        )
    )


# ==========================================================
# TOURNAMENT MATCH DETAIL
# ==========================================================

def get_tournament_matches(
    tournament_name,
    season=None,
    surface="ALL",
    with_odds_only=False,
):
    """
    Devuelve los partidos individuales del torneo.

    Es útil para tablas, gráficos de cuotas y evolución
    histórica.
    """

    matches = (
        load_tournament_matches()
    )

    result = matches[
        matches["tournament_name"]
        .astype(str)
        .eq(str(tournament_name))
    ].copy()

    if season not in [
        None,
        "ALL",
        "CAREER",
    ]:
        result = result[
            result["season"]
            .astype(str)
            .eq(str(season))
        ]

    if surface not in [
        None,
        "ALL",
    ]:
        result = result[
            result["surface"]
            .astype(str)
            .eq(str(surface))
        ]

    if with_odds_only:
        result = result[
            result["has_odds"]
            .eq(True)
        ]

    return (
        result
        .sort_values(
            "match_date",
            ascending=False,
            kind="mergesort",
        )
        .reset_index(
            drop=True,
        )
    )


# ==========================================================
# CAREER TOURNAMENT SUMMARY
# ==========================================================

def get_tournament_summary(
    tournament_name,
):
    """
    Agrega todas las ediciones disponibles de un torneo.
    """

    detail = get_tournament_detail(
        tournament_name
    )

    if detail.empty:
        return None

    matches = int(
        detail["matches"].sum()
    )

    matches_with_odds = int(
        detail["matches_with_odds"].sum()
    )

    valid_favorite_matches = int(
        detail[
            "valid_favorite_matches"
        ].sum()
    )

    favorite_wins = int(
        detail["favorite_wins"].sum()
    )

    favorite_profit = (
        detail[
            "favorite_flat_stake_profit"
        ]
        .sum(
            min_count=1
        )
    )

    underdog_profit = (
        detail[
            "underdog_flat_stake_profit"
        ]
        .sum(
            min_count=1
        )
    )

    return pd.Series(
        {
            "tournament_name": (
                tournament_name
            ),
            "editions": int(
                detail["season"]
                .nunique()
            ),
            "first_season": (
                detail["season"].min()
            ),
            "last_season": (
                detail["season"].max()
            ),
            "matches": matches,
            "matches_with_odds": (
                matches_with_odds
            ),
            "odds_coverage": (
                matches_with_odds / matches
                if matches
                else np.nan
            ),
            "valid_favorite_matches": (
                valid_favorite_matches
            ),
            "favorite_wins": (
                favorite_wins
            ),
            "favorite_win_rate": (
                favorite_wins
                / valid_favorite_matches
                if valid_favorite_matches
                else np.nan
            ),
            "favorite_flat_stake_profit": (
                float(favorite_profit)
                if pd.notna(favorite_profit)
                else 0.0
            ),
            "favorite_flat_stake_roi": (
                favorite_profit
                / valid_favorite_matches
                if (
                    valid_favorite_matches
                    and pd.notna(
                        favorite_profit
                    )
                )
                else np.nan
            ),
            "underdog_flat_stake_profit": (
                float(underdog_profit)
                if pd.notna(underdog_profit)
                else 0.0
            ),
            "underdog_flat_stake_roi": (
                underdog_profit
                / valid_favorite_matches
                if (
                    valid_favorite_matches
                    and pd.notna(
                        underdog_profit
                    )
                )
                else np.nan
            ),
            "average_winner_odds": (
                np.average(
                    detail[
                        "average_winner_odds"
                    ].dropna(),
                    weights=detail.loc[
                        detail[
                            "average_winner_odds"
                        ].notna(),
                        "matches_with_odds",
                    ],
                )
                if (
                    detail[
                        "average_winner_odds"
                    ].notna().any()
                    and detail.loc[
                        detail[
                            "average_winner_odds"
                        ].notna(),
                        "matches_with_odds",
                    ].sum() > 0
                )
                else np.nan
            ),
        }
    )


# ==========================================================
# FILTER VALUES
# ==========================================================

def get_available_tournament_seasons(
    tournament_name=None,
):
    df = load_tournament_betting()

    if tournament_name is not None:
        df = df[
            df["tournament_name"]
            .astype(str)
            .eq(str(tournament_name))
        ]

    values = sorted(
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
        *values,
    ]


def get_available_tournament_surfaces(
    tournament_name=None,
):
    df = load_tournament_betting()

    if tournament_name is not None:
        df = df[
            df["tournament_name"]
            .astype(str)
            .eq(str(tournament_name))
        ]

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


def get_available_tournament_levels():
    df = load_tournament_betting()

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


# ==========================================================
# CACHE CONTROL
# ==========================================================

def clear_tournament_betting_cache():
    """
    Vacía la caché después de regenerar el Parquet.
    """

    _available_columns.cache_clear()
    load_tournament_matches.cache_clear()
    load_tournament_betting.cache_clear()