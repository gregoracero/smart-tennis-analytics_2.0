#!/usr/bin/env python3
"""Acceso adaptado a torneos para Sackmann + TennisMyLife v2.

La identidad analítica de una edición es:

    base_tournament_id + competition_type

Ejemplo:

    2026-1210|ATP
    2026-1210|ATP_QUALIFYING

La API pública conserva los nombres usados por las páginas Streamlit. En
`get_tournaments()`, la columna `tournament_id` contiene la clave compuesta para
que los selectores existentes sigan funcionando sin mezclar cuadros distintos.
La columna `base_tournament_id` conserva el `tourney_id` original.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ==========================================================
# PATHS
# ==========================================================

from streamlit_app.config.data_paths import MATCH_HISTORY_PATH

# Backward-compatible name retained for public diagnostics and loaders.
TENNIS_MATCHES_WITH_PLAYER_STATS = MATCH_HISTORY_PATH


# ==========================================================
# CONFIGURATION
# ==========================================================

TOURNAMENT_KEY_SEPARATOR = "|"
TOURNAMENT_KEY_COLUMNS = [
    "base_tournament_id",
    "competition_type",
]

CONTEXT_COLUMNS = [
    "tourney_id",
    "tourney_name",
    "tourney_date",
    "tourney_level",
    "tourney_level_original",
    "competition_type",
    "source_match_num",
    "draw_size",
    "match_num",
    "match_date",
    "surface",
    "round",
    "best_of",
    "minutes",
    "score",
    "target_player_1_win",
    "target_player_1_wins_first_set",
    "target_player_2_wins_first_set",
    "target_player_1_wins_any_set",
    "target_player_2_wins_any_set",
    "target_valid_first_set",
    "target_valid_completed_score",
    "target_total_games",
    "target_over_18_5",
    "target_over_19_5",
    "target_over_20_5",
    "target_over_21_5",
    "target_over_22_5",
    "target_over_23_5",
    "target_over_24_5",
    "target_over_25_5",
    "source_origin",
    "eligible_for_model",
    "model_exclusion_reason",
]

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

PREMATCH_COLUMNS = [
    "player_1_elo_before",
    "player_2_elo_before",
    "player_1_surface_elo_before",
    "player_2_surface_elo_before",
    "player_1_form_last_5_before",
    "player_2_form_last_5_before",
    "player_1_form_last_10_before",
    "player_2_form_last_10_before",
    "player_1_days_since_last_match",
    "player_2_days_since_last_match",
    "player_1_career_matches_before",
    "player_2_career_matches_before",
    "player_1_career_win_rate_before",
    "player_2_career_win_rate_before",
    "player_1_surface_win_rate_before",
    "player_2_surface_win_rate_before",
]

MATCH_STAT_COLUMNS = [
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
]

ODDS_COLUMNS = [
    "odds_matched",
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


# ==========================================================
# INTERNAL HELPERS
# ==========================================================


def _validate_source() -> None:
    if not TENNIS_MATCHES_WITH_PLAYER_STATS.exists():
        raise FileNotFoundError(
            "No se encuentra el Parquet principal:\n"
            f"{TENNIS_MATCHES_WITH_PLAYER_STATS}"
        )


@lru_cache(maxsize=1)
def _available_columns() -> set[str]:
    """Consulta el esquema del Parquet sin cargar sus datos."""
    _validate_source()
    parquet_file = pq.ParquetFile(TENNIS_MATCHES_WITH_PLAYER_STATS)
    return set(parquet_file.schema_arrow.names)


def _safe_numeric(value: Any) -> Any:
    return pd.to_numeric(value, errors="coerce")


def _numeric_column(
    frame: pd.DataFrame,
    column: str,
    default: float = np.nan,
) -> pd.Series:
    """Devuelve siempre una Series numérica alineada con el DataFrame."""
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")

    return pd.to_numeric(frame[column], errors="coerce")


def _safe_divide(numerator: Any, denominator: Any) -> Any:
    numerator = _safe_numeric(numerator)
    denominator = _safe_numeric(denominator).replace(0, np.nan)
    return numerator / denominator


def _normalize_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    return text or None


def _normalize_competition_type(value: Any) -> str:
    text = _normalize_text(value)
    return text.upper() if text else "UNKNOWN"


def _make_tournament_key(
    base_tournament_id: Any,
    competition_type: Any,
) -> str | None:
    base_id = _normalize_text(base_tournament_id)
    if base_id is None:
        return None

    competition = _normalize_competition_type(competition_type)
    return f"{base_id}{TOURNAMENT_KEY_SEPARATOR}{competition}"


def _split_tournament_reference(
    tournament_id: Any,
    competition_type: Any = None,
) -> tuple[str | None, str | None, str | None]:
    """Interpreta una clave compuesta o un tourney_id histórico.

    Devuelve:
        tournament_key, base_tournament_id, competition_type
    """
    value = _normalize_text(tournament_id)
    if value is None:
        return None, None, None

    if TOURNAMENT_KEY_SEPARATOR in value:
        base_id, embedded_competition = value.rsplit(
            TOURNAMENT_KEY_SEPARATOR,
            1,
        )
        base_id = _normalize_text(base_id)
        competition = _normalize_competition_type(embedded_competition)
        return _make_tournament_key(base_id, competition), base_id, competition

    base_id = value
    if competition_type is None:
        return None, base_id, None

    competition = _normalize_competition_type(competition_type)
    return _make_tournament_key(base_id, competition), base_id, competition


def _validate_unique_tournament_rows(
    frame: pd.DataFrame,
    frame_name: str,
) -> None:
    if frame.empty:
        return

    required = {"tournament_key"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{frame_name} no contiene la clave requerida: {sorted(missing)}"
        )

    duplicate_mask = frame.duplicated(
        subset=["tournament_key"],
        keep=False,
    )

    if duplicate_mask.any():
        sample_columns = [
            column
            for column in [
                "tournament_key",
                "base_tournament_id",
                "competition_type",
                "tournament_name",
                "season",
                "surface",
            ]
            if column in frame.columns
        ]
        sample = (
            frame.loc[duplicate_mask, sample_columns]
            .drop_duplicates()
            .head(20)
        )
        raise ValueError(
            f"{frame_name} contiene "
            f"{int(duplicate_mask.sum()):,} filas con tournament_key duplicada:\n"
            f"{sample.to_string(index=False)}"
        )


def _first_valid_odds_pair(
    frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Selecciona la primera pareja válida del mismo proveedor."""
    player_1_odds = pd.Series(np.nan, index=frame.index, dtype="float64")
    player_2_odds = pd.Series(np.nan, index=frame.index, dtype="float64")
    odds_source = pd.Series(pd.NA, index=frame.index, dtype="string")

    candidate_pairs = [
        ("AVG", "player_1_avg_odds", "player_2_avg_odds"),
        ("PS", "player_1_ps_odds", "player_2_ps_odds"),
        ("B365", "player_1_b365_odds", "player_2_b365_odds"),
        ("BFE", "player_1_bfe_odds", "player_2_bfe_odds"),
    ]

    for source, player_1_column, player_2_column in candidate_pairs:
        if (
            player_1_column not in frame.columns
            or player_2_column not in frame.columns
        ):
            continue

        candidate_1 = _numeric_column(frame, player_1_column)
        candidate_2 = _numeric_column(frame, player_2_column)
        valid = (
            player_1_odds.isna()
            & candidate_1.gt(1.0)
            & candidate_2.gt(1.0)
        )

        player_1_odds = player_1_odds.where(~valid, candidate_1)
        player_2_odds = player_2_odds.where(~valid, candidate_2)
        odds_source = odds_source.where(~valid, source)

    return player_1_odds, player_2_odds, odds_source


def _build_named_aggregation(
    matches: pd.DataFrame,
) -> dict[str, tuple[str, str]]:
    """Construye agregados solo para columnas disponibles."""
    candidates = {
        "matches": ("target_player_1_win", "size"),
        "first_match_date": ("match_date", "min"),
        "last_match_date": ("match_date", "max"),
        "average_match_minutes": ("minutes", "mean"),
        "total_match_minutes": ("minutes", "sum"),
        "average_player_1_rank": ("player_1_rank", "mean"),
        "average_player_2_rank": ("player_2_rank", "mean"),
        "average_winner_rank": ("winner_rank", "mean"),
        "average_loser_rank": ("loser_rank", "mean"),
        "higher_ranked_winner_count": ("winner_was_higher_ranked", "sum"),
        "ranked_comparisons": ("winner_was_higher_ranked", "count"),
        "average_winner_elo": ("winner_elo_before", "mean"),
        "average_loser_elo": ("loser_elo_before", "mean"),
        "average_elo_difference": ("elo_difference", "mean"),
        "total_aces": ("total_aces", "sum"),
        "total_double_faults": ("total_double_faults", "sum"),
        "total_service_points": ("total_service_points", "sum"),
        "total_break_points_faced": ("total_break_points_faced", "sum"),
        "matches_with_odds": ("has_odds", "sum"),
        "favorite_wins": ("favorite_won", "sum"),
        "valid_favorite_matches": ("favorite_won", "count"),
        "average_market_overround": ("market_overround", "mean"),
        "tml_matches": ("is_tml", "sum"),
        "eligible_matches": ("eligible_for_model", "sum"),
    }

    return {
        output: definition
        for output, definition in candidates.items()
        if definition[0] in matches.columns
    }


# ==========================================================
# MATCH DERIVATIONS
# ==========================================================


def _add_match_derived_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Añade identidad de torneo, ganador, cuotas y métricas derivadas."""
    matches = matches.copy()

    matches["base_tournament_id"] = (
        matches["tourney_id"].astype("string").str.strip()
    )
    matches["competition_type"] = (
        matches.get(
            "competition_type",
            pd.Series("UNKNOWN", index=matches.index, dtype="string"),
        )
        .astype("string")
        .fillna("UNKNOWN")
        .str.strip()
        .str.upper()
    )
    matches["tournament_key"] = (
        matches["base_tournament_id"]
        + TOURNAMENT_KEY_SEPARATOR
        + matches["competition_type"]
    )

    # Compatibilidad: tournament_id pasa a ser la clave compuesta.
    matches["tournament_id"] = matches["tournament_key"]
    matches["tournament_name"] = matches["tourney_name"].astype("string")
    matches["tournament_level"] = (
        matches["tourney_level"].astype("string")
        if "tourney_level" in matches.columns
        else pd.Series(pd.NA, index=matches.index, dtype="string")
    )
    matches["season"] = matches["match_date"].dt.year.astype("Int16")
    matches["source_origin"] = (
        matches.get(
            "source_origin",
            pd.Series("UNKNOWN", index=matches.index, dtype="string"),
        )
        .astype("string")
        .fillna("UNKNOWN")
        .str.upper()
    )
    matches["is_tml"] = matches["source_origin"].eq("TML")

    if "eligible_for_model" not in matches.columns:
        matches["eligible_for_model"] = True
    matches["eligible_for_model"] = (
        matches["eligible_for_model"].fillna(False).astype(bool)
    )

    player_1_won = _numeric_column(matches, "target_player_1_win").eq(1)
    matches["winner_side"] = np.where(player_1_won, 1, 2)
    matches["loser_side"] = np.where(player_1_won, 2, 1)

    winner_loser_mapping = [
        ("id", "winner_id", "loser_id"),
        ("name", "winner_name", "loser_name"),
        ("rank", "winner_rank", "loser_rank"),
        ("rank_points", "winner_rank_points", "loser_rank_points"),
        ("elo_before", "winner_elo_before", "loser_elo_before"),
        (
            "surface_elo_before",
            "winner_surface_elo_before",
            "loser_surface_elo_before",
        ),
    ]

    for suffix, winner_output, loser_output in winner_loser_mapping:
        player_1_column = f"player_1_{suffix}"
        player_2_column = f"player_2_{suffix}"

        if (
            player_1_column not in matches.columns
            or player_2_column not in matches.columns
        ):
            continue

        matches[winner_output] = np.where(
            player_1_won,
            matches[player_1_column],
            matches[player_2_column],
        )
        matches[loser_output] = np.where(
            player_1_won,
            matches[player_2_column],
            matches[player_1_column],
        )

    player_1_rank = _numeric_column(matches, "player_1_rank")
    player_2_rank = _numeric_column(matches, "player_2_rank")
    winner_rank = _numeric_column(matches, "winner_rank")
    loser_rank = _numeric_column(matches, "loser_rank")

    matches["ranking_difference"] = player_1_rank - player_2_rank
    matches["winner_was_higher_ranked"] = (
        winner_rank.lt(loser_rank).where(winner_rank.notna() & loser_rank.notna())
    )

    if {
        "player_1_elo_before",
        "player_2_elo_before",
    }.issubset(matches.columns):
        matches["elo_difference"] = (
            _numeric_column(matches, "player_1_elo_before")
            - _numeric_column(matches, "player_2_elo_before")
        )

    if {
        "player_1_surface_elo_before",
        "player_2_surface_elo_before",
    }.issubset(matches.columns):
        matches["surface_elo_difference"] = (
            _numeric_column(matches, "player_1_surface_elo_before")
            - _numeric_column(matches, "player_2_surface_elo_before")
        )

    (
        matches["player_1_odds"],
        matches["player_2_odds"],
        matches["odds_source"],
    ) = _first_valid_odds_pair(matches)

    matches["has_odds"] = (
        matches["player_1_odds"].gt(1.0)
        & matches["player_2_odds"].gt(1.0)
    )

    raw_probability_1 = 1.0 / matches["player_1_odds"]
    raw_probability_2 = 1.0 / matches["player_2_odds"]
    total_probability = raw_probability_1 + raw_probability_2

    matches["player_1_market_probability"] = (
        raw_probability_1 / total_probability
    ).where(matches["has_odds"])
    matches["player_2_market_probability"] = (
        raw_probability_2 / total_probability
    ).where(matches["has_odds"])
    matches["market_overround"] = (
        total_probability - 1.0
    ).where(matches["has_odds"])

    matches["favorite_side"] = np.where(
        matches["player_1_odds"] < matches["player_2_odds"],
        1,
        np.where(
            matches["player_2_odds"] < matches["player_1_odds"],
            2,
            0,
        ),
    )
    matches["favorite_won"] = (
        matches["favorite_side"]
        .eq(matches["winner_side"])
        .where(matches["has_odds"] & matches["favorite_side"].ne(0))
    )

    matches["total_aces"] = (
        _numeric_column(matches, "w_ace", 0).fillna(0)
        + _numeric_column(matches, "l_ace", 0).fillna(0)
    )
    matches["total_double_faults"] = (
        _numeric_column(matches, "w_df", 0).fillna(0)
        + _numeric_column(matches, "l_df", 0).fillna(0)
    )
    matches["total_service_points"] = (
        _numeric_column(matches, "w_svpt", 0).fillna(0)
        + _numeric_column(matches, "l_svpt", 0).fillna(0)
    )
    matches["total_break_points_faced"] = (
        _numeric_column(matches, "w_bpFaced", 0).fillna(0)
        + _numeric_column(matches, "l_bpFaced", 0).fillna(0)
    )

    return matches


# ==========================================================
# MASTER MATCHES
# ==========================================================

@lru_cache(maxsize=1)
def load_master_matches() -> pd.DataFrame:
    """Carga los partidos necesarios para las páginas de torneos."""
    available = _available_columns()
    requested = (
        CONTEXT_COLUMNS
        + PLAYER_COLUMNS
        + PREMATCH_COLUMNS
        + MATCH_STAT_COLUMNS
        + ODDS_COLUMNS
    )
    selected_columns = [
        column
        for column in dict.fromkeys(requested)
        if column in available
    ]

    required = {
        "tourney_id",
        "tourney_name",
        "target_player_1_win",
        "player_1_id",
        "player_2_id",
    }
    missing = required - set(selected_columns)
    if missing:
        raise ValueError(
            "Faltan columnas obligatorias en el Parquet:\n"
            f"{sorted(missing)}"
        )

    matches = pd.read_parquet(
        TENNIS_MATCHES_WITH_PLAYER_STATS,
        columns=selected_columns,
        engine="pyarrow",
    )

    if "match_date" in matches.columns:
        matches["match_date"] = pd.to_datetime(
            matches["match_date"],
            errors="coerce",
        )
    elif "tourney_date" in matches.columns:
        matches["match_date"] = pd.to_datetime(
            pd.to_numeric(matches["tourney_date"], errors="coerce")
            .astype("Int64")
            .astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
    else:
        raise ValueError("El Parquet no contiene match_date ni tourney_date.")

    matches = _add_match_derived_columns(matches)
    matches = _ensure_market_target_columns(matches)

    return (
        matches
        .dropna(
            subset=[
                "base_tournament_id",
                "tournament_key",
                "tournament_name",
                "match_date",
            ]
        )
        .sort_values(
            [
                "match_date",
                "tournament_key",
                "match_num",
            ],
            kind="mergesort",
            na_position="last",
        )
        .reset_index(drop=True)
    )


# ==========================================================
# TOURNAMENT STATISTICS
# ==========================================================

def _canonical_value_by_tournament_key(
    matches: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return one deterministic dominant value per analytical tournament key.

    Source integrations can occasionally assign a minority metadata value to a
    match that otherwise belongs to the same ``tournament_key``. The canonical
    value is the most frequent non-empty value. Ties are resolved
    deterministically by normalized lexical order. Match rows are not removed.
    """
    if column not in matches.columns:
        return pd.Series(pd.NA, index=matches.index, dtype="string")

    normalized = matches[column].astype("string").str.strip()

    def dominant(values: pd.Series) -> Any:
        valid = values.dropna()
        valid = valid.loc[valid.ne("")]
        if valid.empty:
            return pd.NA
        counts = valid.value_counts(dropna=True)
        maximum = int(counts.max())
        candidates = sorted(
            counts.loc[counts.eq(maximum)].index.astype(str).tolist(),
            key=lambda value: (value.casefold(), value),
        )
        return candidates[0]

    canonical_by_key = (
        pd.DataFrame({
            "tournament_key": matches["tournament_key"].astype("string"),
            "value": normalized,
        })
        .groupby("tournament_key", dropna=False, observed=True)["value"]
        .agg(dominant)
    )
    return (
        matches["tournament_key"]
        .astype("string")
        .map(canonical_by_key)
        .astype("string")
    )


def _canonicalize_tournament_metadata(matches: pd.DataFrame) -> pd.DataFrame:
    """Harmonize metadata that must be unique within ``tournament_key``.

    Identity remains ``base_tournament_id + competition_type``. This function
    only resolves inconsistent descriptive metadata and never merges different
    competition types or drops matches.
    """
    result = matches.copy()
    for column in (
        "tournament_name",
        "surface",
        "tournament_level",
    ):
        if column in result.columns:
            result[column] = _canonical_value_by_tournament_key(result, column)
    return result


@lru_cache(maxsize=1)
def load_tournament_stats() -> pd.DataFrame:
    """Calcula una fila por tourney_id + competition_type."""
    matches = load_master_matches()
    if matches.empty:
        return pd.DataFrame()

    matches = _canonicalize_tournament_metadata(matches)

    grouping_columns = [
        "tournament_key",
        "base_tournament_id",
        "competition_type",
        "tournament_name",
        "season",
        "surface",
        "tournament_level",
    ]
    grouping_columns = [
        column for column in grouping_columns if column in matches.columns
    ]

    named_aggregation = _build_named_aggregation(matches)
    stats = (
        matches
        .groupby(
            grouping_columns,
            dropna=False,
            observed=True,
        )
        .agg(**named_aggregation)
        .reset_index()
    )

    # Cuenta ambos lados usando la misma clave compuesta.
    player_1 = matches[
        [
            "tournament_key",
            "base_tournament_id",
            "competition_type",
            "player_1_id",
        ]
    ].rename(columns={"player_1_id": "player_id"})

    player_2 = matches[
        [
            "tournament_key",
            "base_tournament_id",
            "competition_type",
            "player_2_id",
        ]
    ].rename(columns={"player_2_id": "player_id"})

    unique_players = (
        pd.concat([player_1, player_2], ignore_index=True)
        .dropna(subset=["player_id"])
        .groupby(
            [
                "tournament_key",
                "base_tournament_id",
                "competition_type",
            ],
            as_index=False,
            dropna=False,
            observed=True,
        )
        .agg(unique_players=("player_id", "nunique"))
    )

    _validate_unique_tournament_rows(stats, "stats")
    _validate_unique_tournament_rows(unique_players, "unique_players")

    stats = stats.merge(
        unique_players,
        on=[
            "tournament_key",
            "base_tournament_id",
            "competition_type",
        ],
        how="left",
        validate="one_to_one",
    )

    # Compatibilidad con páginas existentes: tournament_id es la clave única.
    stats["tournament_id"] = stats["tournament_key"]
    stats["tourney_id"] = stats["base_tournament_id"]

    stats["matches_without_odds"] = (
        _numeric_column(stats, "matches", 0)
        - _numeric_column(stats, "matches_with_odds", 0)
    )
    stats["odds_coverage"] = _safe_divide(
        stats["matches_with_odds"],
        stats["matches"],
    )
    stats["higher_ranked_winner_rate"] = _safe_divide(
        stats["higher_ranked_winner_count"],
        stats["ranked_comparisons"],
    )
    stats["favorite_losses"] = (
        _numeric_column(stats, "valid_favorite_matches", 0)
        - _numeric_column(stats, "favorite_wins", 0)
    )
    stats["favorite_win_rate"] = _safe_divide(
        stats["favorite_wins"],
        stats["valid_favorite_matches"],
    )
    stats["upset_count"] = (
        _numeric_column(stats, "ranked_comparisons", 0)
        - _numeric_column(stats, "higher_ranked_winner_count", 0)
    )
    stats["upset_rate"] = 1.0 - stats["higher_ranked_winner_rate"]
    stats["aces_per_match"] = _safe_divide(
        stats["total_aces"],
        stats["matches"],
    )
    stats["double_faults_per_match"] = _safe_divide(
        stats["total_double_faults"],
        stats["matches"],
    )
    stats["service_points_per_match"] = _safe_divide(
        stats["total_service_points"],
        stats["matches"],
    )
    stats["break_points_faced_per_match"] = _safe_divide(
        stats["total_break_points_faced"],
        stats["matches"],
    )

    stats["tournament_label"] = (
        stats["tournament_name"].astype("string")
        + " | "
        + stats["competition_type"].astype("string")
        + " | "
        + stats["season"].astype("Int64").astype("string")
    )

    _validate_unique_tournament_rows(stats, "final stats")

    return (
        stats
        .sort_values(
            [
                "season",
                "tournament_name",
                "competition_type",
            ],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


# ==========================================================
# TOURNAMENT LIST
# ==========================================================


def get_tournaments() -> pd.DataFrame:
    """Devuelve una fila por edición analítica única.

    `tournament_id` contiene la clave compuesta para compatibilidad con los
    selectores existentes. `base_tournament_id` contiene el tourney_id original.
    """
    df = load_tournament_stats()
    if df.empty:
        return df.copy()

    output_columns = [
        column
        for column in [
            "tournament_id",
            "tournament_key",
            "base_tournament_id",
            "tourney_id",
            "tournament_name",
            "tournament_label",
            "competition_type",
            "season",
            "surface",
            "tournament_level",
            "matches",
            "unique_players",
            "first_match_date",
            "last_match_date",
        ]
        if column in df.columns
    ]

    result = df[output_columns].copy()
    _validate_unique_tournament_rows(result, "get_tournaments")

    return (
        result
        .sort_values(
            [
                "tournament_name",
                "season",
                "competition_type",
            ],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


# ==========================================================
# TOURNAMENT SELECTION HELPERS
# ==========================================================


def _filter_tournament_reference(
    frame: pd.DataFrame,
    tournament_id: Any,
    competition_type: Any = None,
) -> pd.Series:
    tournament_key, base_id, competition = _split_tournament_reference(
        tournament_id,
        competition_type,
    )

    if tournament_key is not None:
        return frame["tournament_key"].astype(str).eq(tournament_key)

    if base_id is None:
        return pd.Series(False, index=frame.index, dtype=bool)

    mask = frame["base_tournament_id"].astype(str).eq(base_id)
    if competition is not None:
        mask &= frame["competition_type"].astype(str).eq(competition)

    return mask


def get_tournament_matches(
    tournament_id: Any,
    competition_type: Any = None,
) -> pd.DataFrame:
    """Devuelve los partidos de una edición.

    Se recomienda pasar `tournament_id` desde `get_tournaments()`, que ya es una
    clave compuesta. El argumento `competition_type` mantiene compatibilidad con
    llamadas manuales basadas en el tourney_id original.
    """
    df = load_master_matches()
    mask = _filter_tournament_reference(df, tournament_id, competition_type)

    return (
        df.loc[mask]
        .copy()
        .sort_values(
            ["match_date", "match_num"],
            ascending=[True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def get_tournament_stats(
    tournament_id: Any,
    competition_type: Any = None,
) -> pd.DataFrame:
    df = load_tournament_stats()
    mask = _filter_tournament_reference(df, tournament_id, competition_type)
    return df.loc[mask].copy().reset_index(drop=True)


# ==========================================================
# CANONICAL TOURNAMENT NAME ANALYTICS
# ==========================================================
def _normalize_tournament_name(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    import re
    import unicodedata
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


TOURNAMENT_NAME_ALIASES = {
    "genoa nuova valletta": "genoa",
    "genoa challenger": "genoa",
    "genoa": "genoa",
    "road to the rolex shanghai masters": "shanghai",
    "road to rolex shanghai masters": "shanghai",
    "shanghai challenger": "shanghai",
    "shanghai": "shanghai",
    "istanbul challenger ted open": "istanbul",
    "istanbul ted open": "istanbul",
    "istanbul challenger": "istanbul",
    "istanbul": "istanbul",
}


def _canonical_tournament_name(value: Any) -> str:
    normalized = _normalize_tournament_name(value)
    if not normalized:
        return ""
    if normalized in TOURNAMENT_NAME_ALIASES:
        return TOURNAMENT_NAME_ALIASES[normalized]
    for suffix in (
        " atp challenger", " challenger tour", " challenger",
        " qualifying", " qualification", " qual", " atp", " wta",
    ):
        if normalized.endswith(suffix):
            candidate = normalized[:-len(suffix)].strip()
            if candidate:
                normalized = candidate
                break
    return TOURNAMENT_NAME_ALIASES.get(normalized, normalized)


def _parse_score_market_targets(score: Any) -> dict[str, Any]:
    """Parse completed tennis score tokens into direct-market outcomes.

    Walkovers and scores without parseable sets are invalid. Retirements and
    defaults retain total-games information but are not considered completed
    scores for the any-set target. First-set targets remain valid whenever the
    first parseable set is complete.
    """
    import re

    empty = {
        "valid_first_set": False,
        "valid_completed_score": False,
        "player_1_wins_first_set": np.nan,
        "player_2_wins_first_set": np.nan,
        "player_1_wins_any_set": np.nan,
        "player_2_wins_any_set": np.nan,
        "total_games": np.nan,
        "played_sets": np.nan,
    }
    if score is None or pd.isna(score):
        return empty
    text = str(score).upper().strip()
    if not text or text in {"NAN", "NONE", "<NA>", "W/O", "WO", "WALKOVER"}:
        return empty
    retired = any(token in text for token in ("RET", "RETIRED", "DEF", "DEFAULT", "ABD"))
    sets: list[tuple[int, int]] = []
    for token in text.split():
        match = re.match(r"^(\d+)-(\d+)(?:\([^)]*\))?$", token)
        if match is not None:
            sets.append((int(match.group(1)), int(match.group(2))))
    if not sets:
        return empty
    first_left, first_right = sets[0]
    first_complete = (
        max(first_left, first_right) >= 6
        and abs(first_left - first_right) >= 2
    ) or {first_left, first_right} == {6, 7}
    player_1_set_wins = sum(left > right for left, right in sets)
    player_2_set_wins = sum(right > left for left, right in sets)
    total_games = float(sum(left + right for left, right in sets))
    completed = bool(not retired and player_1_set_wins != player_2_set_wins)
    return {
        "valid_first_set": bool(first_complete),
        "valid_completed_score": completed,
        "player_1_wins_first_set": float(first_left > first_right) if first_complete else np.nan,
        "player_2_wins_first_set": float(first_right > first_left) if first_complete else np.nan,
        "player_1_wins_any_set": float(player_1_set_wins >= 1) if completed else np.nan,
        "player_2_wins_any_set": float(player_2_set_wins >= 1) if completed else np.nan,
        "total_games": total_games,
        "played_sets": float(len(sets)),
    }


def _ensure_market_target_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Use dataset targets when available and derive missing values from score."""
    result = matches.copy()
    parsed = result.get("score", pd.Series(pd.NA, index=result.index)).map(
        _parse_score_market_targets
    )
    parsed_frame = pd.DataFrame(parsed.tolist(), index=result.index)
    mapping = {
        "target_valid_first_set": "valid_first_set",
        "target_valid_completed_score": "valid_completed_score",
        "target_player_1_wins_first_set": "player_1_wins_first_set",
        "target_player_2_wins_first_set": "player_2_wins_first_set",
        "target_player_1_wins_any_set": "player_1_wins_any_set",
        "target_player_2_wins_any_set": "player_2_wins_any_set",
        "target_total_games": "total_games",
    }
    for target, parsed_column in mapping.items():
        derived = pd.to_numeric(parsed_frame[parsed_column], errors="coerce")
        if target in result:
            existing = pd.to_numeric(result[target], errors="coerce")
            result[target] = existing.where(existing.notna(), derived)
        else:
            result[target] = derived
    for line in (18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5):
        target = f"target_over_{str(line).replace('.', '_')}"
        total = pd.to_numeric(result["target_total_games"], errors="coerce")
        derived = total.gt(line).where(total.notna()).astype("Float64")
        if target in result:
            existing = pd.to_numeric(result[target], errors="coerce")
            result[target] = existing.where(existing.notna(), derived)
        else:
            result[target] = derived
    return result


def _score_total_games(score: Any) -> float:
    """Parse completed set scores; tiebreak points are not counted as games."""
    import re
    if score is None or pd.isna(score):
        return np.nan
    text = str(score).upper().strip()
    if not text or any(token in text for token in ("W/O", "WALKOVER")) or text == "WO":
        return np.nan
    total = 0
    parsed = 0
    for token in text.split():
        match = re.match(r"^(\d+)-(\d+)(?:\([^)]*\))?$", token)
        if match is None:
            continue
        total += int(match.group(1)) + int(match.group(2))
        parsed += 1
    return float(total) if parsed else np.nan


def analyze_tournament_games_threshold(
    tournament_name: Any,
    threshold: float,
    competition_type: Any = None,
    surface: Any = None,
    as_of_date: Any = None,
    start_year: int | None = None,
    end_year: int | None = None,
    evidence_limit: int = 20,
) -> dict[str, Any]:
    """Deterministic historical count of matches above a total-games threshold."""
    target_key = _canonical_tournament_name(tournament_name)
    if not target_key:
        return {"available": False, "reason": "invalid_tournament_name"}
    matches = load_master_matches().copy()
    if matches.empty or "score" not in matches.columns:
        return {"available": False, "reason": "score_column_unavailable"}
    keys = matches["tournament_name"].map(_canonical_tournament_name)
    matches = matches.loc[keys.eq(target_key)].copy()
    if competition_type not in (None, "", "UNKNOWN"):
        competition = _normalize_competition_type(competition_type)
        matches = matches.loc[matches["competition_type"].astype(str).eq(competition)].copy()
    if surface not in (None, "", "ALL", "UNKNOWN", "Unknown") and "surface" in matches.columns:
        matches = matches.loc[matches["surface"].astype(str).str.casefold().eq(str(surface).casefold())].copy()
    reference = pd.to_datetime(as_of_date, errors="coerce")
    if pd.notna(reference):
        matches = matches.loc[matches["match_date"].lt(reference.normalize())].copy()
    years = pd.to_datetime(matches.get("match_date"), errors="coerce").dt.year
    if start_year is not None:
        matches = matches.loc[years.ge(int(start_year))].copy()
        years = pd.to_datetime(matches.get("match_date"), errors="coerce").dt.year
    if end_year is not None:
        matches = matches.loc[years.le(int(end_year))].copy()
    if matches.empty:
        return {
            "available": False, "reason": "no_tournament_matches",
            "tournament_name": str(tournament_name), "canonical_tournament_key": target_key,
        }
    matches["total_games"] = matches["score"].map(_score_total_games)
    score_status = matches["score"].astype("string").fillna("").str.upper()
    walkovers = int(score_status.str.contains(r"W/O|WALKOVER|^WO$", regex=True).sum())
    scored = matches.dropna(subset=["total_games"]).copy()
    qualifying = scored.loc[pd.to_numeric(scored["total_games"], errors="coerce").gt(float(threshold))].copy()
    names = sorted(matches["tournament_name"].dropna().astype(str).unique().tolist())
    seasons = pd.to_datetime(scored["match_date"], errors="coerce").dt.year.dropna()
    evidence_columns = [
        column for column in (
            "match_date", "season", "tournament_name", "competition_type",
            "surface", "round", "player_1_name", "player_2_name",
            "score", "total_games",
        ) if column in qualifying.columns
    ]
    evidence = (
        qualifying.sort_values("match_date", ascending=False, kind="mergesort")
        .loc[:, evidence_columns].head(max(int(evidence_limit), 1))
        .to_dict(orient="records")
    )
    return {
        "available": True,
        "method": "canonical_tournament_score_parser",
        "tournament_name": str(tournament_name),
        "canonical_tournament_key": target_key,
        "matched_historical_names": names,
        "threshold": float(threshold),
        "operator": "greater_than",
        "equivalent_minimum_integer_games": int(np.floor(float(threshold)) + 1),
        "total_matches": int(len(matches)),
        "score_covered_matches": int(len(scored)),
        "qualifying_matches": int(len(qualifying)),
        "observed_rate": float(len(qualifying) / len(scored)) if len(scored) else np.nan,
        "excluded_matches": int(len(matches) - len(scored)),
        "excluded_walkovers": walkovers,
        "excluded_unparsed_scores": int(len(matches) - len(scored) - walkovers),
        "first_season": int(seasons.min()) if not seasons.empty else None,
        "last_season": int(seasons.max()) if not seasons.empty else None,
        "competition_type": str(competition_type) if competition_type is not None else None,
        "surface": str(surface) if surface is not None else None,
        "as_of_date": str(reference) if pd.notna(reference) else None,
        "evidence": evidence,
    }


def _historical_market_filter(
    tournament_name: Any,
    competition_type: Any = None,
    surface: Any = None,
    as_of_date: Any = None,
) -> tuple[pd.DataFrame, str]:
    target_key = _canonical_tournament_name(tournament_name)
    if not target_key:
        return pd.DataFrame(), ""
    matches = _ensure_market_target_columns(load_master_matches().copy())
    matches = matches.loc[
        matches["tournament_name"].map(_canonical_tournament_name).eq(target_key)
    ].copy()
    if competition_type not in (None, "", "UNKNOWN"):
        competition = _normalize_competition_type(competition_type)
        matches = matches.loc[
            matches["competition_type"].astype(str).eq(competition)
        ].copy()
    if surface not in (None, "", "ALL", "UNKNOWN", "Unknown"):
        matches = matches.loc[
            matches["surface"].astype(str).str.casefold().eq(str(surface).casefold())
        ].copy()
    reference = pd.to_datetime(as_of_date, errors="coerce")
    if pd.notna(reference):
        matches = matches.loc[
            matches["match_date"].lt(reference.normalize())
        ].copy()
    return matches, target_key


def analyze_tournament_first_set_history(
    tournament_name: Any,
    competition_type: Any = None,
    surface: Any = None,
    as_of_date: Any = None,
    evidence_limit: int = 20,
) -> dict[str, Any]:
    """Historical first-set winner and comeback rates for comparable matches."""
    matches, target_key = _historical_market_filter(
        tournament_name, competition_type, surface, as_of_date
    )
    if matches.empty:
        return {"available": False, "reason": "no_tournament_matches"}
    required = {
        "target_valid_first_set",
        "target_player_1_wins_first_set",
        "target_player_1_win",
    }
    missing = sorted(required - set(matches.columns))
    if missing:
        return {
            "available": False,
            "reason": "first_set_targets_unavailable",
            "missing_columns": missing,
        }
    valid_source = matches.get(
        "target_valid_first_set",
        pd.Series(False, index=matches.index, dtype=bool),
    )
    valid = pd.to_numeric(valid_source, errors="coerce").fillna(0).eq(1)
    scored = matches.loc[valid].copy()
    if scored.empty:
        return {"available": False, "reason": "no_valid_first_set_scores"}
    p1_first = pd.to_numeric(
        scored["target_player_1_wins_first_set"], errors="coerce"
    )
    p1_match = pd.to_numeric(scored["target_player_1_win"], errors="coerce")
    first_set_winner_won_match = p1_first.eq(p1_match)
    comeback = ~first_set_winner_won_match
    evidence_columns = [column for column in (
        "match_date", "season", "tournament_name", "competition_type", "surface",
        "round", "player_1_name", "player_2_name", "score",
        "target_player_1_wins_first_set", "target_player_1_win",
    ) if column in scored]
    evidence = (
        scored.assign(comeback_after_losing_first_set=comeback.to_numpy())
        .sort_values("match_date", ascending=False, kind="mergesort")
        .loc[:, [*evidence_columns, "comeback_after_losing_first_set"]]
        .head(max(int(evidence_limit), 1))
        .to_dict(orient="records")
    )
    return {
        "available": True,
        "method": "historical_first_set_targets",
        "canonical_tournament_key": target_key,
        "matches": int(len(scored)),
        "first_set_winner_won_match": int(first_set_winner_won_match.sum()),
        "first_set_winner_match_win_rate": float(first_set_winner_won_match.mean()),
        "comebacks_after_losing_first_set": int(comeback.sum()),
        "comeback_rate": float(comeback.mean()),
        "evidence": evidence,
    }


def analyze_tournament_any_set_history(
    tournament_name: Any,
    competition_type: Any = None,
    surface: Any = None,
    as_of_date: Any = None,
    evidence_limit: int = 20,
) -> dict[str, Any]:
    """Historical frequency with which both players won at least one set."""
    matches, target_key = _historical_market_filter(
        tournament_name, competition_type, surface, as_of_date
    )
    if matches.empty:
        return {"available": False, "reason": "no_tournament_matches"}
    required = {
        "target_valid_completed_score",
        "target_player_1_wins_any_set",
        "target_player_2_wins_any_set",
    }
    missing = sorted(required - set(matches.columns))
    if missing:
        return {
            "available": False,
            "reason": "any_set_targets_unavailable",
            "missing_columns": missing,
        }
    valid_source = matches.get(
        "target_valid_completed_score",
        pd.Series(False, index=matches.index, dtype=bool),
    )
    valid = pd.to_numeric(valid_source, errors="coerce").fillna(0).eq(1)
    scored = matches.loc[valid].copy()
    if scored.empty:
        return {"available": False, "reason": "no_valid_completed_scores"}
    p1_any = pd.to_numeric(
        scored["target_player_1_wins_any_set"], errors="coerce"
    ).eq(1)
    p2_any = pd.to_numeric(
        scored["target_player_2_wins_any_set"], errors="coerce"
    ).eq(1)
    both = p1_any & p2_any
    evidence_columns = [column for column in (
        "match_date", "season", "tournament_name", "competition_type", "surface",
        "round", "best_of", "player_1_name", "player_2_name", "score",
        "target_player_1_wins_any_set", "target_player_2_wins_any_set",
    ) if column in scored]
    evidence = (
        scored.assign(both_players_won_a_set=both.to_numpy())
        .sort_values("match_date", ascending=False, kind="mergesort")
        .loc[:, [*evidence_columns, "both_players_won_a_set"]]
        .head(max(int(evidence_limit), 1))
        .to_dict(orient="records")
    )
    return {
        "available": True,
        "method": "historical_any_set_targets",
        "canonical_tournament_key": target_key,
        "matches": int(len(scored)),
        "both_players_won_a_set": int(both.sum()),
        "both_players_won_a_set_rate": float(both.mean()),
        "straight_sets_matches": int((~both).sum()),
        "straight_sets_rate": float((~both).mean()),
        "evidence": evidence,
    }


# ==========================================================
# TOURNAMENT EDITIONS BY NAME
# ==========================================================


def get_tournament_editions(
    tournament_name: Any,
    competition_type: Any = None,
) -> pd.DataFrame:
    """Devuelve todas las ediciones disponibles de un nombre de torneo."""
    df = load_tournament_stats()
    target_key = _canonical_tournament_name(tournament_name)
    mask = df["tournament_name"].map(_canonical_tournament_name).eq(target_key)

    if competition_type is not None:
        competition = _normalize_competition_type(competition_type)
        mask &= df["competition_type"].astype(str).eq(competition)

    return (
        df.loc[mask]
        .copy()
        .sort_values(
            ["season", "competition_type"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


# ==========================================================
# TOURNAMENT PLAYERS
# ==========================================================


def get_tournament_players(
    tournament_id: Any,
    competition_type: Any = None,
) -> pd.DataFrame:
    matches = get_tournament_matches(tournament_id, competition_type)

    if matches.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "player_name",
                "matches",
                "wins",
                "losses",
                "win_rate",
            ]
        )

    player_1 = matches[
        ["player_1_id", "player_1_name", "target_player_1_win"]
    ].rename(
        columns={
            "player_1_id": "player_id",
            "player_1_name": "player_name",
            "target_player_1_win": "won",
        }
    )

    player_2 = matches[
        ["player_2_id", "player_2_name", "target_player_1_win"]
    ].rename(
        columns={
            "player_2_id": "player_id",
            "player_2_name": "player_name",
            "target_player_1_win": "player_1_won",
        }
    )
    player_2["won"] = 1 - _safe_numeric(player_2["player_1_won"])
    player_2 = player_2.drop(columns=["player_1_won"])

    players = pd.concat([player_1, player_2], ignore_index=True)
    result = (
        players
        .dropna(subset=["player_name"])
        .groupby(
            ["player_id", "player_name"],
            dropna=False,
            observed=True,
        )
        .agg(
            matches=("won", "size"),
            wins=("won", "sum"),
        )
        .reset_index()
    )
    result["losses"] = result["matches"] - result["wins"]
    result["win_rate"] = _safe_divide(result["wins"], result["matches"])

    return (
        result
        .sort_values(
            ["wins", "matches", "player_name"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


# ==========================================================
# ROUND BREAKDOWN
# ==========================================================


def get_tournament_round_breakdown(
    tournament_id: Any,
    competition_type: Any = None,
) -> pd.DataFrame:
    matches = get_tournament_matches(tournament_id, competition_type)

    if matches.empty or "round" not in matches.columns:
        return pd.DataFrame()

    aggregation_candidates = {
        "matches": ("target_player_1_win", "size"),
        "average_match_minutes": ("minutes", "mean"),
        "average_winner_rank": ("winner_rank", "mean"),
        "average_loser_rank": ("loser_rank", "mean"),
        "favorite_wins": ("favorite_won", "sum"),
        "valid_favorite_matches": ("favorite_won", "count"),
        "total_aces": ("total_aces", "sum"),
        "total_double_faults": ("total_double_faults", "sum"),
    }
    aggregation = {
        output: definition
        for output, definition in aggregation_candidates.items()
        if definition[0] in matches.columns
    }

    result = (
        matches
        .groupby("round", dropna=False, observed=True)
        .agg(**aggregation)
        .reset_index()
    )

    if {
        "favorite_wins",
        "valid_favorite_matches",
    }.issubset(result.columns):
        result["favorite_win_rate"] = _safe_divide(
            result["favorite_wins"],
            result["valid_favorite_matches"],
        )

    if {"total_aces", "matches"}.issubset(result.columns):
        result["aces_per_match"] = _safe_divide(
            result["total_aces"],
            result["matches"],
        )

    return result


# ==========================================================
# AVAILABLE FILTERS
# ==========================================================


def get_available_tournament_seasons() -> list[str]:
    df = load_tournament_stats()
    if df.empty or "season" not in df.columns:
        return []

    return sorted(
        df["season"]
        .dropna()
        .astype(int)
        .astype(str)
        .unique()
        .tolist(),
        reverse=True,
    )


def get_available_tournament_surfaces() -> list[str]:
    df = load_tournament_stats()
    if df.empty or "surface" not in df.columns:
        return []

    return sorted(
        df["surface"].dropna().astype(str).unique().tolist()
    )


def get_available_tournament_levels() -> list[str]:
    df = load_tournament_stats()
    if df.empty or "tournament_level" not in df.columns:
        return []

    return sorted(
        df["tournament_level"].dropna().astype(str).unique().tolist()
    )


def get_available_competition_types() -> list[str]:
    df = load_tournament_stats()
    if df.empty or "competition_type" not in df.columns:
        return []

    return sorted(
        df["competition_type"].dropna().astype(str).unique().tolist()
    )


# ==========================================================
# VALIDATION
# ==========================================================


def validate_tournament_data() -> dict[str, Any]:
    """Ejecuta controles rápidos para diagnóstico de la página."""
    matches = load_master_matches()
    stats = load_tournament_stats()

    duplicate_match_keys = 0
    match_key_columns = [
        column
        for column in [
            "base_tournament_id",
            "competition_type",
            "match_num",
        ]
        if column in matches.columns
    ]
    if len(match_key_columns) == 3:
        duplicate_match_keys = int(matches.duplicated(match_key_columns).sum())

    duplicate_tournament_keys = int(
        stats.duplicated(["tournament_key"]).sum()
    ) if "tournament_key" in stats.columns else -1

    return {
        "source": str(TENNIS_MATCHES_WITH_PLAYER_STATS),
        "match_rows": int(len(matches)),
        "tournament_rows": int(len(stats)),
        "duplicate_match_keys": duplicate_match_keys,
        "duplicate_tournament_keys": duplicate_tournament_keys,
        "competition_types": get_available_competition_types(),
        "first_match_date": matches["match_date"].min(),
        "last_match_date": matches["match_date"].max(),
    }


# ==========================================================
# CACHE CONTROL
# ==========================================================


def clear_tournament_stats_cache() -> None:
    """Vacía la caché después de regenerar el Parquet."""
    _available_columns.cache_clear()
    load_master_matches.cache_clear()
    load_tournament_stats.cache_clear()
