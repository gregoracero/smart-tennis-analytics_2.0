#!/usr/bin/env python3
"""Acceso eficiente a jugadores, estado actual e historico de partidos.

Fuentes canonicas
-----------------
- player_current_state.sackmann_v5.parquet
- player_pair_current_state.sackmann_v5.parquet
- tennis_matches_with_player_stats.sackmann_v5.parquet

Las rutas se centralizan en ``streamlit_app.config.data_paths``. La API
conserva las funciones utilizadas por Streamlit y evita cualquier recursion
entre get_latest_snapshot() y get_player_inference_snapshot().
"""
from __future__ import annotations

import gc
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from streamlit_app.config.data_paths import (
    MATCH_HISTORY_PATH,
    PLAYER_CURRENT_STATE_PATH,
    PLAYER_PAIR_CURRENT_STATE_PATH,
    PROJECT_ROOT,
    PROCESSED_DATA_DIR,
)

# Backward-compatible alias used by a few diagnostics. New code should import
# PROCESSED_DATA_DIR directly from streamlit_app.config.data_paths.
PROCESSED = PROCESSED_DATA_DIR


CONTEXT_COLUMNS = [
    "tourney_id", "tourney_name", "tourney_date", "tourney_level",
    "competition_type", "draw_size", "match_num", "match_date", "surface",
    "round", "best_of", "minutes", "score", "tournament_key",
    "canonical_tournament_name", "tourney_name_original",
]

# Higher values represent later rounds. This is a deterministic tie-breaker
# when a source gives every match in one edition the same tournament date.
ROUND_ORDER = {
    "PRE-Q": 0, "Q1": 5, "Q2": 6, "Q3": 7, "RR": 8,
    "R128": 10, "R64": 20, "R32": 30, "R16": 40,
    "QF": 50, "SF": 60, "F": 70, "BR": 75,
}


def _round_order(value: Any) -> int:
    text = str(value or "").strip().upper()
    aliases = {
        "FINAL": "F", "SEMIFINAL": "SF", "SEMI-FINAL": "SF",
        "QUARTERFINAL": "QF", "QUARTER-FINAL": "QF",
        "ROUND OF 16": "R16", "ROUND OF 32": "R32",
        "ROUND OF 64": "R64", "ROUND OF 128": "R128",
    }
    return int(ROUND_ORDER.get(aliases.get(text, text), -1))


def _add_round_order(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rounds = result.get("round", pd.Series("", index=result.index))
    result["_round_order"] = rounds.map(_round_order).astype("Int16")
    return result
PLAYER_IDENTITY_SUFFIXES = [
    "id", "name", "hand", "height", "ioc", "age", "rank", "rank_points",
]
PREMATCH_FEATURE_SUFFIXES = [
    "career_matches_before", "career_wins_before", "career_win_rate_before",
    "surface_matches_before", "surface_wins_before", "surface_win_rate_before",
    "elo_before", "surface_elo_before", "form_last_5_before",
    "form_last_10_before", "avg_minutes_last_5_before",
    "days_since_last_match", "stat_matches_before",
    "aces_per_service_game_before", "double_faults_per_service_game_before",
    "first_serve_in_pct_before", "first_serve_win_pct_before",
    "second_serve_win_pct_before", "service_points_won_pct_before",
    "break_points_saved_pct_before", "return_points_won_pct_before",
    "surface_form_last_5_before", "surface_form_last_10_before",
    "surface_stat_matches_before", "surface_service_points_won_pct_before",
    "surface_return_points_won_pct_before",
    "surface_aces_per_service_game_before",
    "surface_double_faults_per_service_game_before",
    "surface_first_serve_in_pct_before", "surface_first_serve_win_pct_before",
    "surface_second_serve_win_pct_before",
    "surface_break_points_saved_pct_before",
    "surface_elo_change_last_5_before", "elo_change_last_5_before",
    "avg_opponent_elo_last_5_before", "avg_opponent_rank_last_5_before",
    "quality_adjusted_form_last_5_before", "level_matches_before",
    "level_wins_before", "level_win_rate_before", "round_matches_before",
    "round_wins_before", "round_win_rate_before", "best_of_matches_before",
    "best_of_wins_before", "best_of_win_rate_before",
]
V5_PREMATCH_FEATURE_SUFFIXES = [
    "momentum_ewma_30d_before", "momentum_ewma_90d_before",
    "surface_momentum_ewma_90d_before", "momentum_acceleration_before",
    "momentum_volatility_before", "performance_vs_expected_ewma_before",
    "games_won_share_last_5_before", "games_won_share_last_10_before",
    "surface_games_won_share_last_10_before", "sets_won_share_last_10_before",
    "straight_sets_rate_last_10_before", "deciding_set_rate_last_10_before",
    "tiebreak_win_rate_last_20_before", "opponent_adjusted_game_dominance_before",
    "surface_opponent_adjusted_game_dominance_before",
    "matches_last_3d_before", "matches_last_7d_before", "matches_last_14d_before",
    "matches_last_28d_before", "minutes_last_3d_before", "minutes_last_7d_before",
    "minutes_last_14d_before", "minutes_last_28d_before",
    "minutes_coverage_last_7d_before", "games_last_3d_before",
    "games_last_7d_before", "games_last_14d_before", "sets_last_3d_before",
    "sets_last_7d_before", "sets_last_14d_before", "deciding_sets_last_7d_before",
    "tiebreak_sets_last_7d_before", "consecutive_match_days_before",
    "rest_days_before", "rest_days_capped_30_before", "rest_days_squared_before",
    "workload_per_rest_day_before", "same_tournament_matches_before",
    "same_tournament_minutes_before", "same_tournament_wins_before",
    "same_tournament_losses_before", "same_tournament_win_rate_smoothed_before",
    "same_tournament_result_residual_before",
    "same_tournament_games_won_share_before",
    "same_tournament_sets_won_share_before",
    "same_tournament_effective_sample_before",
    "service_points_won_ewma_short_before",
    "service_points_won_ewma_long_before", "return_points_won_ewma_short_before",
    "return_points_won_ewma_long_before", "first_serve_in_ewma_before",
    "first_serve_win_ewma_before", "second_serve_win_ewma_before",
    "aces_per_service_game_ewma_before", "double_faults_per_service_game_ewma_before",
    "break_points_saved_ewma_before", "surface_service_points_won_ewma_before",
    "surface_return_points_won_ewma_before", "serve_return_strength_ewma_before",
    "surface_serve_return_strength_ewma_before",
    "opponent_adjusted_serve_strength_before",
    "opponent_adjusted_return_strength_before", "serve_form_change_before",
    "return_form_change_before", "rank_4w_ago_before", "rank_8w_ago_before",
    "rank_26w_ago_before", "rank_change_4w_before", "rank_change_8w_before",
    "rank_change_26w_before", "rank_points_4w_ago_before",
    "rank_points_8w_ago_before", "rank_points_26w_ago_before",
    "rank_points_change_4w_before", "rank_points_change_8w_before",
    "rank_points_change_26w_before", "elo_change_30d_before",
    "elo_change_90d_before", "surface_elo_change_90d_before",
    "ranking_vs_elo_disagreement_before", "ranking_points_momentum_before",
    "career_high_rank_before", "rank_distance_from_career_high_before",
]
MOMENTUM_V6_PREMATCH_FEATURE_SUFFIXES = [
    "result_residual_ewma_30d_before",
    "result_residual_ewma_90d_before",
    "surface_result_residual_ewma_90d_before",
    "game_share_residual_ewma_30d_before",
    "game_share_residual_ewma_90d_before",
    "serve_residual_ewma_30d_before",
    "return_residual_ewma_30d_before",
    "surface_serve_residual_ewma_90d_before",
    "surface_return_residual_ewma_90d_before",
    "dominance_ratio_ewma_30d_before",
    "momentum_effective_sample_before",
    "momentum_stat_effective_sample_before",
    "momentum_confidence_before",
]
PREMATCH_FEATURE_SUFFIXES = list(dict.fromkeys(
    PREMATCH_FEATURE_SUFFIXES
    + V5_PREMATCH_FEATURE_SUFFIXES
    + MOMENTUM_V6_PREMATCH_FEATURE_SUFFIXES
))

ODDS_SUFFIXES = ["b365_odds", "ps_odds", "max_odds", "avg_odds", "bfe_odds"]
WINNER_MATCH_STATISTICS = {
    "aces": "w_ace", "double_faults": "w_df", "service_points": "w_svpt",
    "first_serves_in": "w_1stIn", "first_serve_points_won": "w_1stWon",
    "second_serve_points_won": "w_2ndWon", "service_games": "w_SvGms",
    "break_points_saved": "w_bpSaved", "break_points_faced": "w_bpFaced",
}
LOSER_MATCH_STATISTICS = {
    "aces": "l_ace", "double_faults": "l_df", "service_points": "l_svpt",
    "first_serves_in": "l_1stIn", "first_serve_points_won": "l_1stWon",
    "second_serve_points_won": "l_2ndWon", "service_games": "l_SvGms",
    "break_points_saved": "l_bpSaved", "break_points_faced": "l_bpFaced",
}
DEFAULT_OVER_GAMES_LINE_BO3 = 22.5
DEFAULT_OVER_GAMES_LINE_BO5 = 38.5

SURFACE_PREFIX = {
    "Hard": "hard", "Clay": "clay", "Grass": "grass",
    "Carpet": "carpet", "Unknown": "unknown",
}


def _require_file(path: Path, label: str) -> Path:
    if path.is_file():
        return path
    raise FileNotFoundError(f"No se encontro {label}: {path}")


@lru_cache(maxsize=1)
def _history_path() -> Path:
    return _require_file(MATCH_HISTORY_PATH, "el historico canonico con features")


@lru_cache(maxsize=1)
def _state_path() -> Path | None:
    return PLAYER_CURRENT_STATE_PATH if PLAYER_CURRENT_STATE_PATH.is_file() else None


@lru_cache(maxsize=1)
def _pair_state_path() -> Path | None:
    return (
        PLAYER_PAIR_CURRENT_STATE_PATH
        if PLAYER_PAIR_CURRENT_STATE_PATH.is_file()
        else None
    )


@lru_cache(maxsize=1)
def _pair_state_dataset() -> ds.Dataset | None:
    path = _pair_state_path()
    return None if path is None else ds.dataset(str(path), format="parquet")


@lru_cache(maxsize=1)
def _available_columns() -> set[str]:
    return set(pq.ParquetFile(_history_path()).schema_arrow.names)


@lru_cache(maxsize=1)
def _history_dataset() -> ds.Dataset:
    return ds.dataset(str(_history_path()), format="parquet")


@lru_cache(maxsize=1)
def _current_state_columns() -> set[str]:
    path = _state_path()
    return set() if path is None else set(pq.ParquetFile(path).schema_arrow.names)


def _safe_numeric(value: Any) -> Any:
    return pd.to_numeric(value, errors="coerce")


def _safe_divide(numerator: Any, denominator: Any) -> Any:
    numerator = _safe_numeric(numerator)
    denominator = _safe_numeric(denominator)
    if isinstance(denominator, pd.Series):
        denominator = denominator.replace(0, np.nan)
    elif pd.notna(denominator) and denominator == 0:
        denominator = np.nan
    return numerator / denominator


def _normalize_player_id(value: Any) -> int | None:
    normalized = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(normalized) else int(normalized)


def _normalize_season(value: Any) -> int | None:
    if value in (None, "ALL", "CAREER"):
        return None
    normalized = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(normalized) else int(normalized)


def _normalize_surface(value: Any) -> str:
    if value in (None, "ALL", "Overall"):
        return "ALL"
    text = str(value).strip().casefold()
    return {
        "hard": "Hard", "clay": "Clay", "grass": "Grass",
        "carpet": "Carpet", "unknown": "Unknown",
    }.get(text, str(value).strip())


def _read_history(columns: list[str], expression: ds.Expression | None = None) -> pd.DataFrame:
    selected = [column for column in dict.fromkeys(columns) if column in _available_columns()]
    if not selected:
        return pd.DataFrame()
    result = _history_dataset().to_table(columns=selected, filter=expression).to_pandas()
    if "match_date" in result:
        result["match_date"] = pd.to_datetime(result["match_date"], errors="coerce")
    return result


def _player_expression(player_id: int) -> ds.Expression:
    return (ds.field("player_1_id") == player_id) | (ds.field("player_2_id") == player_id)


def _calculate_match_statistics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Añade estadísticas derivadas sin fragmentar el DataFrame."""

    service_points = _safe_numeric(
        frame["service_points"]
    )

    first_serves_in = _safe_numeric(
        frame["first_serves_in"]
    )

    first_serve_points_won = _safe_numeric(
        frame["first_serve_points_won"]
    )

    second_serve_points_won = _safe_numeric(
        frame["second_serve_points_won"]
    )

    service_points_won = (
        first_serve_points_won
        + second_serve_points_won
    )

    second_serve_points_played = (
        service_points
        - first_serves_in
    )

    derived = pd.DataFrame(
        {
            "first_serve_in_pct": (
                _safe_divide(
                    first_serves_in,
                    service_points,
                )
            ),
            "first_serve_win_pct": (
                _safe_divide(
                    first_serve_points_won,
                    first_serves_in,
                )
            ),
            "second_serve_win_pct": (
                _safe_divide(
                    second_serve_points_won,
                    second_serve_points_played,
                )
            ),
            "service_points_won": (
                service_points_won
            ),
            "service_points_won_pct": (
                _safe_divide(
                    service_points_won,
                    service_points,
                )
            ),
            "break_points_saved_pct": (
                _safe_divide(
                    frame[
                        "break_points_saved"
                    ],
                    frame[
                        "break_points_faced"
                    ],
                )
            ),
            "aces_per_service_game": (
                _safe_divide(
                    frame["aces"],
                    frame["service_games"],
                )
            ),
            "double_faults_per_service_game": (
                _safe_divide(
                    frame["double_faults"],
                    frame["service_games"],
                )
            ),
        },
        index=frame.index,
    )

    return pd.concat(
        [
            frame,
            derived,
        ],
        axis=1,
        copy=False,
    ).copy()



def _create_player_side(
    matches: pd.DataFrame,
    side: int,
) -> pd.DataFrame:
    """Construye la vista orientada al jugador sin inserciones repetidas."""

    opponent_side = (
        2
        if side == 1
        else 1
    )

    index = matches.index

    columns: dict[str, Any] = {}

    # ------------------------------------------------------
    # CONTEXTO DEL PARTIDO
    # ------------------------------------------------------

    for column in CONTEXT_COLUMNS:
        if column in matches.columns:
            columns[column] = matches[
                column
            ]

    # ------------------------------------------------------
    # IDENTIDAD DE JUGADOR Y RIVAL
    # ------------------------------------------------------

    for suffix in PLAYER_IDENTITY_SUFFIXES:
        own_column = (
            f"player_{side}_{suffix}"
        )

        opponent_column = (
            f"player_{opponent_side}_{suffix}"
        )

        columns[
            f"player_{suffix}"
        ] = (
            matches[own_column]
            if own_column in matches.columns
            else pd.Series(
                pd.NA,
                index=index,
            )
        )

        columns[
            f"opponent_{suffix}"
        ] = (
            matches[opponent_column]
            if opponent_column
            in matches.columns
            else pd.Series(
                pd.NA,
                index=index,
            )
        )

    # ------------------------------------------------------
    # RESULTADO ORIENTADO AL JUGADOR
    # ------------------------------------------------------

    player_1_won = _safe_numeric(
        matches[
            "target_player_1_win"
        ]
    )

    won = (
        player_1_won
        if side == 1
        else 1 - player_1_won
    ).astype("Int8")

    columns["won"] = won

    columns["result"] = pd.Series(
        np.where(
            won.eq(1),
            "W",
            "L",
        ),
        index=index,
        dtype="string",
    )

    # ------------------------------------------------------
    # FEATURES PREPARTIDO
    # ------------------------------------------------------

    for suffix in PREMATCH_FEATURE_SUFFIXES:
        source_column = (
            f"player_{side}_{suffix}"
        )

        if source_column in matches.columns:
            columns[suffix] = matches[
                source_column
            ]

    # ------------------------------------------------------
    # H2H
    # ------------------------------------------------------

    own_h2h_column = (
        f"player_{side}"
        "_h2h_wins_before"
    )

    opponent_h2h_column = (
        f"player_{opponent_side}"
        "_h2h_wins_before"
    )

    h2h_wins = (
        matches[own_h2h_column]
        if own_h2h_column
        in matches.columns
        else pd.Series(
            np.nan,
            index=index,
        )
    )

    opponent_h2h_wins = (
        matches[
            opponent_h2h_column
        ]
        if opponent_h2h_column
        in matches.columns
        else pd.Series(
            np.nan,
            index=index,
        )
    )

    columns[
        "h2h_wins_before"
    ] = h2h_wins

    columns[
        "opponent_h2h_wins_before"
    ] = opponent_h2h_wins

    total_h2h = (
        _safe_numeric(
            h2h_wins
        )
        + _safe_numeric(
            opponent_h2h_wins
        )
    )

    columns[
        "h2h_win_rate_before"
    ] = _safe_divide(
        h2h_wins,
        total_h2h,
    )

    # ------------------------------------------------------
    # CUOTAS
    # ------------------------------------------------------

    for suffix in ODDS_SUFFIXES:
        source_column = (
            f"player_{side}_{suffix}"
        )

        if source_column in matches.columns:
            columns[suffix] = (
                _safe_numeric(
                    matches[
                        source_column
                    ]
                )
            )

    # ------------------------------------------------------
    # ESTADÍSTICAS DEL JUGADOR
    # ------------------------------------------------------

    player_is_winner = won.eq(1)

    opponent_is_winner = (
        ~player_is_winner
    )

    missing_numeric = pd.Series(
        np.nan,
        index=index,
        dtype="float64",
    )

    for (
        output,
        winner_column,
    ) in WINNER_MATCH_STATISTICS.items():
        loser_column = (
            LOSER_MATCH_STATISTICS[
                output
            ]
        )

        winner_values = (
            _safe_numeric(
                matches[
                    winner_column
                ]
            )
            if winner_column
            in matches.columns
            else missing_numeric
        )

        loser_values = (
            _safe_numeric(
                matches[
                    loser_column
                ]
            )
            if loser_column
            in matches.columns
            else missing_numeric
        )

        columns[output] = (
            winner_values.where(
                player_is_winner,
                loser_values,
            )
        )

    # ------------------------------------------------------
    # ESTADÍSTICAS DE SERVICIO DEL RIVAL
    # ------------------------------------------------------

    opponent_stat_outputs = (
        "service_points",
        "first_serve_points_won",
        "second_serve_points_won",
        "service_games",
        "break_points_saved",
        "break_points_faced",
    )

    for output in opponent_stat_outputs:
        winner_column = (
            WINNER_MATCH_STATISTICS[
                output
            ]
        )

        loser_column = (
            LOSER_MATCH_STATISTICS[
                output
            ]
        )

        winner_values = (
            _safe_numeric(
                matches[
                    winner_column
                ]
            )
            if winner_column
            in matches.columns
            else missing_numeric
        )

        loser_values = (
            _safe_numeric(
                matches[
                    loser_column
                ]
            )
            if loser_column
            in matches.columns
            else missing_numeric
        )

        columns[
            f"opponent_{output}"
        ] = winner_values.where(
            opponent_is_winner,
            loser_values,
        )

    # ------------------------------------------------------
    # RETORNO DEL JUGADOR
    # ------------------------------------------------------

    opponent_service_points = (
        _safe_numeric(
            columns[
                "opponent_service_points"
            ]
        )
    )

    opponent_first_won = (
        _safe_numeric(
            columns[
                "opponent_first_serve_points_won"
            ]
        )
    )

    opponent_second_won = (
        _safe_numeric(
            columns[
                "opponent_second_serve_points_won"
            ]
        )
    )

    return_points_won = (
        opponent_service_points
        - opponent_first_won
        - opponent_second_won
    ).clip(
        lower=0
    )

    columns[
        "return_points_played"
    ] = opponent_service_points

    columns[
        "return_points_won"
    ] = return_points_won

    columns[
        "return_points_won_pct"
    ] = _safe_divide(
        return_points_won,
        opponent_service_points,
    )

    columns[
        "player_side"
    ] = pd.Series(
        side,
        index=index,
        dtype="Int8",
    )

    # Una única materialización evita la fragmentación.
    result = pd.DataFrame(
        columns,
        index=index,
    ).copy()

    return _calculate_match_statistics(
        result
    )


def _requested_match_columns() -> list[str]:
    requested = CONTEXT_COLUMNS + ["target_player_1_win"]
    requested += list(WINNER_MATCH_STATISTICS.values()) + list(LOSER_MATCH_STATISTICS.values())
    for side in (1, 2):
        requested += [f"player_{side}_{suffix}" for suffix in PLAYER_IDENTITY_SUFFIXES]
        requested += [f"player_{side}_{suffix}" for suffix in PREMATCH_FEATURE_SUFFIXES]
        requested += [f"player_{side}_{suffix}" for suffix in ODDS_SUFFIXES]
        requested += [f"player_{side}_h2h_wins_before"]
    return list(dict.fromkeys(requested))


@lru_cache(maxsize=256)
def _load_player_matches_cached(player_id: int) -> pd.DataFrame:
    matches = _read_history(_requested_match_columns(), _player_expression(player_id))
    if matches.empty:
        return pd.DataFrame()
    p1 = _create_player_side(matches.loc[matches["player_1_id"].eq(player_id)], 1)
    p2 = _create_player_side(matches.loc[matches["player_2_id"].eq(player_id)], 2)
    result = pd.concat([p1, p2], ignore_index=True)
    result["player_id"] = pd.to_numeric(result["player_id"], errors="coerce").astype("Int64")
    result["opponent_id"] = pd.to_numeric(result["opponent_id"], errors="coerce").astype("Int64")
    result = result.assign(
        season=(
            pd.to_datetime(
                result["match_date"],
                errors="coerce",
            )
            .dt.year
            .astype("Int16")
        )
    )

    if "surface" in result:
        result["surface"] = result["surface"].map(_normalize_surface).astype("string")
    result = result.dropna(subset=["player_id", "player_name", "match_date"])
    result = _add_round_order(result)
    sort_columns = [
        column for column in
        ("match_date", "tourney_id", "_round_order", "match_num")
        if column in result
    ]
    return (
        result.sort_values(
            sort_columns, ascending=True, kind="mergesort", na_position="last"
        )
        .drop(columns=["_round_order"], errors="ignore")
        .reset_index(drop=True)
    )


@lru_cache(maxsize=1)
def load_matches() -> pd.DataFrame:
    return _read_history(_requested_match_columns())


@lru_cache(maxsize=1)
def load_player_per_match() -> pd.DataFrame:
    matches = load_matches()
    return pd.concat([_create_player_side(matches, 1), _create_player_side(matches, 2)], ignore_index=True)


@lru_cache(maxsize=1)
def load_player_snapshots() -> pd.DataFrame:
    per_match = load_player_per_match()
    base = [
        "player_id", "player_name", "player_hand", "player_height", "player_ioc",
        "player_age", "player_rank", "player_rank_points", "match_date", "season",
        "surface", "tourney_id", "tourney_name", "tourney_level", "match_num",
        "round", "best_of", "minutes", "opponent_id", "opponent_name",
        "opponent_rank", "won", "result", "h2h_wins_before",
        "opponent_h2h_wins_before", "h2h_win_rate_before",
    ]
    selected = [c for c in dict.fromkeys(base + PREMATCH_FEATURE_SUFFIXES) if c in per_match]
    return per_match[selected].sort_values(["player_id", "match_date"], kind="mergesort").reset_index(drop=True)


@lru_cache(maxsize=1)
def get_players() -> pd.DataFrame:
    state_path = _state_path()
    if state_path is not None:
        columns = [c for c in ("player_id", "player_name") if c in _current_state_columns()]
        if set(columns) == {"player_id", "player_name"}:
            players = pd.read_parquet(state_path, columns=columns, engine="pyarrow")
            return players.dropna().drop_duplicates("player_id").sort_values(["player_name", "player_id"], kind="mergesort").reset_index(drop=True)

    parquet = pq.ParquetFile(_history_path())
    lookup: dict[int, str] = {}
    for batch in parquet.iter_batches(
        batch_size=50000,
        columns=["player_1_id", "player_1_name", "player_2_id", "player_2_name"],
        use_threads=False,
    ):
        frame = batch.to_pandas()
        for side in (1, 2):
            ids = pd.to_numeric(frame[f"player_{side}_id"], errors="coerce")
            names = frame[f"player_{side}_name"].astype("string").str.strip()
            valid = ids.notna() & names.notna() & names.ne("")
            lookup.update({int(i): str(n) for i, n in zip(ids[valid], names[valid])})
    return pd.DataFrame([{"player_id": i, "player_name": n} for i, n in lookup.items()]).sort_values(["player_name", "player_id"], kind="mergesort").reset_index(drop=True)


@lru_cache(maxsize=512)
def get_player_current_state(player_id: Any) -> pd.Series | None:
    """Lee directamente player_current_state. No llama a otros snapshots."""
    normalized = _normalize_player_id(player_id)
    state_path = _state_path()
    if normalized is None or state_path is None:
        return None
    dataset = ds.dataset(str(state_path), format="parquet")
    frame = dataset.to_table(filter=ds.field("player_id") == normalized).to_pandas()
    if frame.empty:
        return None
    for column in ("state_as_of_date", "last_match_date", "match_date"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    date_column = next((c for c in ("state_as_of_date", "last_match_date", "match_date") if c in frame), None)
    if date_column:
        frame = frame.sort_values(date_column, ascending=False, kind="mergesort", na_position="last")
    return frame.iloc[0].copy()


def get_player_surface_state(player_id: Any, surface: str) -> pd.Series | None:
    state = get_player_current_state(player_id)
    if state is None:
        return None
    normalized_surface = _normalize_surface(surface)
    prefix = SURFACE_PREFIX.get(normalized_surface, "unknown")
    result = state.copy()
    mapping = {
        "surface_matches_before": f"{prefix}_matches",
        "surface_wins_before": f"{prefix}_wins",
        "surface_win_rate_before": f"{prefix}_win_rate",
        "surface_elo_before": f"{prefix}_elo",
        "surface_form_last_5_before": f"{prefix}_form_last_5",
        "surface_form_last_10_before": f"{prefix}_form_last_10",
        "surface_stat_matches_before": f"{prefix}_stat_matches",
        "surface_service_points_won_pct_before": f"{prefix}_service_points_won_pct",
        "surface_return_points_won_pct_before": f"{prefix}_return_points_won_pct",
        "surface_aces_per_service_game_before": f"{prefix}_aces_per_service_game",
        "surface_double_faults_per_service_game_before": f"{prefix}_double_faults_per_service_game",
        "surface_first_serve_in_pct_before": f"{prefix}_first_serve_in_pct",
        "surface_first_serve_win_pct_before": f"{prefix}_first_serve_win_pct",
        "surface_second_serve_win_pct_before": f"{prefix}_second_serve_win_pct",
        "surface_break_points_saved_pct_before": f"{prefix}_break_points_saved_pct",
        "surface_elo_change_last_5_before": f"{prefix}_elo_change_last_5",
    }
    for target, source in mapping.items():
        result[target] = state.get(source, np.nan)
    result["selected_surface"] = normalized_surface
    return result


def _latest_history_snapshot(player_id: int) -> pd.Series | None:
    """Fallback independiente que nunca llama a get_latest_snapshot."""
    history = _load_player_matches_cached(player_id)
    return None if history.empty else history.iloc[-1].copy()


def _latest_context_snapshot(
    player_id: int,
    tourney_level: str | None,
    round_name: str | None,
    best_of: Any,
) -> pd.Series | None:
    history = _load_player_matches_cached(player_id)
    if history.empty:
        return None
    output = history.iloc[-1].copy()
    filters = {
        "level": ("tourney_level", tourney_level),
        "round": ("round", round_name),
        "best_of": ("best_of", str(best_of) if best_of is not None else None),
    }
    suffixes = {
        "level": ("level_matches_before", "level_wins_before", "level_win_rate_before"),
        "round": ("round_matches_before", "round_wins_before", "round_win_rate_before"),
        "best_of": ("best_of_matches_before", "best_of_wins_before", "best_of_win_rate_before"),
    }
    for key, (column, value) in filters.items():
        if value is None or column not in history:
            continue
        subset = history.loc[history[column].astype(str).eq(str(value))]
        if subset.empty:
            for suffix in suffixes[key]:
                output[suffix] = 0.0 if suffix.endswith(("matches_before", "wins_before")) else np.nan
            continue
        row = subset.iloc[-1]
        for suffix in suffixes[key]:
            output[suffix] = row.get(suffix, np.nan)
            if suffix.endswith("matches_before") and pd.notna(output[suffix]):
                output[suffix] = float(output[suffix]) + 1.0
            if suffix.endswith("wins_before") and pd.notna(output[suffix]):
                output[suffix] = float(output[suffix]) + float(row.get("won", 0))
        output[suffixes[key][2]] = _safe_divide(output[suffixes[key][1]], output[suffixes[key][0]])
    return output


def get_player_inference_snapshot(
    player_id: Any,
    surface: str,
    tourney_level: str | None = None,
    round_name: str | None = None,
    best_of: Any = None,
    prediction_date: Any = None,
    competition_type: str | None = None,
    indoor: bool | None = None,
    as_of_date: Any = None,
    tournament_id: Any = None,
    tournament_name: str | None = None,
) -> pd.Series | None:
    """Construye un snapshot v5 de inferencia sin recursividad.

    El estado v5 guarda valores actuales sin el sufijo ``_before``. Esta
    función expone el contrato prepartido esperado por el modelo y aplica el
    contexto solicitado. Los valores no reconstruibles se mantienen como NaN.
    """
    normalized = _normalize_player_id(player_id)
    if normalized is None:
        return None
    state = get_player_surface_state(normalized, surface)
    if state is None:
        state = _latest_history_snapshot(normalized)
    if state is None:
        return None
    result = state.copy()
    aliases = {
        "career_matches_before": "career_matches",
        "career_wins_before": "career_wins",
        "career_win_rate_before": "career_win_rate",
        "elo_before": "elo",
        "form_last_5_before": "form_last_5",
        "form_last_10_before": "form_last_10",
        "avg_minutes_last_5_before": "avg_minutes_last_5",
        "stat_matches_before": "stat_matches",
        "aces_per_service_game_before": "aces_per_service_game",
        "double_faults_per_service_game_before": "double_faults_per_service_game",
        "first_serve_in_pct_before": "first_serve_in_pct",
        "first_serve_win_pct_before": "first_serve_win_pct",
        "second_serve_win_pct_before": "second_serve_win_pct",
        "service_points_won_pct_before": "service_points_won_pct",
        "break_points_saved_pct_before": "break_points_saved_pct",
        "return_points_won_pct_before": "return_points_won_pct",
    }
    for target in PREMATCH_FEATURE_SUFFIXES:
        candidates = [target, aliases.get(target), target.removesuffix("_before")]
        value = np.nan
        for source in candidates:
            if source and source in state.index and pd.notna(state.get(source)):
                value = state.get(source)
                break
        result[target] = value
    context = _latest_context_snapshot(
        normalized, tourney_level, round_name, best_of
    )
    if context is not None:
        for suffix in (
            "level_matches_before", "level_wins_before", "level_win_rate_before",
            "round_matches_before", "round_wins_before", "round_win_rate_before",
            "best_of_matches_before", "best_of_wins_before", "best_of_win_rate_before",
        ):
            result[suffix] = context.get(suffix, result.get(suffix, np.nan))
    prediction_ts = pd.to_datetime(
        prediction_date if prediction_date is not None else as_of_date,
        errors="coerce",
    )
    last_match = pd.to_datetime(
        state.get("last_match_date", state.get("match_date", pd.NaT)),
        errors="coerce",
    )
    if pd.notna(prediction_ts) and pd.notna(last_match):
        rest = max((prediction_ts.normalize() - last_match.normalize()).days, 0)
        result["days_since_last_match"] = float(rest)
        result["rest_days_before"] = float(rest)
        result["rest_days_capped_30_before"] = float(min(rest, 30))
        result["rest_days_squared_before"] = float(min(rest, 30) ** 2)
        minutes_7 = pd.to_numeric(
            result.get("minutes_last_7d_before"), errors="coerce"
        )
        result["workload_per_rest_day_before"] = (
            float(minutes_7) / max(float(rest), 1.0)
            if pd.notna(minutes_7) else np.nan
        )
    result["player_id"] = normalized
    result["selected_surface"] = _normalize_surface(surface)
    result["selected_tourney_level"] = tourney_level
    result["selected_round"] = round_name
    result["selected_best_of"] = best_of
    result["selected_competition_type"] = competition_type
    result["selected_indoor"] = indoor
    result["selected_tournament_id"] = tournament_id
    result["selected_tournament_name"] = tournament_name

    # Current-edition features are contextual. Use them when a reliable
    # tournament name is supplied; otherwise retain the state-file values/NaN.
    if tournament_name:
        current = get_player_current_tournament_statistics(
            player_id=normalized,
            tournament_name=str(tournament_name),
            tournament_id=tournament_id,
            as_of_date=prediction_ts if pd.notna(prediction_ts) else as_of_date,
            surface=surface,
            competition_type=competition_type,
        )
        if current is not None:
            matches = pd.to_numeric(current.get("matches"), errors="coerce")
            wins = pd.to_numeric(current.get("wins"), errors="coerce")
            losses = matches - wins if pd.notna(matches) and pd.notna(wins) else np.nan
            games_won = pd.to_numeric(current.get("games_won"), errors="coerce")
            games_played = pd.to_numeric(current.get("games_played"), errors="coerce")
            sets_won = pd.to_numeric(current.get("sets_won"), errors="coerce")
            sets_played = pd.to_numeric(current.get("sets_played"), errors="coerce")
            result["same_tournament_matches_before"] = matches
            result["same_tournament_minutes_before"] = current.get("minutes_accumulated", np.nan)
            result["same_tournament_wins_before"] = wins
            result["same_tournament_losses_before"] = losses
            result["same_tournament_win_rate_smoothed_before"] = (
                (wins + 1.0) / (matches + 2.0)
                if pd.notna(wins) and pd.notna(matches) else np.nan
            )
            result["same_tournament_games_won_share_before"] = (
                games_won / games_played
                if pd.notna(games_won) and pd.notna(games_played) and games_played > 0
                else np.nan
            )
            result["same_tournament_sets_won_share_before"] = (
                sets_won / sets_played
                if pd.notna(sets_won) and pd.notna(sets_played) and sets_played > 0
                else np.nan
            )
            result["same_tournament_effective_sample_before"] = matches
            # A residual requires the exact training expectation baseline. Do
            # not fabricate it during inference.
            result["same_tournament_result_residual_before"] = np.nan
    return result


@lru_cache(maxsize=2048)
def get_pair_inference_snapshot(
    player_1_id: int,
    player_2_id: int,
    surface: str,
) -> dict[str, Any]:
    """Devuelve H2H global/superficie orientado al primer jugador.

    Las métricas de oponentes comunes se recuperan del último snapshot
    histórico disponible del matchup. Si nunca existió, permanecen como NaN.
    """
    p1, p2 = int(player_1_id), int(player_2_id)
    low, high = sorted((p1, p2))
    p1_is_low = p1 == low
    output: dict[str, Any] = {
        "h2h_matches_before": 0.0,
        "player_1_h2h_win_rate_before": np.nan,
        "player_1_h2h_wins_before": 0.0,
        "player_2_h2h_wins_before": 0.0,
        "h2h_games_won_share_player_1_before": np.nan,
        "surface_h2h_matches_before": 0.0,
        "surface_h2h_win_rate_player_1_before": np.nan,
        "surface_h2h_games_won_share_player_1_before": np.nan,
        "recent_h2h_win_rate_player_1_before": np.nan,
        "recent_h2h_games_won_share_player_1_before": np.nan,
        "common_opponents_count_before": 0.0,
        "common_opponent_score_player_1_before": np.nan,
        "surface_common_opponents_count_before": 0.0,
        "surface_common_opponent_score_player_1_before": np.nan,
        "intransitivity_score_player_1_before": np.nan,
        "surface_intransitivity_score_player_1_before": np.nan,
    }
    dataset = _pair_state_dataset()
    if dataset is not None:
        frame = dataset.to_table(
            filter=(ds.field("player_1_id") == low)
            & (ds.field("player_2_id") == high)
        ).to_pandas()
        for surface_name, prefix in (("ALL", ""), (_normalize_surface(surface), "surface_")):
            row = frame.loc[frame["surface"].astype(str).eq(surface_name)]
            if row.empty:
                continue
            record = row.iloc[0]
            matches = float(pd.to_numeric(record.get("h2h_matches"), errors="coerce"))
            low_wins = float(pd.to_numeric(record.get("player_1_h2h_wins"), errors="coerce"))
            low_rate = low_wins / matches if matches > 0 else np.nan
            low_games = pd.to_numeric(record.get("player_1_games_won_share"), errors="coerce")
            low_recent = pd.to_numeric(record.get("recent_player_1_win_rate"), errors="coerce")
            rate = low_rate if p1_is_low else 1.0 - low_rate
            games = low_games if p1_is_low else 1.0 - low_games
            recent = low_recent if p1_is_low else 1.0 - low_recent
            if prefix:
                output["surface_h2h_matches_before"] = matches
                output["surface_h2h_win_rate_player_1_before"] = rate
                output["surface_h2h_games_won_share_player_1_before"] = games
            else:
                output["h2h_matches_before"] = matches
                output["player_1_h2h_win_rate_before"] = rate
                output["player_1_h2h_wins_before"] = matches * rate
                output["player_2_h2h_wins_before"] = matches * (1.0 - rate)
                output["h2h_games_won_share_player_1_before"] = games
                output["recent_h2h_win_rate_player_1_before"] = recent
    history = _history_dataset()
    names = set(history.schema.names)
    matchup_columns = [
        "match_date", "player_1_id", "player_2_id",
        "recent_h2h_games_won_share_player_1_before",
        "common_opponents_count_before", "common_opponent_score_player_1_before",
        "surface_common_opponents_count_before",
        "surface_common_opponent_score_player_1_before",
        "intransitivity_score_player_1_before",
        "surface_intransitivity_score_player_1_before",
    ]
    selected = [c for c in matchup_columns if c in names]
    if {"player_1_id", "player_2_id"}.issubset(names) and len(selected) > 2:
        frame = history.to_table(
            columns=selected,
            filter=(ds.field("player_1_id") == low)
            & (ds.field("player_2_id") == high),
        ).to_pandas()
        if not frame.empty:
            if "match_date" in frame:
                frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
                frame = frame.sort_values("match_date", kind="mergesort")
            latest = frame.iloc[-1]
            for feature in matchup_columns[3:]:
                if feature not in latest.index:
                    continue
                value = pd.to_numeric(latest.get(feature), errors="coerce")
                if not p1_is_low and feature.endswith("player_1_before") and pd.notna(value):
                    value = 1.0 - value
                output[feature] = value
    return output


def get_latest_snapshot(player_id: Any) -> pd.Series | None:
    """Devuelve el estado global sin llamar a get_player_inference_snapshot."""
    normalized = _normalize_player_id(player_id)
    if normalized is None:
        return None
    state = get_player_current_state(normalized)
    if state is not None:
        result = state.copy()
        aliases = {
            "career_matches_before": "career_matches",
            "career_wins_before": "career_wins",
            "career_win_rate_before": "career_win_rate",
            "elo_before": "elo", "form_last_5_before": "form_last_5",
            "form_last_10_before": "form_last_10",
            "stat_matches_before": "stat_matches",
        }
        for target, source in aliases.items():
            result[target] = state.get(source, state.get(target, np.nan))
        result["player_id"] = normalized
        return result
    return _latest_history_snapshot(normalized)


def get_data_freshness(player_id: Any | None = None, prediction_date: Any = None) -> dict[str, Any]:
    as_of = pd.Timestamp.today().normalize() if prediction_date is None else pd.Timestamp(prediction_date).normalize()
    state_path = _state_path()
    if state_path is None:
        return {"available": False, "status": "MISSING"}
    if player_id is None:
        columns = _current_state_columns()
        date_column = next((c for c in ("state_as_of_date", "last_match_date", "match_date") if c in columns), None)
        if date_column is None:
            state_date = pd.Timestamp.fromtimestamp(state_path.stat().st_mtime).normalize()
        else:
            dates = pd.read_parquet(state_path, columns=[date_column], engine="pyarrow")
            state_date = pd.to_datetime(dates[date_column], errors="coerce").max()
    else:
        state = get_player_current_state(player_id)
        state_date = pd.to_datetime(state.get("state_as_of_date", state.get("last_match_date", pd.NaT)), errors="coerce") if state is not None else pd.NaT
    age = max(int((as_of - state_date.normalize()).days), 0) if pd.notna(state_date) else None
    status = "CURRENT" if age is not None and age <= 1 else "RECENT" if age is not None and age <= 3 else "STALE"
    return {
        "available": pd.notna(state_date), "state_as_of_date": state_date,
        "freshness_days": age, "status": status, "source": str(state_path),
    }


def get_player_matches(
    player_id: Any,
    season: Any = None,
    surface: str = "ALL",
    last_n_matches: int | None = None,
) -> pd.DataFrame:
    normalized = _normalize_player_id(player_id)
    if normalized is None:
        return pd.DataFrame()
    data = _load_player_matches_cached(normalized).copy()
    normalized_season = _normalize_season(season)
    if normalized_season is not None:
        data = data.loc[data["season"].eq(normalized_season)]
    normalized_surface = _normalize_surface(surface)
    if normalized_surface != "ALL" and "surface" in data:
        data = data.loc[data["surface"].astype(str).eq(normalized_surface)]
    data = _add_round_order(data)
    sort_columns = [
        column for column in
        ("match_date", "tourney_id", "_round_order", "match_num")
        if column in data
    ]
    data = data.sort_values(
        sort_columns,
        ascending=[False] * len(sort_columns),
        kind="mergesort",
        na_position="last",
    )
    if last_n_matches:
        data = data.head(int(last_n_matches))
    return data.drop(columns=["_round_order"], errors="ignore").reset_index(drop=True)


def get_player_match_history(player_id: Any, season: Any = None, surface: str = "ALL", last_n_matches: int | None = None) -> pd.DataFrame:
    return get_player_matches(player_id, season, surface, last_n_matches).sort_values("match_date", kind="mergesort").reset_index(drop=True)


def get_player_elo_history(player_id: Any) -> pd.DataFrame:
    columns = [
        "player_id", "player_name", "match_date", "season", "surface",
        "tourney_name", "tourney_level", "opponent_id", "opponent_name",
        "won", "result", "elo_before", "surface_elo_before",
    ]
    history = get_player_match_history(player_id)
    return history[[column for column in columns if column in history]].copy()


def _aggregate_statistics(data: pd.DataFrame, grouping_columns: list[str], scope_name: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    aggregations: dict[str, tuple[str, str]] = {
        "matches": ("won", "size"), "wins": ("won", "sum"),
        "first_match_date": ("match_date", "min"), "last_match_date": ("match_date", "max"),
    }
    optional = {
        "aces": ("aces", "sum"), "double_faults": ("double_faults", "sum"),
        "service_points": ("service_points", "sum"), "first_serves_in": ("first_serves_in", "sum"),
        "first_serve_points_won": ("first_serve_points_won", "sum"),
        "second_serve_points_won": ("second_serve_points_won", "sum"),
        "service_games": ("service_games", "sum"),
        "break_points_saved": ("break_points_saved", "sum"),
        "break_points_faced": ("break_points_faced", "sum"),
        "return_points_won": ("return_points_won", "sum"),
        "return_points_played": ("return_points_played", "sum"),
        "average_match_minutes": ("minutes", "mean"), "latest_rank": ("player_rank", "last"),
        "latest_rank_points": ("player_rank_points", "last"), "latest_elo": ("elo_before", "last"),
        "latest_surface_elo": ("surface_elo_before", "last"),
    }
    aggregations.update({k: v for k, v in optional.items() if v[0] in data})
    grouped = data.groupby(grouping_columns, dropna=False, observed=True).agg(**aggregations).reset_index()
    grouped["scope"] = scope_name
    grouped["losses"] = grouped["matches"] - grouped["wins"]
    grouped["win_rate"] = _safe_divide(grouped["wins"], grouped["matches"])
    for column in optional:
        if column not in grouped:
            grouped[column] = np.nan
    grouped["aces_per_service_game"] = _safe_divide(grouped["aces"], grouped["service_games"])
    grouped["double_faults_per_service_game"] = _safe_divide(grouped["double_faults"], grouped["service_games"])
    grouped["first_serve_in_pct"] = _safe_divide(grouped["first_serves_in"], grouped["service_points"])
    grouped["first_serve_win_pct"] = _safe_divide(grouped["first_serve_points_won"], grouped["first_serves_in"])
    grouped["second_serve_win_pct"] = _safe_divide(grouped["second_serve_points_won"], grouped["service_points"] - grouped["first_serves_in"])
    grouped["service_points_won_pct"] = _safe_divide(grouped["first_serve_points_won"] + grouped["second_serve_points_won"], grouped["service_points"])
    grouped["break_points_saved_pct"] = _safe_divide(grouped["break_points_saved"], grouped["break_points_faced"])
    grouped["return_points_won_pct"] = _safe_divide(grouped["return_points_won"], grouped["return_points_played"])
    return grouped


@lru_cache(maxsize=64)
def load_single_player_statistics(player_id: Any) -> pd.DataFrame:
    normalized = _normalize_player_id(player_id)
    if normalized is None:
        return pd.DataFrame()
    data = get_player_matches(normalized)
    if data.empty:
        return pd.DataFrame()
    grouping = ["player_id", "player_name", "season", "surface", "competition_type", "tournament_level"]
    variants = [
        (data.assign(season="CAREER", surface="ALL", competition_type="ALL", tournament_level="ALL"), "CAREER"),
        (data.assign(season="CAREER", competition_type="ALL", tournament_level="ALL"), "CAREER"),
        (data.assign(surface="ALL", competition_type="ALL", tournament_level="ALL"), "SEASON"),
        (data.assign(competition_type="ALL", tournament_level="ALL"), "SEASON"),
    ]
    if "tourney_level" in data:
        variants.append((data.assign(season="CAREER", surface="ALL", competition_type="ALL", tournament_level=data["tourney_level"].astype("string").fillna("UNKNOWN")), "CAREER"))
    parts = [_aggregate_statistics(frame, grouping, scope) for frame, scope in variants]
    parts = [part for part in parts if not part.empty]
    return pd.concat(parts, ignore_index=True).sort_values(["scope", "season", "surface", "tournament_level"], kind="mergesort").reset_index(drop=True) if parts else pd.DataFrame()


def get_player_statistics(player_id: Any, season: Any = None, surface: str = "ALL", scope: str = "CAREER", last_n_matches: int | None = None) -> pd.DataFrame:
    normalized = _normalize_player_id(player_id)
    if normalized is None:
        return pd.DataFrame()
    if last_n_matches:
        history = get_player_matches(normalized, season, surface, last_n_matches)
        if history.empty:
            return pd.DataFrame()
        history = history.assign(
            season=str(season) if season not in (None, "CAREER", "ALL") else "CAREER",
            surface=str(surface) if surface not in (None, "ALL") else "ALL",
            competition_type="ALL", tournament_level="ALL",
        )
        return _aggregate_statistics(history, ["player_id", "player_name", "season", "surface", "competition_type", "tournament_level"], f"LAST_{int(last_n_matches)}")
    result = load_single_player_statistics(normalized)
    if result.empty:
        return result
    if surface is not None:
        result = result.loc[result["surface"].astype(str).eq(str(surface))]
    if scope is not None:
        result = result.loc[result["scope"].astype(str).eq(str(scope))]
    if season not in (None, "ALL"):
        result = result.loc[result["season"].astype(str).eq(str(season))]
    return result.reset_index(drop=True)


def search_player(player_name: Any) -> pd.DataFrame:
    text = str(player_name).strip()
    players = get_players()
    if not text:
        return players.iloc[0:0].copy()
    return players.loc[players["player_name"].astype("string").str.contains(text, case=False, na=False, regex=False)].reset_index(drop=True)


def get_player_recent_form(player_id: Any, last_n: int = 10) -> dict[str, list[str]]:
    data = get_player_matches(player_id, last_n_matches=last_n)
    if data.empty:
        return {}
    return {surface: subset["result"].fillna("L").astype(str).tolist() for surface, subset in data.groupby("surface", dropna=True)}


def get_available_seasons() -> list[str]:
    dates = _read_history(["match_date"])
    return sorted(pd.to_datetime(dates["match_date"], errors="coerce").dt.year.dropna().astype(int).astype(str).unique().tolist(), reverse=True)


def get_player_surface_breakdown(player_id: Any) -> pd.DataFrame:
    stats = load_single_player_statistics(player_id)
    if stats.empty:
        return stats
    return stats.loc[
        stats["surface"].astype(str).ne("ALL")
        & stats["competition_type"].astype(str).eq("ALL")
        & stats["tournament_level"].astype(str).eq("ALL")
        & stats["scope"].astype(str).eq("CAREER")
        & stats["season"].astype(str).eq("CAREER")
    ].sort_values("matches", ascending=False, kind="mergesort").reset_index(drop=True)


def get_player_tournament_breakdown(player_id: Any) -> pd.DataFrame:
    stats = load_single_player_statistics(player_id)
    if stats.empty:
        return stats
    return stats.loc[
        stats["surface"].astype(str).eq("ALL")
        & stats["competition_type"].astype(str).eq("ALL")
        & stats["tournament_level"].astype(str).ne("ALL")
        & stats["scope"].astype(str).eq("CAREER")
        & stats["season"].astype(str).eq("CAREER")
    ].sort_values("matches", ascending=False, kind="mergesort").reset_index(drop=True)


def get_available_surfaces() -> list[str]:
    values = _read_history(["surface"])["surface"].dropna().map(_normalize_surface).unique().tolist()
    preferred = ["Hard", "Clay", "Grass", "Carpet", "Unknown"]
    return ["ALL"] + [value for value in preferred if value in values]


def get_available_tourney_levels() -> list[str]:
    values = _read_history(["tourney_level"])["tourney_level"].dropna().astype(str).unique().tolist()
    return ["ALL"] + sorted(values)


def get_head_to_head_history(
    player_1_id: Any,
    player_2_id: Any,
    last_n_matches: int | None = None,
) -> pd.DataFrame:
    """Direct meetings, newest first, including the recorded score."""
    first = _normalize_player_id(player_1_id)
    second = _normalize_player_id(player_2_id)
    if first is None or second is None or first == second:
        return pd.DataFrame()
    required = list(dict.fromkeys(CONTEXT_COLUMNS + [
        "player_1_id", "player_1_name", "player_2_id", "player_2_name",
        "target_player_1_win",
    ]))
    expression = (
        ((ds.field("player_1_id") == first) & (ds.field("player_2_id") == second))
        | ((ds.field("player_1_id") == second) & (ds.field("player_2_id") == first))
    )
    matches = _read_history(required, expression)
    if matches.empty:
        return pd.DataFrame()
    p1_won = pd.to_numeric(matches["target_player_1_win"], errors="coerce").eq(1)
    matches["winner_id"] = matches["player_1_id"].where(p1_won, matches["player_2_id"])
    matches["winner_name"] = matches["player_1_name"].where(p1_won, matches["player_2_name"])
    matches["loser_id"] = matches["player_2_id"].where(p1_won, matches["player_1_id"])
    matches["loser_name"] = matches["player_2_name"].where(p1_won, matches["player_1_name"])
    matches = _add_round_order(matches)
    sort_columns = [c for c in ("match_date", "tourney_id", "_round_order", "match_num") if c in matches]
    matches = matches.sort_values(
        sort_columns, ascending=[False] * len(sort_columns),
        kind="mergesort", na_position="last",
    )
    if last_n_matches:
        matches = matches.head(int(last_n_matches))
    output = [
        "match_date", "tourney_name", "tourney_level", "competition_type",
        "surface", "round", "winner_id", "winner_name", "loser_id",
        "loser_name", "score", "match_num",
    ]
    return matches[[c for c in output if c in matches]].reset_index(drop=True)


def _normalize_tournament_label(value: Any) -> str:
    """Normalize a tournament label for display-insensitive comparisons."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    return re.sub(r"\s+", " ", text).strip()


TOURNAMENT_CANONICAL_ALIASES = {
    # Shanghai Challenger promotional naming.
    "road to the rolex shanghai masters": "shanghai",
    "road to rolex shanghai masters": "shanghai",
    "rolex shanghai masters road to": "shanghai",
    "shanghai challenger": "shanghai",
    "shanghai": "shanghai",

    # Istanbul Challenger promotional naming. Alias keys are already normalized.
    "istanbul challenger ted open": "istanbul",
    "istanbul ted open": "istanbul",
    "ted open istanbul": "istanbul",
    "istanbul challenger": "istanbul",
    "istanbul": "istanbul",

    # Genoa provider naming.
    "genoa nuova valletta": "genoa",
    "genoa challenger": "genoa",
    "genoa": "genoa",

    # Grand Slam naming.
    "u s open": "us open",
    "u s open new york": "us open",
    "us open new york": "us open",
    "us open": "us open",
    "french open": "roland garros",
    "roland garros paris": "roland garros",
    "roland garros": "roland garros",
    "wimbledon london": "wimbledon",
    "australian open melbourne": "australian open",
}

def _canonical_tournament_key(
    value: Any,
) -> str:
    """Return a stable key across fixture and historical tournament labels.

    Examples:
        U.S. Open - New York -> us open
        US Open             -> us open
        Manacor Challenger  -> manacor
        Manacor             -> manacor
        Porto Challenger    -> porto
        Porto (CTP)         -> porto

    Location suffixes, competition-category suffixes and known technical
    codes are removed conservatively.
    """
    normalized = _normalize_tournament_label(
        value
    )

    if not normalized:
        return ""

    if normalized in TOURNAMENT_CANONICAL_ALIASES:
        return TOURNAMENT_CANONICAL_ALIASES[
            normalized
        ]

    location_suffixes = (
        " new york",
        " paris",
        " london",
        " melbourne",
    )

    category_suffixes = (
        " atp challenger",
        " challenger tour",
        " challenger",
        " futures",
        " future",
        " qualifying",
        " qualification",
        " qual",
        " ch",
    )

    # Technical provider codes found in tournament labels.
    # Because punctuation is removed by _normalize_tournament_label(),
    # "Porto (CTP)" reaches this function as "porto ctp".
    technical_suffixes = (
        " ctp",
        " atp",
        " wta",
        " itf",
    )

    changed = True

    while changed:
        changed = False

        for suffix in (
            *location_suffixes,
            *category_suffixes,
            *technical_suffixes,
        ):
            if not normalized.endswith(
                suffix
            ):
                continue

            candidate = normalized[
                :-len(suffix)
            ].strip()

            if not candidate:
                continue

            normalized = candidate
            changed = True
            break

    return TOURNAMENT_CANONICAL_ALIASES.get(
        normalized,
        normalized,
    )


def _tournament_key_series(data: pd.DataFrame) -> pd.Series:
    """Build the best canonical tournament key available for every row."""
    result = pd.Series("", index=data.index, dtype="string")
    for column in (
        "canonical_tournament_name",
        "tourney_name",
        "tourney_name_original",
    ):
        if column not in data:
            continue
        keys = data[column].map(_canonical_tournament_key).astype("string")
        result = result.mask(result.eq("") & keys.ne(""), keys)
    return result

def _score_totals(score: Any) -> tuple[float, float, float, float]:
    """Return winner/loser sets and games from a winner-oriented score.

    Played set tokens are counted even when a set is unfinished because of a
    retirement. Tiebreak points inside parentheses are not counted as games.
    Walkovers or scores without parseable set tokens return missing values.
    """
    if score is None or pd.isna(score):
        return np.nan, np.nan, np.nan, np.nan
    text = str(score).upper().strip()
    if not text or text in {"NAN", "NONE", "<NA>", "W/O", "WO", "WALKOVER"}:
        return np.nan, np.nan, np.nan, np.nan

    winner_sets = 0
    loser_sets = 0
    winner_games = 0
    loser_games = 0
    parsed_sets = 0

    for token in text.split():
        score_match = re.match(r"^(\d+)-(\d+)(?:\([^)]*\))?$", token)
        if score_match is None:
            continue
        left = int(score_match.group(1))
        right = int(score_match.group(2))
        parsed_sets += 1
        winner_games += left
        loser_games += right
        if left > right:
            winner_sets += 1
        elif right > left:
            loser_sets += 1

    if parsed_sets == 0:
        return np.nan, np.nan, np.nan, np.nan
    return (
        float(winner_sets),
        float(loser_sets),
        float(winner_games),
        float(loser_games),
    )


def _score_set_totals(score: Any) -> tuple[float, float]:
    """Backward-compatible set-only score parser."""
    winner_sets, loser_sets, _, _ = _score_totals(score)
    return winner_sets, loser_sets


def _match_score_totals_for_player(
    row: pd.Series,
) -> tuple[float, float, float, float]:
    winner_sets, loser_sets, winner_games, loser_games = _score_totals(
        row.get("score")
    )
    if pd.isna(winner_sets):
        return np.nan, np.nan, np.nan, np.nan
    if str(row.get("result", "")).upper() == "W":
        return winner_sets, loser_sets, winner_games, loser_games
    return loser_sets, winner_sets, loser_games, winner_games


def _match_sets_for_player(row: pd.Series) -> tuple[float, float]:
    """Backward-compatible player-oriented set totals."""
    sets_won, sets_lost, _, _ = _match_score_totals_for_player(row)
    return sets_won, sets_lost


def _filter_current_tournament_matches(
    player_id: Any,
    tournament_name: str,
    tournament_id: Any = None,
    as_of_date: Any = None,
    surface: Any = None,
    competition_type: Any = None,
    maximum_edition_days: int = 28,
) -> pd.DataFrame:
    """Return completed matches from the current tournament edition.

    Resolution is based on canonical tournament identity plus a strict date
    window ending on the fixture date. A provider tournament ID is only used as
    supporting evidence after the canonical name matches, because fixture IDs
    and historical Sackmann/TML IDs can use unrelated namespaces and collide.
    """
    normalized_player_id = _normalize_player_id(player_id)
    target_key = _canonical_tournament_key(tournament_name)
    if normalized_player_id is None or not target_key:
        return pd.DataFrame()

    data = _load_player_matches_cached(normalized_player_id).copy()
    if data.empty:
        return data

    data["match_date"] = pd.to_datetime(data["match_date"], errors="coerce")
    reference = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(reference):
        reference = data["match_date"].max()
    if pd.isna(reference):
        return data.iloc[0:0].copy()
    reference = reference.normalize()
    minimum_date = reference - pd.Timedelta(
        days=max(int(maximum_edition_days), 1)
    )

    # Only completed matches close to and not after the upcoming fixture.
    data = data.loc[
        data["match_date"].between(
            minimum_date,
            reference,
            inclusive="both",
        )
    ].copy()
    if data.empty:
        return data

    historical_keys = _tournament_key_series(data)
    name_mask = historical_keys.eq(target_key)

    # Surface and competition are optional safeguards. They are deliberately
    # applied only when the calling page supplies a meaningful value.
    surface_mask = pd.Series(True, index=data.index, dtype=bool)
    normalized_surface = _normalize_surface(surface)
    if (
        surface not in (None, "", "UNKNOWN", "Unknown")
        and normalized_surface != "ALL"
        and "surface" in data.columns
    ):
        surface_mask = (
            data["surface"]
            .map(_normalize_surface)
            .astype("string")
            .eq(normalized_surface)
        )

    competition_mask = pd.Series(True, index=data.index, dtype=bool)
    competition_text = str(competition_type or "").strip().upper()
    if competition_text not in {"", "UNKNOWN", "NONE", "NAN"} and "competition_type" in data.columns:
        competition_mask = (
            data["competition_type"]
            .astype("string")
            .str.strip()
            .str.upper()
            .eq(competition_text)
        )

    selected = data.loc[
        name_mask & surface_mask & competition_mask
    ].copy()
    if selected.empty:
        return selected

    # A matching ID can narrow rows only after canonical identity, date,
    # surface and competition have matched. Never return id_mask by itself.
    if tournament_id not in (None, "") and "tourney_id" in selected.columns:
        id_mask = (
            selected["tourney_id"]
            .astype("string")
            .str.strip()
            .eq(str(tournament_id).strip())
        )
        if id_mask.any():
            selected = selected.loc[id_mask].copy()

    selected["_fixture_tournament_name"] = str(tournament_name)
    selected["_fixture_canonical_key"] = target_key
    selected["_historical_canonical_key"] = _tournament_key_series(selected)
    selected["_edition_window_start"] = minimum_date
    selected["_edition_window_end"] = reference
    selected["_external_tournament_id"] = str(tournament_id or "")

    return selected.sort_values(
        [column for column in ("match_date", "match_num") if column in selected],
        ascending=True,
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

def get_player_current_tournament_matches(
    player_id: Any,
    tournament_name: str,
    tournament_id: Any = None,
    as_of_date: Any = None,
    surface: Any = None,
    competition_type: Any = None,
    maximum_edition_days: int = 28,
) -> pd.DataFrame:
    """Completed current-edition matches with player-oriented statistics."""
    data = _filter_current_tournament_matches(
        player_id=player_id,
        tournament_name=tournament_name,
        tournament_id=tournament_id,
        as_of_date=as_of_date,
        surface=surface,
        competition_type=competition_type,
        maximum_edition_days=maximum_edition_days,
    ).copy()
    if data.empty:
        return data
    totals = data.apply(_match_score_totals_for_player, axis=1, result_type="expand")
    totals.columns = ["sets_won", "sets_lost", "games_won", "games_lost"]
    data = pd.concat([data.reset_index(drop=True), totals.reset_index(drop=True)], axis=1)
    generated = pd.to_numeric(data.get("opponent_break_points_faced"), errors="coerce")
    saved_by_opponent = pd.to_numeric(data.get("opponent_break_points_saved"), errors="coerce")
    data["break_points_generated"] = generated
    data["break_points_converted"] = (generated - saved_by_opponent).clip(lower=0)
    data["break_point_conversion_rate"] = _safe_divide(data["break_points_converted"], generated)
    conceded = (
        pd.to_numeric(data.get("break_points_faced"), errors="coerce")
        - pd.to_numeric(data.get("break_points_saved"), errors="coerce")
    ).clip(lower=0)
    data["breaks_conceded"] = conceded
    data["service_games_held"] = (
        pd.to_numeric(data.get("service_games"), errors="coerce") - conceded
    ).clip(lower=0)
    data["service_hold_rate"] = _safe_divide(data["service_games_held"], data.get("service_games"))
    data["total_points_won"] = (
        pd.to_numeric(data.get("service_points_won"), errors="coerce")
        + pd.to_numeric(data.get("return_points_won"), errors="coerce")
    )
    data["total_points_played"] = (
        pd.to_numeric(data.get("service_points"), errors="coerce")
        + pd.to_numeric(data.get("return_points_played"), errors="coerce")
    )
    data["total_points_won_pct"] = _safe_divide(data["total_points_won"], data["total_points_played"])
    return data.sort_values("match_date", ascending=False, kind="mergesort").reset_index(drop=True)


def get_player_current_tournament_statistics(
    player_id: Any,
    tournament_name: str,
    tournament_id: Any = None,
    as_of_date: Any = None,
    surface: Any = None,
    competition_type: Any = None,
    maximum_edition_days: int = 28,
) -> pd.Series | None:
    """Aggregate completed matches in the current tournament edition."""
    data = _filter_current_tournament_matches(
        player_id=player_id,
        tournament_name=tournament_name,
        tournament_id=tournament_id,
        as_of_date=as_of_date,
        surface=surface,
        competition_type=competition_type,
        maximum_edition_days=maximum_edition_days,
    )
    if data.empty:
        return None

    score_totals = data.apply(
        _match_score_totals_for_player,
        axis=1,
        result_type="expand",
    )
    score_totals.columns = [
        "sets_won",
        "sets_lost",
        "games_won",
        "games_lost",
    ]
    data = pd.concat(
        [data.reset_index(drop=True), score_totals.reset_index(drop=True)],
        axis=1,
    )

    def total(column: str) -> float:
        if column not in data:
            return np.nan
        values = pd.to_numeric(data[column], errors="coerce")
        return float(values.sum(min_count=1))

    matches = int(len(data))
    wins = int(pd.to_numeric(data["won"], errors="coerce").fillna(0).sum())
    minutes = pd.to_numeric(data.get("minutes"), errors="coerce")
    covered_minutes = int(minutes.notna().sum())
    score_coverage = int(data["score"].notna().sum()) if "score" in data else 0
    stat_matches = int(
        pd.to_numeric(data.get("service_points"), errors="coerce").notna().sum()
    )
    service_game_stat_matches = int(
        pd.to_numeric(data.get("service_games"), errors="coerce").notna().sum()
    )

    sets_won = total("sets_won")
    sets_lost = total("sets_lost")
    sets_played = sets_won + sets_lost
    games_won = total("games_won")
    games_lost = total("games_lost")
    games_played = games_won + games_lost

    break_points_saved = total("break_points_saved")
    break_points_faced = total("break_points_faced")
    breaks_conceded = (
        max(break_points_faced - break_points_saved, 0.0)
        if pd.notna(break_points_faced) and pd.notna(break_points_saved)
        else np.nan
    )
    service_games_played = total("service_games")
    service_games_held = (
        max(service_games_played - breaks_conceded, 0.0)
        if pd.notna(service_games_played) and pd.notna(breaks_conceded)
        else np.nan
    )

    aces = total("aces")
    double_faults = total("double_faults")

    break_points_generated = total("opponent_break_points_faced")
    opponent_break_points_saved = total("opponent_break_points_saved")
    break_points_converted = (
        max(break_points_generated - opponent_break_points_saved, 0.0)
        if pd.notna(break_points_generated) and pd.notna(opponent_break_points_saved)
        else np.nan
    )
    service_points_won = total("service_points_won")
    return_points_won = total("return_points_won")
    total_points_won = service_points_won + return_points_won
    total_points_played = total("service_points") + total("return_points_played")
    average_opponent_rank = pd.to_numeric(data.get("opponent_rank"), errors="coerce").mean()
    average_opponent_elo = pd.to_numeric(data.get("opponent_elo_before"), errors="coerce").mean()

    output = {
        "player_id": _normalize_player_id(player_id),
        "player_name": str(data["player_name"].dropna().iloc[-1]),
        "tournament": str(tournament_name),
        "fixture_canonical_tournament_key": _canonical_tournament_key(
            tournament_name
        ),
        "matched_historical_tournament_names": sorted(
            data.get("tourney_name", pd.Series(dtype="string"))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "matched_historical_tournament_ids": sorted(
            data.get("tourney_id", pd.Series(dtype="string"))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "edition_window_start": (
            data["_edition_window_start"].iloc[0]
            if "_edition_window_start" in data.columns else pd.NaT
        ),
        "edition_window_end": (
            data["_edition_window_end"].iloc[0]
            if "_edition_window_end" in data.columns else pd.NaT
        ),
        "matches": matches,
        "wins": wins,
        "service_points_won_pct": _safe_divide(
            total("service_points_won"), total("service_points")
        ),
        "return_points_won_pct": _safe_divide(
            total("return_points_won"), total("return_points_played")
        ),
        "first_serve_in_pct": _safe_divide(
            total("first_serves_in"), total("service_points")
        ),
        "first_serve_win_pct": _safe_divide(
            total("first_serve_points_won"), total("first_serves_in")
        ),
        "second_serve_win_pct": _safe_divide(
            total("second_serve_points_won"),
            total("service_points") - total("first_serves_in"),
        ),
        "break_points_saved": break_points_saved,
        "break_points_saved_per_match": _safe_divide(
            break_points_saved, matches
        ),
        "break_points_faced": break_points_faced,
        "break_points_saved_pct": _safe_divide(break_points_saved, break_points_faced),
        "break_points_generated": break_points_generated,
        "break_points_generated_per_match": _safe_divide(break_points_generated, matches),
        "break_points_converted": break_points_converted,
        "break_points_converted_per_match": _safe_divide(break_points_converted, matches),
        "break_point_conversion_rate": _safe_divide(break_points_converted, break_points_generated),
        "breaks_conceded": breaks_conceded,
        "service_games_played": service_games_played,
        "service_games_held": service_games_held,
        "service_hold_rate": _safe_divide(
            service_games_held, service_games_played
        ),
        "aces": aces,
        "aces_per_match": _safe_divide(aces, matches),
        "double_faults": double_faults,
        "double_faults_per_match": _safe_divide(double_faults, matches),
        "service_points_won": service_points_won,
        "return_points_won": return_points_won,
        "total_points_won": total_points_won,
        "total_points_played": total_points_played,
        "total_points_won_pct": _safe_divide(total_points_won, total_points_played),
        "average_opponent_rank": average_opponent_rank,
        "average_opponent_elo": average_opponent_elo,
        "games_played": games_played,
        "games_played_per_match": _safe_divide(games_played, matches),
        "games_won": games_won,
        "games_won_per_match": _safe_divide(games_won, matches),
        "games_lost": games_lost,
        "games_lost_per_match": _safe_divide(games_lost, matches),
        "sets_played": sets_played,
        "sets_played_per_match": _safe_divide(sets_played, matches),
        "sets_won": sets_won,
        "sets_won_per_match": _safe_divide(sets_won, matches),
        "sets_lost": sets_lost,
        "sets_lost_per_match": _safe_divide(sets_lost, matches),
        "minutes_accumulated": total("minutes"),
        "minutes_per_match": float(minutes.mean()) if covered_minutes else np.nan,
        "matches_with_score": score_coverage,
        "stat_matches": stat_matches,
        "service_game_stat_matches": service_game_stat_matches,
        "matches_with_minutes": covered_minutes,
        "score_coverage": _safe_divide(score_coverage, matches),
        "point_stat_coverage": _safe_divide(stat_matches, matches),
        "service_game_stat_coverage": _safe_divide(
            service_game_stat_matches, matches
        ),
        "minutes_coverage": _safe_divide(covered_minutes, matches),
    }
    return pd.Series(output)


def get_players_current_tournament_statistics(
    player_1_id: Any,
    player_2_id: Any,
    tournament_name: str,
    tournament_id: Any = None,
    as_of_date: Any = None,
    surface: Any = None,
    competition_type: Any = None,
    maximum_edition_days: int = 28,
) -> pd.DataFrame:
    """Current-edition tournament statistics for the selected players."""
    rows = [
        get_player_current_tournament_statistics(
            player_id=player_id,
            tournament_name=tournament_name,
            tournament_id=tournament_id,
            as_of_date=as_of_date,
            surface=surface,
            competition_type=competition_type,
            maximum_edition_days=maximum_edition_days,
        )
        for player_id in (player_1_id, player_2_id)
    ]
    rows = [row for row in rows if row is not None]
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()



def _aggregate_tournament_match_window(
    data: pd.DataFrame,
    player_id: Any,
    tournament_label: str,
) -> pd.Series | None:
    """Aggregate a pre-filtered tournament edition using current-tournament metrics."""
    if data is None or data.empty:
        return None
    data = data.copy().reset_index(drop=True)
    score_totals = data.apply(
        _match_score_totals_for_player,
        axis=1,
        result_type="expand",
    )
    score_totals.columns = ["sets_won", "sets_lost", "games_won", "games_lost"]
    data = pd.concat([data, score_totals], axis=1)

    def total(column: str) -> float:
        if column not in data:
            return np.nan
        return float(pd.to_numeric(data[column], errors="coerce").sum(min_count=1))

    matches = int(len(data))
    wins = int(pd.to_numeric(data.get("won"), errors="coerce").fillna(0).sum())
    minutes = pd.to_numeric(data.get("minutes"), errors="coerce")
    covered_minutes = int(minutes.notna().sum())
    score_coverage = int(data.get("score", pd.Series(pd.NA, index=data.index)).notna().sum())
    stat_matches = int(pd.to_numeric(data.get("service_points"), errors="coerce").notna().sum())
    service_game_stat_matches = int(pd.to_numeric(data.get("service_games"), errors="coerce").notna().sum())

    sets_won, sets_lost = total("sets_won"), total("sets_lost")
    games_won, games_lost = total("games_won"), total("games_lost")
    sets_played = sets_won + sets_lost
    games_played = games_won + games_lost
    break_points_saved = total("break_points_saved")
    break_points_faced = total("break_points_faced")
    breaks_conceded = (
        max(break_points_faced - break_points_saved, 0.0)
        if pd.notna(break_points_faced) and pd.notna(break_points_saved)
        else np.nan
    )
    service_games_played = total("service_games")
    service_games_held = (
        max(service_games_played - breaks_conceded, 0.0)
        if pd.notna(service_games_played) and pd.notna(breaks_conceded)
        else np.nan
    )
    aces, double_faults = total("aces"), total("double_faults")
    latest = data.sort_values("match_date", kind="mergesort").iloc[-1]
    first_date = pd.to_datetime(data["match_date"], errors="coerce").min()
    last_date = pd.to_datetime(data["match_date"], errors="coerce").max()

    return pd.Series({
        "player_id": _normalize_player_id(player_id),
        "player_name": str(latest.get("player_name", "")),
        "tournament": str(tournament_label),
        "tourney_id": str(latest.get("tourney_id", "")),
        "surface": str(latest.get("surface", "")),
        "first_match_date": first_date,
        "last_match_date": last_date,
        "matches": matches,
        "wins": wins,
        "service_points_won_pct": _safe_divide(total("service_points_won"), total("service_points")),
        "return_points_won_pct": _safe_divide(total("return_points_won"), total("return_points_played")),
        "first_serve_in_pct": _safe_divide(total("first_serves_in"), total("service_points")),
        "first_serve_win_pct": _safe_divide(total("first_serve_points_won"), total("first_serves_in")),
        "second_serve_win_pct": _safe_divide(
            total("second_serve_points_won"), total("service_points") - total("first_serves_in")
        ),
        "break_points_saved": break_points_saved,
        "break_points_saved_per_match": _safe_divide(break_points_saved, matches),
        "break_points_faced": break_points_faced,
        "breaks_conceded": breaks_conceded,
        "service_games_played": service_games_played,
        "service_games_held": service_games_held,
        "service_hold_rate": _safe_divide(service_games_held, service_games_played),
        "aces": aces,
        "aces_per_match": _safe_divide(aces, matches),
        "double_faults": double_faults,
        "double_faults_per_match": _safe_divide(double_faults, matches),
        "games_played": games_played,
        "games_played_per_match": _safe_divide(games_played, matches),
        "games_won": games_won,
        "games_won_per_match": _safe_divide(games_won, matches),
        "games_lost": games_lost,
        "games_lost_per_match": _safe_divide(games_lost, matches),
        "sets_played": sets_played,
        "sets_played_per_match": _safe_divide(sets_played, matches),
        "sets_won": sets_won,
        "sets_won_per_match": _safe_divide(sets_won, matches),
        "sets_lost": sets_lost,
        "sets_lost_per_match": _safe_divide(sets_lost, matches),
        "minutes_accumulated": total("minutes"),
        "minutes_per_match": float(minutes.mean()) if covered_minutes else np.nan,
        "matches_with_score": score_coverage,
        "stat_matches": stat_matches,
        "service_game_stat_matches": service_game_stat_matches,
        "matches_with_minutes": covered_minutes,
        "score_coverage": _safe_divide(score_coverage, matches),
        "point_stat_coverage": _safe_divide(stat_matches, matches),
        "service_game_stat_coverage": _safe_divide(service_game_stat_matches, matches),
        "minutes_coverage": _safe_divide(covered_minutes, matches),
    })


def get_player_recent_tournament_statistics(
    player_id: Any,
    last_n_tournaments: int = 3,
    as_of_date: Any = None,
) -> pd.DataFrame:
    """Return one aggregate row for each of a player's most recent tournaments."""
    normalized = _normalize_player_id(player_id)
    if normalized is None:
        return pd.DataFrame()
    data = _load_player_matches_cached(normalized).copy()
    if data.empty:
        return data
    data["match_date"] = pd.to_datetime(data["match_date"], errors="coerce")
    reference = pd.to_datetime(as_of_date, errors="coerce")
    if pd.notna(reference):
        data = data.loc[data["match_date"].le(reference.normalize())].copy()
    data = data.dropna(subset=["match_date"]).copy()
    if data.empty:
        return data

    canonical_keys = _tournament_key_series(data)
    years = data["match_date"].dt.year.astype("Int64").astype("string")
    ids = data.get("tourney_id", pd.Series("", index=data.index)).astype("string").fillna("").str.strip()
    data["_edition_key"] = ids.where(ids.ne(""), canonical_keys.astype("string") + "|" + years)
    edition_dates = (
        data.groupby("_edition_key", dropna=False)["match_date"]
        .max()
        .sort_values(ascending=False, kind="mergesort")
    )
    selected_keys = edition_dates.head(max(int(last_n_tournaments), 1)).index.tolist()
    rows: list[pd.Series] = []
    for recency_rank, edition_key in enumerate(selected_keys, start=1):
        edition = data.loc[data["_edition_key"].eq(edition_key)].copy()
        latest = edition.sort_values("match_date", kind="mergesort").iloc[-1]
        label = latest.get("canonical_tournament_name", latest.get("tourney_name", ""))
        if pd.isna(label) or not str(label).strip():
            label = latest.get("tourney_name", "")
        aggregate = _aggregate_tournament_match_window(edition, normalized, str(label))
        if aggregate is not None:
            aggregate["recency_rank"] = recency_rank
            rows.append(aggregate)
    return pd.DataFrame(rows).sort_values("recency_rank", kind="mergesort").reset_index(drop=True) if rows else pd.DataFrame()


def get_players_recent_tournament_statistics(
    player_1_id: Any,
    player_2_id: Any,
    last_n_tournaments: int = 3,
    as_of_date: Any = None,
) -> pd.DataFrame:
    """Recent-tournament aggregates for both selected players."""
    parts = [
        get_player_recent_tournament_statistics(player_id, last_n_tournaments, as_of_date)
        for player_id in (player_1_id, player_2_id)
    ]
    parts = [part for part in parts if not part.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def get_player_tournament_history(
    player_id: Any,
    tournament_name: str,
    tournament_id: Any = None,
    years: int = 10,
    as_of_date: Any = None,
    exclude_current_edition: bool = True,
) -> pd.DataFrame:
    """One row per season showing the furthest round at a tournament.

    Matching uses a canonical name key rather than literal text. Therefore
    provider labels such as ``U.S. Open - New York`` match historical labels
    such as ``US Open``. By default the edition in progress is excluded because
    its statistics are reported separately by the current-tournament section.
    """
    normalized = _normalize_player_id(player_id)
    target_key = _canonical_tournament_key(tournament_name)
    if normalized is None or not target_key:
        return pd.DataFrame()
    data = _load_player_matches_cached(normalized).copy()
    if data.empty:
        return pd.DataFrame()
    data["match_date"] = pd.to_datetime(data["match_date"], errors="coerce")
    data = data.loc[_tournament_key_series(data).eq(target_key)].copy()
    if data.empty:
        return pd.DataFrame()
    reference = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(reference):
        reference = pd.Timestamp.today().normalize()
    data = data.loc[data["match_date"].le(reference.normalize())].copy()
    data["season"] = data["match_date"].dt.year.astype("Int16")
    minimum_year = int(reference.year) - max(int(years), 1) + 1
    data = data.loc[data["season"].ge(minimum_year)].copy()
    if exclude_current_edition:
        current_mask = pd.Series(False, index=data.index, dtype=bool)
        if tournament_id not in (None, "") and "tourney_id" in data:
            current_mask |= (
                data["tourney_id"]
                .astype("string")
                .str.strip()
                .eq(str(tournament_id).strip())
            )
        # IDs from external fixture providers may not match historical IDs.
        # The current year is therefore the authoritative fallback.
        current_mask |= data["season"].eq(int(reference.year))
        data = data.loc[~current_mask].copy()
    if data.empty:
        return pd.DataFrame()
    data["_round_order"] = data["round"].map(_round_order).astype("Int16")
    rows = []
    for season, edition in data.groupby("season", sort=False, dropna=False):
        edition = edition.sort_values(
            ["_round_order", "match_num", "match_date"],
            kind="mergesort",
            na_position="last",
        )
        furthest = edition.iloc[-1]
        best_round = str(furthest.get("round", ""))
        won_value = pd.to_numeric(furthest.get("won"), errors="coerce")
        won_furthest = int(won_value) if pd.notna(won_value) else 0
        outcome = (
            "Champion"
            if best_round == "F" and won_furthest == 1
            else f"Reached {best_round}"
        )
        wins = int(pd.to_numeric(edition["won"], errors="coerce").fillna(0).sum())
        display_name = furthest.get(
            "canonical_tournament_name",
            furthest.get("tourney_name", tournament_name),
        )
        if pd.isna(display_name) or not str(display_name).strip():
            display_name = furthest.get("tourney_name", tournament_name)
        rows.append({
            "player_id": normalized,
            "player_name": str(furthest.get("player_name", "")),
            "season": int(season),
            "tournament": str(display_name),
            "tournament_key": target_key,
            "surface": str(furthest.get("surface", "")),
            "matches": int(len(edition)),
            "wins": wins,
            "losses": int(len(edition) - wins),
            "best_round": best_round,
            "best_round_order": int(furthest.get("_round_order", -1)),
            "outcome": outcome,
            "last_match_date": edition["match_date"].max(),
            "last_opponent": str(furthest.get("opponent_name", "")),
            "last_result": str(furthest.get("result", "")),
            "last_score": str(furthest.get("score", "")),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("season", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )

def get_players_tournament_history(
    player_1_id: Any,
    player_2_id: Any,
    tournament_name: str,
    tournament_id: Any = None,
    years: int = 10,
    as_of_date: Any = None,
    exclude_current_edition: bool = True,
) -> pd.DataFrame:
    """Comparable canonical tournament history for two selected players."""
    parts = [
        get_player_tournament_history(
            player_id=player_id,
            tournament_name=tournament_name,
            tournament_id=tournament_id,
            years=years,
            as_of_date=as_of_date,
            exclude_current_edition=exclude_current_edition,
        )
        for player_id in (player_1_id, player_2_id)
    ]
    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.DataFrame()
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(
            ["season", "player_name"],
            ascending=[False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Empirical set and total-games market analytics
# ---------------------------------------------------------------------------
def _beta_smoothed_probability(successes: Any, sample_size: Any, prior: float = 0.5, strength: float = 8.0) -> float:
    """Beta-binomial shrinkage. Small samples remain close to the neutral prior."""
    successes = pd.to_numeric(successes, errors="coerce")
    sample_size = pd.to_numeric(sample_size, errors="coerce")
    if pd.isna(successes) or pd.isna(sample_size) or float(sample_size) <= 0:
        return np.nan
    return float((float(successes) + prior * strength) / (float(sample_size) + strength))


def _wilson_bounds(successes: int, sample_size: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if sample_size <= 0:
        return np.nan, np.nan
    p = successes / sample_size
    denominator = 1.0 + z * z / sample_size
    centre = p + z * z / (2.0 * sample_size)
    adjustment = z * np.sqrt(p * (1.0 - p) / sample_size + z * z / (4.0 * sample_size * sample_size))
    return float(max(0.0, (centre - adjustment) / denominator)), float(min(1.0, (centre + adjustment) / denominator))


def _market_outcomes_for_player(data: pd.DataFrame, over_games_line: float) -> pd.DataFrame:
    """Add player-oriented set and game outcomes from completed score strings."""
    if data.empty:
        return data.copy()
    result = data.copy()
    totals = result.apply(_match_score_totals_for_player, axis=1, result_type="expand")
    totals.columns = ["sets_won", "sets_lost", "games_won", "games_lost"]
    result = pd.concat([result.reset_index(drop=True), totals.reset_index(drop=True)], axis=1)
    result["total_games"] = pd.to_numeric(result["games_won"], errors="coerce") + pd.to_numeric(result["games_lost"], errors="coerce")
    result["won_at_least_one_set"] = pd.to_numeric(result["sets_won"], errors="coerce").ge(1).astype("Int8")
    result["opponent_won_at_least_one_set"] = pd.to_numeric(result["sets_lost"], errors="coerce").ge(1).astype("Int8")
    result["over_games"] = result["total_games"].gt(float(over_games_line)).astype("Int8")
    valid_score = result[["sets_won", "sets_lost", "games_won", "games_lost"]].notna().all(axis=1)
    return result.loc[valid_score].copy()


def _similarity_distance(history: pd.DataFrame, opponent_state: pd.Series | None) -> pd.Series:
    """Distance between historical opponents and the upcoming opponent's current profile."""
    if history.empty or opponent_state is None:
        return pd.Series(np.inf, index=history.index, dtype=float)
    specifications = (
        ("opponent_elo_before", ("elo", "elo_before"), 100.0, 1.00),
        ("opponent_surface_elo_before", ("surface_elo_before",), 100.0, 1.20),
        ("opponent_service_points_won_ewma_long_before", ("service_points_won_ewma_long", "service_points_won_ewma_long_before", "service_points_won_pct"), 0.035, 0.85),
        ("opponent_return_points_won_ewma_long_before", ("return_points_won_ewma_long", "return_points_won_ewma_long_before", "return_points_won_pct"), 0.035, 0.85),
        ("opponent_form_last_10_before", ("form_last_10", "form_last_10_before"), 0.20, 0.55),
    )
    squared = pd.Series(0.0, index=history.index)
    weight = pd.Series(0.0, index=history.index)
    for historical_column, state_columns, scale, importance in specifications:
        if historical_column not in history:
            continue
        target = np.nan
        for state_column in state_columns:
            candidate = pd.to_numeric(opponent_state.get(state_column), errors="coerce")
            if pd.notna(candidate):
                target = float(candidate)
                break
        values = pd.to_numeric(history[historical_column], errors="coerce")
        valid = values.notna() & np.isfinite(target)
        squared = squared + (((values - target) / scale) ** 2).where(valid, 0.0) * importance
        weight = weight + valid.astype(float) * importance
    return np.sqrt(squared / weight.replace(0.0, np.nan)).fillna(np.inf)


def _summarize_market_sample(data: pd.DataFrame, scope: str, over_games_line: float) -> dict[str, Any]:
    if data.empty:
        return {"scope": scope, "sample_size": 0, "score_coverage": 0, "win_probability": np.nan, "win_set_probability": np.nan, "opponent_win_set_probability": np.nan, "over_games_probability": np.nan}
    scored = _market_outcomes_for_player(data, over_games_line)
    n = len(scored)
    if n == 0:
        return {"scope": scope, "sample_size": len(data), "score_coverage": 0, "win_probability": _beta_smoothed_probability(pd.to_numeric(data.get("won"), errors="coerce").sum(), len(data)), "win_set_probability": np.nan, "opponent_win_set_probability": np.nan, "over_games_probability": np.nan}
    set_wins = int(scored["won_at_least_one_set"].sum())
    opponent_set_wins = int(scored["opponent_won_at_least_one_set"].sum())
    overs = int(scored["over_games"].sum())
    lower, upper = _wilson_bounds(set_wins, n)
    return {
        "scope": scope,
        "sample_size": int(len(data)),
        "score_coverage": int(n),
        "win_probability": _beta_smoothed_probability(pd.to_numeric(data.get("won"), errors="coerce").sum(), len(data)),
        "win_set_probability": _beta_smoothed_probability(set_wins, n),
        "win_set_confidence_lower": lower,
        "win_set_confidence_upper": upper,
        "opponent_win_set_probability": _beta_smoothed_probability(opponent_set_wins, n),
        "over_games_line": float(over_games_line),
        "over_games_probability": _beta_smoothed_probability(overs, n),
        "average_total_games": float(pd.to_numeric(scored["total_games"], errors="coerce").mean()),
    }


def _combine_market_scopes(scopes: list[dict[str, Any]], probability_key: str) -> float:
    """Evidence-weighted blend with capped scope weights to avoid H2H overfitting."""
    scope_weights = {"H2H_SURFACE": 1.35, "H2H_ALL": 1.10, "SIMILAR_SURFACE": 1.00, "SIMILAR_ALL": 0.75, "RECENT_SURFACE": 0.60}
    numerator = 0.0
    denominator = 0.0
    for item in scopes:
        probability = pd.to_numeric(item.get(probability_key), errors="coerce")
        sample = pd.to_numeric(item.get("score_coverage", item.get("sample_size", 0)), errors="coerce")
        if pd.isna(probability) or pd.isna(sample) or sample <= 0:
            continue
        evidence = min(float(sample), 30.0) ** 0.5
        weight = scope_weights.get(str(item.get("scope")), 0.5) * evidence
        numerator += float(probability) * weight
        denominator += weight
    return numerator / denominator if denominator > 0 else np.nan


def get_match_market_probabilities(
    player_1_id: Any,
    player_2_id: Any,
    surface: str,
    best_of: int = 3,
    over_games_line: float | None = None,
    as_of_date: Any = None,
    similar_opponents: int = 40,
) -> dict[str, Any]:
    """Empirical H2H/comparable-opponent probabilities for set and games markets.

    These are descriptive, beta-smoothed historical estimates, not outputs from
    the match-winner model. Similarity uses opponent Elo, surface Elo and rolling
    serve/return/form features available before each historical match.
    """
    p1 = _normalize_player_id(player_1_id)
    p2 = _normalize_player_id(player_2_id)
    line = float(
        over_games_line
        if over_games_line is not None
        else (
            DEFAULT_OVER_GAMES_LINE_BO5
            if int(best_of) == 5
            else DEFAULT_OVER_GAMES_LINE_BO3
        )
    )
    if p1 is None or p2 is None or p1 == p2:
        return {"available": False, "reason": "invalid_player_ids", "over_games_line": line}
    normalized_surface = _normalize_surface(surface)
    reference = pd.to_datetime(as_of_date, errors="coerce")
    state_1 = get_player_surface_state(p1, normalized_surface)
    state_2 = get_player_surface_state(p2, normalized_surface)

    def player_scopes(player_id: int, opponent_id: int, opponent_state: pd.Series | None) -> list[dict[str, Any]]:
        history = _load_player_matches_cached(player_id).copy()
        if pd.notna(reference) and not history.empty:
            history = history.loc[pd.to_datetime(history["match_date"], errors="coerce").lt(reference.normalize())].copy()
        if history.empty:
            return []
        h2h = history.loc[pd.to_numeric(history["opponent_id"], errors="coerce").eq(opponent_id)].copy()
        same_surface = history.loc[history["surface"].astype(str).eq(normalized_surface)].copy()
        h2h_surface = h2h.loc[h2h["surface"].astype(str).eq(normalized_surface)].copy()
        pool = same_surface.copy()
        pool["_similarity_distance"] = _similarity_distance(pool, opponent_state)
        similar_surface = pool.loc[np.isfinite(pool["_similarity_distance"])].sort_values(["_similarity_distance", "match_date"], ascending=[True, False], kind="mergesort").head(max(int(similar_opponents), 1))
        all_pool = history.copy()
        all_pool["_similarity_distance"] = _similarity_distance(all_pool, opponent_state)
        similar_all = all_pool.loc[np.isfinite(all_pool["_similarity_distance"])].sort_values(["_similarity_distance", "match_date"], ascending=[True, False], kind="mergesort").head(max(int(similar_opponents), 1))
        recent_surface = same_surface.sort_values("match_date", ascending=False, kind="mergesort").head(20)
        return [
            _summarize_market_sample(h2h_surface, "H2H_SURFACE", line),
            _summarize_market_sample(h2h, "H2H_ALL", line),
            _summarize_market_sample(similar_surface, "SIMILAR_SURFACE", line),
            _summarize_market_sample(similar_all, "SIMILAR_ALL", line),
            _summarize_market_sample(recent_surface, "RECENT_SURFACE", line),
        ]

    p1_scopes = player_scopes(p1, p2, state_2)
    p2_scopes = player_scopes(p2, p1, state_1)
    p1_set = _combine_market_scopes(p1_scopes, "win_set_probability")
    p2_set = _combine_market_scopes(p2_scopes, "win_set_probability")
    over_values = [
        _combine_market_scopes(p1_scopes, "over_games_probability"),
        _combine_market_scopes(p2_scopes, "over_games_probability"),
    ]
    valid_over = [value for value in over_values if pd.notna(value)]
    h2h_rows = next((int(item.get("sample_size", 0)) for item in p1_scopes if item.get("scope") == "H2H_ALL"), 0)
    comparable_rows = next((int(item.get("score_coverage", 0)) for item in p1_scopes if item.get("scope") == "SIMILAR_SURFACE"), 0) + next((int(item.get("score_coverage", 0)) for item in p2_scopes if item.get("scope") == "SIMILAR_SURFACE"), 0)
    return {
        "available": bool(pd.notna(p1_set) or pd.notna(p2_set) or valid_over),
        "method": "beta_smoothed_h2h_and_similar_opponents",
        "surface": normalized_surface,
        "best_of": int(best_of),
        "over_games_line": line,
        "player_1_win_set_probability": p1_set,
        "player_2_win_set_probability": p2_set,
        "over_games_probability": float(np.mean(valid_over)) if valid_over else np.nan,
        "under_games_probability": float(1.0 - np.mean(valid_over)) if valid_over else np.nan,
        "h2h_matches": h2h_rows,
        "similar_surface_score_sample": comparable_rows,
        "player_1_scopes": p1_scopes,
        "player_2_scopes": p2_scopes,
        "limitations": "Historical descriptive estimate; no guarantee and not calibrated as a betting-market model.",
    }

def clear_player_statistics_cache() -> None:
    for function in (
        _history_path, _state_path, _available_columns, _history_dataset,
        _current_state_columns, _load_player_matches_cached, get_players,
        get_player_current_state, load_matches, load_player_per_match,
        load_player_snapshots, load_single_player_statistics,
        _pair_state_path, _pair_state_dataset, get_pair_inference_snapshot,
    ):
        function.cache_clear()
    gc.collect()
