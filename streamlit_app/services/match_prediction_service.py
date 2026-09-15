#!/usr/bin/env python3
"""Servicio de inferencia compartido por Upcoming Matches.

Carga el champion activo, construye las mismas features deportivas que Match
Prediction y devuelve probabilidades en el orden visual solicitado.
"""
from __future__ import annotations

import hashlib
from collections import Counter
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
import pyarrow.dataset as ds
from catboost import CatBoostClassifier, Pool

from streamlit_app.config.data_paths import (
    ACTIVE_MODEL_PATH,
    MATCH_HISTORY_PATH,
    PRODUCTION_ARTIFACTS_DIR,
)
from streamlit_app.data_access.players_adapted import (
    get_pair_inference_snapshot,
    get_player_inference_snapshot,
)

# Backward-compatible local name retained for the existing bundle loader.
PRODUCTION_ROOT = PRODUCTION_ARTIFACTS_DIR
SYNTHETIC_ID_MIN = 90_000_000


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _match_features_path() -> Path:
    if MATCH_HISTORY_PATH.is_file():
        return MATCH_HISTORY_PATH
    raise FileNotFoundError(
        f"No se encontro el historico canonico de features: {MATCH_HISTORY_PATH}"
    )


@lru_cache(maxsize=4)
def _load_bundle(active_mtime_ns: int) -> dict[str, Any]:
    del active_mtime_ns
    active = _read_json(ACTIVE_MODEL_PATH)
    artifact_dir = (PRODUCTION_ROOT / active["artifact_directory"]).resolve()
    metadata_path = artifact_dir / active["metadata_file"]
    model_path = artifact_dir / active["model_file"]
    manifest_path = artifact_dir / active.get("registry_manifest_file", "registry_manifest.json")
    metadata = _read_json(metadata_path)
    manifest = _read_json(manifest_path)

    for relative_name, expected_hash in manifest.get("hashes_sha256", {}).items():
        artifact = artifact_dir / relative_name
        if not artifact.exists() or _sha256(artifact) != str(expected_hash).upper():
            raise RuntimeError(f"Invalid production artifact: {relative_name}")

    champion = str(active["champion"]).strip().lower()
    policy = str(active["probability_policy"]).strip().lower()
    if champion in {"xgboost", "logistic"}:
        model = joblib.load(model_path)
        model_kind = "sklearn_pipeline"
    elif champion.startswith("catboost"):
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        model_kind = "catboost"
    else:
        raise ValueError(f"Unsupported champion: {champion}")

    calibrator = None
    if policy == "calibrated":
        calibrator_path = artifact_dir / "platt_calibrator.joblib"
        if not calibrator_path.exists():
            raise FileNotFoundError(calibrator_path)
        calibrator = joblib.load(calibrator_path)

    features = list(metadata.get("features", metadata.get("feature_columns", [])))
    categoricals = list(metadata.get("categoricals", metadata.get("categorical_columns", [])))
    if not features:
        raise ValueError("Production metadata does not declare features")

    market_model = None
    market_metadata: dict[str, Any] = {}
    market_model_name = active.get("market_model_file")
    market_metadata_name = active.get("market_metadata_file")
    if market_model_name:
        market_model_path = artifact_dir / str(market_model_name)
        if not market_model_path.exists():
            raise FileNotFoundError(market_model_path)
        market_model = joblib.load(market_model_path)
        if not market_model.get("features"):
            raise RuntimeError("Market model does not declare features")
        for key in ("player_1_serve_point_model", "player_2_serve_point_model"):
            if market_model.get(key) is None:
                raise RuntimeError(f"Market model is missing {key}")
    if market_metadata_name:
        market_metadata_path = artifact_dir / str(market_metadata_name)
        if not market_metadata_path.exists():
            raise FileNotFoundError(market_metadata_path)
        market_metadata = _read_json(market_metadata_path)

    # Supervised direct-market bundle. It is optional so the winner model and
    # the point-based simulator continue working during staged rollout.
    direct_market_model = None
    direct_market_metadata: dict[str, Any] = {}
    direct_model_name = active.get("direct_market_model_file")
    direct_metadata_name = active.get("direct_market_metadata_file")
    if direct_model_name:
        direct_model_path = artifact_dir / str(direct_model_name)
        if not direct_model_path.exists():
            raise FileNotFoundError(direct_model_path)
        direct_market_model = joblib.load(direct_model_path)
        required = {"features", "models", "calibrators"}
        missing_direct_keys = sorted(required - set(direct_market_model))
        if missing_direct_keys:
            raise RuntimeError(
                f"Direct-market bundle is incomplete: {missing_direct_keys}"
            )
    if direct_metadata_name:
        direct_metadata_path = artifact_dir / str(direct_metadata_name)
        if not direct_metadata_path.exists():
            raise FileNotFoundError(direct_metadata_path)
        direct_market_metadata = _read_json(direct_metadata_path)
    return {
        "active": active,
        "metadata": metadata,
        "model": model,
        "model_kind": model_kind,
        "calibrator": calibrator,
        "features": features,
        "categoricals": categoricals,
        "policy": policy,
        "market_model": market_model,
        "market_metadata": market_metadata,
        "direct_market_model": direct_market_model,
        "direct_market_metadata": direct_market_metadata,
    }


def _bundle() -> dict[str, Any]:
    if not ACTIVE_MODEL_PATH.exists():
        raise FileNotFoundError(ACTIVE_MODEL_PATH)
    return _load_bundle(ACTIVE_MODEL_PATH.stat().st_mtime_ns)


def _safe_numeric(value: Any) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _snapshot_value(snapshot: pd.Series | None, column: str, default: Any = np.nan) -> Any:
    return default if snapshot is None or column not in snapshot.index else snapshot[column]


def _coerce_binary_feature(value: Any, default: int = 0) -> int:
    """Convierte indicadores potencialmente nulos a un entero 0/1."""
    if value is None:
        return int(default)
    try:
        if pd.isna(value):
            return int(default)
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "y", "1"}:
            return 1
        if normalized in {"false", "no", "n", "0", ""}:
            return 0
    numeric = _safe_numeric(value)
    return int(numeric > 0) if pd.notna(numeric) else int(default)


def _first_snapshot_value(
    snapshot: pd.Series | None,
    columns: tuple[str, ...],
    default: Any = None,
) -> Any:
    """Devuelve el primer alias no nulo disponible en un snapshot."""
    if snapshot is None:
        return default
    for column in columns:
        if column not in snapshot.index:
            continue
        value = snapshot[column]
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        return value
    return default


def _player_cold_start(snapshot: pd.Series | None, minimum_matches: int = 5) -> int:
    """Calcula cold start usando flag explícito o historial de carrera."""
    explicit = _first_snapshot_value(
        snapshot,
        (
            "cold_start",
            "cold_start_before",
            "player_cold_start",
        ),
        default=None,
    )
    if explicit is not None:
        return _coerce_binary_feature(explicit)
    career_matches = _safe_numeric(
        _first_snapshot_value(
            snapshot,
            ("career_matches_before", "career_matches"),
            default=np.nan,
        )
    )
    return int(pd.isna(career_matches) or career_matches < minimum_matches)


def _player_is_synthetic(snapshot: pd.Series | None) -> int:
    """Detecta identidades sintéticas mediante flag explícito o rango de ID."""
    explicit = _first_snapshot_value(
        snapshot,
        ("synthetic_player", "synthetic_player_before"),
        default=None,
    )
    if explicit is not None and _coerce_binary_feature(explicit):
        return 1
    player_id = _safe_numeric(
        _first_snapshot_value(snapshot, ("player_id",), default=np.nan)
    )
    return int(pd.notna(player_id) and player_id >= SYNTHETIC_ID_MIN)


def _derive_pair_quality_features(
    player_1: pd.Series | None,
    player_2: pd.Series | None,
) -> dict[str, int]:
    """Deriva indicadores de calidad que pertenecen al partido, no a un jugador."""
    return {
        "cold_start_any_player": int(
            _player_cold_start(player_1) or _player_cold_start(player_2)
        ),
        "synthetic_player_any": int(
            _player_is_synthetic(player_1) or _player_is_synthetic(player_2)
        ),
    }


def _difference(left: Any, right: Any) -> float:
    left_value, right_value = _safe_numeric(left), _safe_numeric(right)
    return left_value - right_value if pd.notna(left_value) and pd.notna(right_value) else np.nan


def _ratio(left: Any, right: Any) -> float:
    left_value, right_value = _safe_numeric(left), _safe_numeric(right)
    if pd.isna(left_value) or pd.isna(right_value) or right_value == 0:
        return np.nan
    return float(np.clip(left_value / right_value, 0, 20))


def _elo_probability(value: Any) -> float:
    difference = _safe_numeric(value)
    return np.nan if pd.isna(difference) else float(1 / (1 + 10 ** (-difference / 400)))


def _feature_from_snapshot(snapshot: pd.Series, feature: str) -> Any:
    generic = feature
    for prefix in ("player_1_", "player_2_"):
        if generic.startswith(prefix):
            generic = generic[len(prefix):]
            break
    aliases = {
        "height": "player_height", "age": "player_age", "rank": "player_rank",
        "rank_points": "player_rank_points", "hand": "player_hand", "ioc": "player_ioc",
    }
    return _snapshot_value(snapshot, aliases.get(generic, generic))


def _quality_feature(feature: str, player_1: pd.Series, player_2: pd.Series) -> tuple[Any, str]:
    mappings = {
        "min_career_matches_before": "career_matches_before",
        "min_stat_matches_before": "stat_matches_before",
        "min_surface_matches_before": "surface_matches_before",
        "min_surface_stat_matches_before": "surface_stat_matches_before",
    }
    if feature in mappings:
        column = mappings[feature]
        values = [_safe_numeric(_snapshot_value(player_1, column)), _safe_numeric(_snapshot_value(player_2, column))]
        return (min(values) if all(pd.notna(value) for value in values) else np.nan), "minimum coverage"
    if feature == "cold_start_any_player":
        return _derive_pair_quality_features(player_1, player_2)[feature], "coverage flag"
    if feature == "surface_cold_start_any_player":
        return int(min(_safe_numeric(_snapshot_value(player_1, "surface_matches_before", 0)), _safe_numeric(_snapshot_value(player_2, "surface_matches_before", 0))) < 5), "coverage flag"
    if feature == "stats_reliable_both":
        return int(min(_safe_numeric(_snapshot_value(player_1, "stat_matches_before", 0)), _safe_numeric(_snapshot_value(player_2, "stat_matches_before", 0))) >= 10), "coverage flag"
    if feature == "surface_stats_reliable_both":
        return int(min(_safe_numeric(_snapshot_value(player_1, "surface_stat_matches_before", 0)), _safe_numeric(_snapshot_value(player_2, "surface_stat_matches_before", 0))) >= 10), "coverage flag"
    if feature == "synthetic_player_any":
        return _derive_pair_quality_features(player_1, player_2)[feature], "identity flag"
    return None, ""


def _derived_feature(feature: str, player_1: pd.Series, player_2: pd.Series) -> tuple[Any, str]:
    elo_diff = _difference(_snapshot_value(player_1, "elo_before"), _snapshot_value(player_2, "elo_before"))
    surface_diff = _difference(_snapshot_value(player_1, "surface_elo_before"), _snapshot_value(player_2, "surface_elo_before"))
    values = {
        "elo_expected_p1": _elo_probability(elo_diff),
        "surface_elo_expected_p1": _elo_probability(surface_diff),
        "absolute_elo_difference": abs(elo_diff) if pd.notna(elo_diff) else np.nan,
        "absolute_surface_elo_difference": abs(surface_diff) if pd.notna(surface_diff) else np.nan,
        "rank_ratio_p1_p2": _ratio(_snapshot_value(player_1, "player_rank"), _snapshot_value(player_2, "player_rank")),
        "rank_points_ratio_p1_p2": _ratio(_safe_numeric(_snapshot_value(player_1, "player_rank_points")) + 1, _safe_numeric(_snapshot_value(player_2, "player_rank_points")) + 1),
    }
    return (values.get(feature), "derived feature") if feature in values else (None, "")


@lru_cache(maxsize=512)
def _h2h(player_1_id: int, player_2_id: int) -> tuple[int, int, int]:
    path = _match_features_path()
    dataset = ds.dataset(str(path), format="parquet")
    names = set(dataset.schema.names)
    required = {"player_1_id", "player_2_id", "target_player_1_win"}
    if not required.issubset(names):
        return 0, 0, 0
    expression = (
        ((ds.field("player_1_id") == player_1_id) & (ds.field("player_2_id") == player_2_id))
        | ((ds.field("player_1_id") == player_2_id) & (ds.field("player_2_id") == player_1_id))
    )
    frame = dataset.to_table(columns=list(required), filter=expression).to_pandas()
    if frame.empty:
        return 0, 0, 0
    target = pd.to_numeric(frame["target_player_1_win"], errors="coerce")
    first_is_side_1 = pd.to_numeric(frame["player_1_id"], errors="coerce").eq(player_1_id)
    first_won = np.where(first_is_side_1, target.eq(1), target.eq(0))
    wins = int(np.asarray(first_won).sum())
    return len(frame), wins, len(frame) - wins


def _signed_log(value: Any) -> float:
    number = _safe_numeric(value)
    return float(np.sign(number) * np.log1p(abs(number))) if pd.notna(number) else np.nan


def _clipped(value: Any, lower: float = -500.0, upper: float = 500.0) -> float:
    number = _safe_numeric(value)
    return float(np.clip(number, lower, upper)) if pd.notna(number) else np.nan


def _availability(snapshot: pd.Series, column: str) -> int:
    return int(pd.notna(_snapshot_value(snapshot, column)))


def _matchup_derived(feature: str, pair: dict[str, Any]) -> Any:
    common = _safe_numeric(pair.get("common_opponents_count_before"))
    surface_common = _safe_numeric(pair.get("surface_common_opponents_count_before"))
    intransitivity = _safe_numeric(pair.get("intransitivity_score_player_1_before"))
    surface_intransitivity = _safe_numeric(
        pair.get("surface_intransitivity_score_player_1_before")
    )
    values = {
        "intransitivity_shrunk_before": (
            intransitivity * common / (common + 3.0)
            if pd.notna(intransitivity) and pd.notna(common) else np.nan
        ),
        "surface_intransitivity_shrunk_before": (
            surface_intransitivity * surface_common / (surface_common + 3.0)
            if pd.notna(surface_intransitivity) and pd.notna(surface_common)
            else np.nan
        ),
        "log_common_opponents_count_before": (
            float(np.log1p(common)) if pd.notna(common) else np.nan
        ),
        "log_surface_common_opponents_count_before": (
            float(np.log1p(surface_common)) if pd.notna(surface_common) else np.nan
        ),
    }
    return values.get(feature)


def _build_features(
    player_1: pd.Series,
    player_2: pd.Series,
    surface: str,
    tournament_level: str,
    round_name: str,
    best_of: int,
    competition_type: str,
    indoor: bool,
    pair: dict[str, Any],
    match_date_exact: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Construye exactamente el contrato de features declarado por metadata."""
    bundle = _bundle()
    features = bundle["features"]
    categoricals = bundle["categoricals"]
    direct_market_model = bundle.get("direct_market_model") or {}
    direct_features = [str(value) for value in direct_market_model.get("features", [])]
    build_features = list(dict.fromkeys([*features, *direct_features]))
    context = {
        "surface": surface,
        "tourney_level": tournament_level,
        "competition_type": competition_type,
        "indoor": str(bool(indoor)),
        "round": round_name,
        "best_of": str(best_of),
        "match_date_exact": int(bool(match_date_exact)),
    }
    h2h_matches = _safe_numeric(pair.get("h2h_matches_before", 0))
    p1_h2h_rate = _safe_numeric(pair.get("player_1_h2h_win_rate_before"))
    p1_h2h_wins = _safe_numeric(pair.get("player_1_h2h_wins_before"))
    p2_h2h_wins = _safe_numeric(pair.get("player_2_h2h_wins_before"))
    if pd.isna(p1_h2h_wins):
        p1_h2h_wins = h2h_matches * p1_h2h_rate if pd.notna(p1_h2h_rate) else 0.0
    if pd.isna(p2_h2h_wins):
        p2_h2h_wins = h2h_matches - p1_h2h_wins if pd.notna(h2h_matches) else 0.0
    row: dict[str, Any] = {}
    unresolved: list[str] = []
    for feature in build_features:
        value: Any = None
        if feature in context:
            value = context[feature]
        elif feature in pair:
            value = pair[feature]
        elif feature == "player_1_h2h_wins_before":
            value = p1_h2h_wins
        elif feature == "player_2_h2h_wins_before":
            value = p2_h2h_wins
        elif feature == "player_1_h2h_win_rate_before":
            value = p1_h2h_rate
        elif feature.startswith("diff_rank_change_") and feature.endswith("_clipped"):
            raw = feature.removesuffix("_clipped")
            value = _clipped(_difference(
                _feature_from_snapshot(player_1, f"player_1_{raw[len('diff_'):]}"),
                _feature_from_snapshot(player_2, f"player_2_{raw[len('diff_'):]}")
            ))
        elif feature.startswith("diff_rank_change_") and feature.endswith("_signed_log"):
            raw = feature.removesuffix("_signed_log")
            value = _signed_log(_difference(
                _feature_from_snapshot(player_1, f"player_1_{raw[len('diff_'):]}"),
                _feature_from_snapshot(player_2, f"player_2_{raw[len('diff_'):]}")
            ))
        elif feature.startswith("player_1_rank_change_") and feature.endswith("_clipped"):
            value = _clipped(_feature_from_snapshot(
                player_1, feature.removesuffix("_clipped")
            ))
        elif feature.startswith("player_2_rank_change_") and feature.endswith("_clipped"):
            value = _clipped(_feature_from_snapshot(
                player_2, feature.removesuffix("_clipped")
            ))
        elif feature == "dynamic_stats_available_player_1":
            value = _availability(player_1, "service_points_won_ewma_long_before")
        elif feature == "dynamic_stats_available_player_2":
            value = _availability(player_2, "service_points_won_ewma_long_before")
        elif feature == "dynamic_stats_available_both":
            value = min(
                _availability(player_1, "service_points_won_ewma_long_before"),
                _availability(player_2, "service_points_won_ewma_long_before"),
            )
        elif feature == "dynamic_stats_available_count":
            value = (
                _availability(player_1, "service_points_won_ewma_long_before")
                + _availability(player_2, "service_points_won_ewma_long_before")
            )
        elif feature == "rank_trend_available_player_1":
            value = _availability(player_1, "rank_change_8w_before")
        elif feature == "rank_trend_available_player_2":
            value = _availability(player_2, "rank_change_8w_before")
        elif feature == "rank_trend_available_both":
            value = min(
                _availability(player_1, "rank_change_8w_before"),
                _availability(player_2, "rank_change_8w_before"),
            )
        elif feature.startswith("diff_"):
            suffix = feature[len("diff_"):]
            value = _difference(
                _feature_from_snapshot(player_1, f"player_1_{suffix}"),
                _feature_from_snapshot(player_2, f"player_2_{suffix}"),
            )
        elif feature.startswith("player_1_"):
            value = _feature_from_snapshot(player_1, feature)
        elif feature.startswith("player_2_"):
            value = _feature_from_snapshot(player_2, feature)
        else:
            value, _ = _quality_feature(feature, player_1, player_2)
            if value is None:
                value, _ = _derived_feature(feature, player_1, player_2)
            if value is None:
                value = _matchup_derived(feature, pair)
        if feature in categoricals:
            row[feature] = (
                "__MISSING__" if value is None or pd.isna(value) else str(value)
            )
        else:
            row[feature] = _safe_numeric(value)
            if pd.isna(row[feature]):
                unresolved.append(feature)
    # Keep the champion columns first and retain additional direct-market
    # columns. The winner model still consumes frame[features] in exact order.
    frame = pd.DataFrame([row], columns=build_features)
    for feature in categoricals:
        frame[feature] = (
            frame[feature].astype("string").fillna("__MISSING__").astype(str)
        )

    # El modelo de mercados puede contener indicadores de partido que no forman
    # parte del contrato del champion principal. Se derivan explícitamente antes
    # de validar las features del bundle SPW.
    pair_quality = _derive_pair_quality_features(player_1, player_2)
    market_model = bundle.get("market_model") or {}
    market_features = {str(value) for value in market_model.get("features", [])}
    market_only_values = {
        name: value
        for name, value in pair_quality.items()
        if name in market_features and name not in frame.columns
    }
    if market_only_values:
        frame = frame.assign(**market_only_values)

    diagnostics = {
        "feature_count": len(features),
        "direct_market_feature_count": len(direct_features),
        "missing_numeric_count": len(unresolved),
        "missing_numeric_features": unresolved,
        "contract_exact": list(frame.loc[:, features].columns) == features,
        "pair_quality_features": pair_quality,
        "market_only_features_added": sorted(market_only_values),
    }
    if not diagnostics["contract_exact"]:
        raise RuntimeError("El orden de features no coincide con model_metadata.json")
    return frame, diagnostics


def _raw_probability(frame: pd.DataFrame) -> float:
    bundle = _bundle()
    model = bundle["model"]
    features = bundle["features"]
    if bundle["model_kind"] == "catboost":
        pool = Pool(frame[features], cat_features=bundle["categoricals"], feature_names=features)
        return float(model.predict_proba(pool)[0, 1])
    return float(model.predict_proba(frame[features])[0, 1])


def _apply_policy(raw_probability: float) -> float:
    bundle = _bundle()
    if bundle["policy"] != "calibrated":
        return raw_probability
    calibrator = bundle["calibrator"]
    values = np.asarray([raw_probability], dtype=float)
    if hasattr(calibrator, "predict"):
        return float(np.asarray(calibrator.predict(values), dtype=float)[0])
    result = np.asarray(calibrator.predict_proba(values), dtype=float)
    return float(result[0, -1])


def _numeric_market_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Build an ordered numeric market matrix, materializing absent fields as NaN.

    Direct-market HistGradientBoosting models and the structural market regressors
    were trained with native missing-value handling. A future fixture can lack
    tournament-edition fields, so absence must not cancel the complete prediction.
    """
    result = pd.DataFrame(index=frame.index)
    for feature in features:
        if feature in frame.columns:
            result[feature] = pd.to_numeric(
                frame[feature], errors="coerce"
            ).astype("float32")
        else:
            result[feature] = pd.Series(
                np.nan, index=frame.index, dtype="float32"
            )
    return result.loc[:, features].replace([np.inf, -np.inf], np.nan)


def _simulate_market_distribution(
    p1_serve_point: float,
    p2_serve_point: float,
    best_of: int,
    over_line: float,
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    """Point-to-match Monte Carlo using standard advantage games and 6-6 tiebreaks."""
    rng = np.random.default_rng(seed)
    sets_needed = int(best_of) // 2 + 1
    scores: Counter[str] = Counter()
    totals = np.empty(simulations, dtype=np.int16)
    p1_match = p1_set = p2_set = 0

    def point(server: int) -> int:
        probability = p1_serve_point if server == 1 else p2_serve_point
        return server if rng.random() < probability else (2 if server == 1 else 1)

    def game(server: int) -> int:
        a = b = 0
        while True:
            winner = point(server)
            a += int(winner == 1); b += int(winner == 2)
            if max(a, b) >= 4 and abs(a - b) >= 2:
                return 1 if a > b else 2

    def tiebreak(first_server: int) -> int:
        a = b = points = 0
        while True:
            if points == 0:
                server = first_server
            else:
                block = (points - 1) // 2
                server = (2 if first_server == 1 else 1) if block % 2 == 0 else first_server
            winner = point(server)
            a += int(winner == 1); b += int(winner == 2); points += 1
            if max(a, b) >= 7 and abs(a - b) >= 2:
                return 1 if a > b else 2

    for index in range(simulations):
        sets1 = sets2 = total = 0
        first_server = 1 if rng.random() < 0.5 else 2
        while sets1 < sets_needed and sets2 < sets_needed:
            games1 = games2 = 0
            server = first_server
            while True:
                winner = tiebreak(server) if games1 == games2 == 6 else game(server)
                games1 += int(winner == 1); games2 += int(winner == 2)
                if games1 == games2 == 6:
                    pass
                elif max(games1, games2) >= 6 and abs(games1 - games2) >= 2:
                    break
                elif max(games1, games2) == 7:
                    break
                server = 2 if server == 1 else 1
            total += games1 + games2
            sets1 += int(games1 > games2); sets2 += int(games2 > games1)
            first_server = 2 if server == 1 else 1
        totals[index] = total
        p1_match += int(sets1 > sets2)
        p1_set += int(sets1 > 0); p2_set += int(sets2 > 0)
        scores[f"{sets1}-{sets2}"] += 1
    over = float(np.mean(totals > float(over_line)))
    return {
        "available": True,
        "method": "serve_point_models_plus_hierarchical_monte_carlo",
        "simulations": int(simulations),
        "over_games_line": float(over_line),
        "match_win_player_1": p1_match / simulations,
        "player_1_win_set_probability": p1_set / simulations,
        "player_2_win_set_probability": p2_set / simulations,
        "expected_total_games": float(totals.mean()),
        "over_games_probability": over,
        "under_games_probability": 1.0 - over,
        "set_score_probabilities": {key: value / simulations for key, value in sorted(scores.items())},
    }


def _swap_direct_market_features(
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Apply the direct-training P1/P2 transformation without requiring fields."""
    output: dict[str, Any] = {}
    missing = pd.Series(np.nan, index=frame.index, dtype="float32")

    def source(column: str) -> pd.Series:
        if column not in frame.columns:
            return missing
        return pd.to_numeric(frame[column], errors="coerce")

    for feature in features:
        if feature.startswith("player_1_"):
            output[feature] = source("player_2_" + feature[len("player_1_"):])
        elif feature.startswith("player_2_"):
            output[feature] = source("player_1_" + feature[len("player_2_"):])
        elif feature.startswith("diff_"):
            output[feature] = -source(feature)
        elif feature.endswith("_p1_p2"):
            value = source(feature)
            output[feature] = pd.Series(
                np.where(value.ne(0), 1.0 / value, np.nan),
                index=frame.index,
                dtype="float32",
            )
        elif feature in {"elo_expected_p1", "surface_elo_expected_p1"}:
            output[feature] = 1.0 - source(feature)
        elif "_player_1_before" in feature:
            value = source(feature)
            if any(token in feature for token in ("rate", "share", "probability")):
                output[feature] = 1.0 - value
            elif any(token in feature for token in ("score", "intransitivity")):
                output[feature] = -value
            else:
                output[feature] = value
        else:
            output[feature] = source(feature)
    return _numeric_market_frame(pd.DataFrame(output, index=frame.index), features)

def _apply_direct_calibrator(calibrator: Any, probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    logit = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    return np.asarray(calibrator.predict_proba(logit)[:, 1], dtype=float)


def _predict_direct_markets(
    feature_frame: pd.DataFrame,
    best_of: int,
    first_is_model_side_1: bool,
) -> dict[str, Any]:
    """Predict first set, at-least-one-set and cumulative BO3 over markets."""
    bundle = _bundle()
    direct = bundle.get("direct_market_model")
    if not direct:
        return {"available": False, "reason": "direct_market_model_unavailable"}
    features = [str(value) for value in direct["features"]]
    original = _numeric_market_frame(feature_frame, features)
    swapped = _swap_direct_market_features(feature_frame, features)
    unmaterialized_features = sorted(set(features) - set(feature_frame.columns))
    all_missing_features = [
        feature for feature in features if original[feature].isna().all()
    ]
    available_feature_count = len(features) - len(all_missing_features)
    feature_coverage = (
        available_feature_count / len(features) if features else 0.0
    )
    minimum_feature_coverage = 0.70
    if feature_coverage < minimum_feature_coverage:
        return {
            "available": False,
            "reason": "insufficient_direct_market_feature_coverage",
            "feature_count": len(features),
            "available_feature_count": available_feature_count,
            "missing_feature_count": len(all_missing_features),
            "feature_coverage": float(feature_coverage),
            "missing_features": all_missing_features,
            "unmaterialized_features": unmaterialized_features,
        }
    models = direct.get("models", {})
    calibrators = direct.get("calibrators", {})

    def probability(market: str, matrix: pd.DataFrame) -> float:
        if market not in models or market not in calibrators:
            raise RuntimeError(f"Missing direct market model/calibrator: {market}")
        raw = models[market].predict_proba(matrix)[:, 1]
        return float(_apply_direct_calibrator(calibrators[market], raw)[0])

    # First set is complementary. Average both orientations to enforce exact
    # coherence and reduce small orientation noise.
    first_original = probability("first_set", original)
    first_swapped = probability("first_set", swapped)
    model_side_1_first = float(np.clip(
        (first_original + (1.0 - first_swapped)) / 2.0, 0.0, 1.0
    ))
    model_side_2_first = 1.0 - model_side_1_first

    # Winning at least one set is not complementary. The swapped prediction is
    # the probability for model side 2 as the focal player.
    model_side_1_any = probability("win_any_set", original)
    model_side_2_any = probability("win_any_set", swapped)

    if first_is_model_side_1:
        player_1_first, player_2_first = model_side_1_first, model_side_2_first
        player_1_any, player_2_any = model_side_1_any, model_side_2_any
    else:
        player_1_first, player_2_first = model_side_2_first, model_side_1_first
        player_1_any, player_2_any = model_side_2_any, model_side_1_any

    result: dict[str, Any] = {
        "available": True,
        "method": "calibrated_supervised_direct_market_models",
        "model_status": {
            "first_set": "provisional",
            "win_any_set": "validated_candidate",
        },
        "player_1_first_set_probability": float(player_1_first),
        "player_2_first_set_probability": float(player_2_first),
        "player_1_win_any_set_probability": float(player_1_any),
        "player_2_win_any_set_probability": float(player_2_any),
        # Backward-compatible names consumed by the existing Streamlit page.
        "player_1_win_set_probability": float(player_1_any),
        "player_2_win_set_probability": float(player_2_any),
        "first_set_symmetry_absolute_error": float(
            abs(first_original - (1.0 - first_swapped))
        ),
        "feature_count": len(features),
        "available_feature_count": available_feature_count,
        "missing_feature_count": len(all_missing_features),
        "feature_coverage": float(feature_coverage),
        "missing_features": all_missing_features,
        "unmaterialized_features": unmaterialized_features,
    }

    if int(best_of) == 3:
        lines = [float(value) for value in direct.get(
            "over_lines_bo3", [18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5]
        )]
        raw_over = []
        for line in lines:
            market = f"over_{line:.1f}"
            p_original = probability(market, original)
            p_swapped = probability(market, swapped)
            raw_over.append((p_original + p_swapped) / 2.0)
        monotonic = np.minimum.accumulate(np.asarray(raw_over, dtype=float))
        result["over_probabilities"] = {
            f"{line:.1f}": float(value) for line, value in zip(lines, monotonic)
        }
        result["over_model_status"] = {
            "18.5": "provisional",
            "19.5": "experimental",
            "20.5": "experimental",
            "21.5": "experimental",
            "22.5": "experimental",
            "23.5": "not_validated",
            "24.5": "not_validated",
            "25.5": "not_validated",
        }
    else:
        result["over_probabilities"] = {}
        result["over_model_status"] = {}
        result["over_reason"] = "direct_over_models_are_bo3_only"
    return result


def _select_direct_over(
    direct: dict[str, Any],
    requested_line: float,
) -> tuple[float, float]:
    probabilities = direct.get("over_probabilities", {})
    if not probabilities:
        return float(requested_line), np.nan
    available = sorted(float(value) for value in probabilities)
    selected = min(available, key=lambda value: abs(value - float(requested_line)))
    return selected, float(probabilities[f"{selected:.1f}"])


def _predict_model_markets(
    feature_frame: pd.DataFrame,
    best_of: int,
    over_games_line: float,
    simulations: int | None,
    first_is_model_side_1: bool,
    seed: int,
) -> dict[str, Any]:
    """Calcula mercados secundarios y adjunta una auditoria completa del SPW."""

    bundle = _bundle()
    market = bundle.get("market_model")
    if not market:
        return {"available": False, "reason": "market_model_unavailable"}

    features = [str(value) for value in market["features"]]

    # Compatibility guard for candidates trained with pair-level quality flags.
    # Normally these columns are already added by _build_features.
    missing_pair_flags = [
        feature
        for feature in ("cold_start_any_player", "synthetic_player_any")
        if feature in features and feature not in feature_frame.columns
    ]
    if missing_pair_flags:
        raise RuntimeError(
            "Missing derived pair-quality market features: "
            f"{missing_pair_flags}. The inference frame was not enriched."
        )

    numeric = _numeric_market_frame(feature_frame, features)

    raw1 = float(
        market["player_1_serve_point_model"].predict(numeric)[0]
    )
    raw2 = float(
        market["player_2_serve_point_model"].predict(numeric)[0]
    )

    correction1 = np.asarray(
        market.get("player_1_calibration_linear", [1.0, 0.0]),
        dtype=float,
    )
    correction2 = np.asarray(
        market.get("player_2_calibration_linear", [1.0, 0.0]),
        dtype=float,
    )
    bounds = np.asarray(
        market.get("serve_point_probability_bounds", [0.40, 0.82]),
        dtype=float,
    )

    if correction1.shape != (2,):
        raise RuntimeError(
            "Invalid player 1 SPW correction: "
            f"{correction1.tolist()}"
        )
    if correction2.shape != (2,):
        raise RuntimeError(
            "Invalid player 2 SPW correction: "
            f"{correction2.tolist()}"
        )
    if bounds.shape != (2,):
        raise RuntimeError(
            f"Invalid SPW bounds: {bounds.tolist()}"
        )

    lower_bound = float(bounds[0])
    upper_bound = float(bounds[1])
    if not (0.0 < lower_bound < upper_bound < 1.0):
        raise RuntimeError(
            f"Invalid SPW bounds: {bounds.tolist()}"
        )

    corrected1 = float(np.polyval(correction1, raw1))
    corrected2 = float(np.polyval(correction2, raw2))
    spw1 = float(np.clip(corrected1, lower_bound, upper_bound))
    spw2 = float(np.clip(corrected2, lower_bound, upper_bound))

    missing_features = [
        str(column)
        for column in features
        if numeric[column].isna().any()
    ]
    feature_count = int(len(features))
    missing_feature_count = int(len(missing_features))
    missing_feature_rate = (
        float(missing_feature_count / feature_count)
        if feature_count
        else float("nan")
    )

    boundary_tolerance = 0.005

    def side_audit(
        raw_probability: float,
        corrected_probability: float,
        final_probability: float,
        correction: np.ndarray,
    ) -> dict[str, Any]:
        at_lower_bound = bool(
            final_probability <= lower_bound + boundary_tolerance
        )
        at_upper_bound = bool(
            final_probability >= upper_bound - boundary_tolerance
        )
        return {
            "raw_probability": float(raw_probability),
            "corrected_probability": float(corrected_probability),
            "final_probability": float(final_probability),
            "correction_slope": float(correction[0]),
            "correction_intercept": float(correction[1]),
            "clipping_applied": bool(
                not np.isclose(
                    corrected_probability,
                    final_probability,
                    rtol=0.0,
                    atol=1e-12,
                )
            ),
            "at_lower_bound": at_lower_bound,
            "at_upper_bound": at_upper_bound,
            "distance_to_lower_bound": float(
                final_probability - lower_bound
            ),
            "distance_to_upper_bound": float(
                upper_bound - final_probability
            ),
        }

    model_side_1_audit = side_audit(
        raw1,
        corrected1,
        spw1,
        correction1,
    )
    model_side_2_audit = side_audit(
        raw2,
        corrected2,
        spw2,
        correction2,
    )

    spw_audit: dict[str, Any] = {
        "model_side_player_1": model_side_1_audit,
        "model_side_player_2": model_side_2_audit,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "boundary_tolerance": boundary_tolerance,
        "boundary_warning": bool(
            model_side_1_audit["at_lower_bound"]
            or model_side_1_audit["at_upper_bound"]
            or model_side_2_audit["at_lower_bound"]
            or model_side_2_audit["at_upper_bound"]
        ),
        "feature_count": feature_count,
        "missing_feature_count": missing_feature_count,
        "missing_feature_rate": missing_feature_rate,
        "missing_features": missing_features,
    }

    if first_is_model_side_1:
        spw_audit["screen_player_1"] = dict(model_side_1_audit)
        spw_audit["screen_player_2"] = dict(model_side_2_audit)
    else:
        spw_audit["screen_player_1"] = dict(model_side_2_audit)
        spw_audit["screen_player_2"] = dict(model_side_1_audit)

    n = int(simulations or market.get("recommended_simulations", 20_000))
    n = max(1_000, min(n, 100_000))
    result = _simulate_market_distribution(
        spw1,
        spw2,
        best_of,
        over_games_line,
        n,
        seed,
    )

    if first_is_model_side_1:
        result["player_1_serve_point_win_probability"] = spw1
        result["player_2_serve_point_win_probability"] = spw2
    else:
        result["player_1_win_set_probability"], result[
            "player_2_win_set_probability"
        ] = (
            result["player_2_win_set_probability"],
            result["player_1_win_set_probability"],
        )
        result["match_win_player_1"] = (
            1.0 - result["match_win_player_1"]
        )
        result["player_1_serve_point_win_probability"] = spw2
        result["player_2_serve_point_win_probability"] = spw1
        result["set_score_probabilities"] = {
            "-".join(reversed(score.split("-"))): probability
            for score, probability in result[
                "set_score_probabilities"
            ].items()
        }

    result["spw_audit"] = spw_audit
    result["market_model_version"] = bundle["active"].get(
        "active_model_version"
    )
    return result

def predict_upcoming_fixture(
    player_1_id: int,
    player_2_id: int,
    surface: str,
    tournament_level: str,
    round_name: str,
    best_of: int,
    competition_type: str,
    indoor: bool,
    prediction_date: Any = None,
    tournament_id: str | None = None,
    tournament_name: str | None = None,
    over_games_line: float | None = None,
    market_simulations: int | None = None,
) -> dict[str, Any]:
    """Devuelve las probabilidades en el mismo orden que el fixture."""
    try:
        player_1_id, player_2_id = int(player_1_id), int(player_2_id)
        if player_1_id == player_2_id:
            return {"available": False, "reason": "same_player"}

        first_is_model_side_1 = player_1_id < player_2_id
        model_id_1, model_id_2 = (
            (player_1_id, player_2_id)
            if first_is_model_side_1
            else (player_2_id, player_1_id)
        )
        model_player_1 = get_player_inference_snapshot(
            model_id_1, surface, tournament_level, round_name, best_of,
            prediction_date=prediction_date,
            competition_type=competition_type,
            indoor=indoor,
            tournament_id=tournament_id,
            tournament_name=tournament_name,
        )
        model_player_2 = get_player_inference_snapshot(
            model_id_2, surface, tournament_level, round_name, best_of,
            prediction_date=prediction_date,
            competition_type=competition_type,
            indoor=indoor,
            tournament_id=tournament_id,
            tournament_name=tournament_name,
        )
        if model_player_1 is None or model_player_2 is None:
            return {"available": False, "reason": "missing_player_state"}
        model_player_1, model_player_2 = model_player_1.copy(), model_player_2.copy()
        model_player_1["player_id"], model_player_2["player_id"] = model_id_1, model_id_2

        pair = get_pair_inference_snapshot(model_id_1, model_id_2, surface)
        frame, diagnostics = _build_features(
            model_player_1,
            model_player_2,
            surface,
            tournament_level,
            round_name,
            best_of,
            competition_type,
            indoor,
            pair,
            match_date_exact=prediction_date is not None,
        )
        probability_model_side_1 = float(np.clip(_apply_policy(_raw_probability(frame)), 1e-7, 1 - 1e-7))
        probability_player_1 = probability_model_side_1 if first_is_model_side_1 else 1 - probability_model_side_1
        selected_line = float(over_games_line if over_games_line is not None else (38.5 if int(best_of) == 5 else 22.5))
        deterministic_seed = int(hashlib.sha256(
            f"{model_id_1}|{model_id_2}|{surface}|{best_of}|{prediction_date}|{selected_line}".encode("utf-8")
        ).hexdigest()[:8], 16)
        # Structural simulator remains available as an independent baseline.
        structural_market_probabilities = _predict_model_markets(
            feature_frame=frame, best_of=int(best_of), over_games_line=selected_line,
            simulations=market_simulations, first_is_model_side_1=first_is_model_side_1,
            seed=deterministic_seed,
        )
        direct_market_probabilities = _predict_direct_markets(
            feature_frame=frame,
            best_of=int(best_of),
            first_is_model_side_1=first_is_model_side_1,
        )
        if direct_market_probabilities.get("available"):
            direct_line, direct_over = _select_direct_over(
                direct_market_probabilities, selected_line
            )
            market_probabilities = {
                **direct_market_probabilities,
                "over_games_line": direct_line,
                "over_games_probability": direct_over,
                "under_games_probability": (
                    1.0 - direct_over if np.isfinite(direct_over) else np.nan
                ),
                "structural_baseline": structural_market_probabilities,
            }
        else:
            market_probabilities = {
                **structural_market_probabilities,
                "direct_market_model": direct_market_probabilities,
            }
        bundle = _bundle()
        return {
            "available": True,
            "probability_player_1": float(probability_player_1),
            "probability_player_2": float(1 - probability_player_1),
            "model_version": bundle["active"].get("active_model_version"),
            "champion": bundle["active"].get("champion"),
            "probability_policy": bundle["policy"],
            "feature_diagnostics": diagnostics,
            "market_probabilities": market_probabilities,
            "tournament_id": tournament_id,
            "tournament_name": tournament_name,
        }
    except Exception as error:
        return {"available": False, "reason": "prediction_error", "error": str(error)}


# ---------------------------------------------------------------------------
# Tennis-API market odds integration
# ---------------------------------------------------------------------------
def _odds_normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _odds_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) and number > 1.0 else np.nan


def _odds_candidates(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("matches", "odds", "data", "result", "results", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = _odds_candidates(value)
            if nested:
                return nested
    return []


def _nested_value(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current if current not in (None, "") else default


def _match_names(item: dict[str, Any]) -> tuple[str, str]:
    left = next(
        (
            _nested_value(item, path)
            for path in ("player1.name", "player_1.name", "participant1", "home.name")
            if _nested_value(item, path) not in (None, "")
        ),
        "",
    )
    right = next(
        (
            _nested_value(item, path)
            for path in ("player2.name", "player_2.name", "participant2", "away.name")
            if _nested_value(item, path) not in (None, "")
        ),
        "",
    )
    return str(left), str(right)


def _same_players(
    item: dict[str, Any],
    player_1_name: str,
    player_2_name: str,
) -> tuple[bool, bool]:
    item_1, item_2 = _match_names(item)
    a, b = _odds_normalize_text(item_1), _odds_normalize_text(item_2)
    p1, p2 = _odds_normalize_text(player_1_name), _odds_normalize_text(player_2_name)
    if a == p1 and b == p2:
        return True, False
    if a == p2 and b == p1:
        return True, True
    return False, False


def _extract_match_prices(item: dict[str, Any]) -> tuple[float, float]:
    """Extract k1/k2 or player1.odd/player2.odd from documented payloads."""
    odds = item.get("odds") if isinstance(item.get("odds"), dict) else {}
    first = next(
        (
            _odds_number(value)
            for value in (
                _nested_value(item, "player1.odd"),
                _nested_value(item, "player1.odds"),
                odds.get("k1"),
                item.get("k1"),
            )
            if np.isfinite(_odds_number(value))
        ),
        np.nan,
    )
    second = next(
        (
            _odds_number(value)
            for value in (
                _nested_value(item, "player2.odd"),
                _nested_value(item, "player2.odds"),
                odds.get("k2"),
                item.get("k2"),
            )
            if np.isfinite(_odds_number(value))
        ),
        np.nan,
    )
    return first, second


def _provider_result(
    value: Any,
    *,
    reason: str,
    source: str,
    **extra: Any,
) -> dict[str, Any]:
    """Normalize provider outputs so public integration functions never return None."""
    if isinstance(value, dict):
        return value
    return {
        "available": False,
        "reason": reason,
        "source": source,
        "returned_type": type(value).__name__,
        **extra,
    }


def _request_json(
    url: str,
    api_key: str,
    host: str,
    params: dict[str, Any] | None = None,
) -> Any:
    response = requests.get(
        url,
        headers={
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": host,
            "Accept": "application/json",
        },
        params=params,
        timeout=30,
    )
    if response.status_code in {401, 403}:
        raise PermissionError(
            "The Tennis API endpoint is not enabled for the configured "
            f"RapidAPI plan/host (HTTP {response.status_code})."
        )
    # Keep HTTP 429 as requests.HTTPError. Upcoming Matches uses the status code
    # to activate SofaScore and stop further Tennis API calls for that action.
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as error:
        raise requests.JSONDecodeError(
            "Tennis API returned a non-JSON response",
            response.text,
            0,
        ) from error


def fetch_upcoming_market_odds(
    *,
    player_1_name: str,
    player_2_name: str,
    match_date: Any,
    tournament: str,
    api_key: str,
    host: str,
    tour_type: str = "atp",
    tournament_id: Any = None,
    round_id: Any = None,
    provider_player_1_id: Any = None,
    provider_player_2_id: Any = None,
) -> dict[str, Any]:
    """Fetch pre-match winner odds and return them in screen player order.

    Uses the documented MS Upcoming API first, whose payload exposes
    player1.odd/player2.odd and odds.k1/odds.k2. If provider IDs are available,
    it then tries the dedicated matchodds endpoint. No model code depends on
    this function.
    """
    if not api_key:
        return {"available": False, "reason": "api_key_missing"}

    requested_date = pd.to_datetime(match_date, errors="coerce")
    base = f"https://{host}/tennis/v2/ms-api/upcoming"
    diagnostics: list[str] = []

    # 1. Upcoming matches includes odds in the documented response and lets us
    # match safely by names, date and tournament without relying on Live IDs.
    try:
        payload = _request_json(
            f"{base}/matches/{str(tour_type).lower()}",
            api_key,
            host,
            params={"limit": 500, "tournament": tournament},
        )
        for item in _odds_candidates(payload):
            matched, reversed_order = _same_players(item, player_1_name, player_2_name)
            if not matched:
                continue
            item_date = pd.to_datetime(
                item.get("date", item.get("startTime", item.get("startTimestamp"))),
                errors="coerce",
                utc=True,
            )
            if pd.notna(requested_date) and pd.notna(item_date):
                if item_date.tz_convert(None).date() != requested_date.date():
                    continue
            first, second = _extract_match_prices(item)
            if reversed_order:
                first, second = second, first
            if np.isfinite(first) or np.isfinite(second):
                return {
                    "available": True,
                    "odds_player_1": first,
                    "odds_player_2": second,
                    "source": "ms_api_upcoming_matches",
                    "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    "provider_match": item,
                }
        diagnostics.append("upcoming_match_found_without_prices_or_not_matched")
    except PermissionError as error:
        return {"available": False, "reason": "odds_plan_restricted", "error": str(error)}
    except requests.RequestException as error:
        diagnostics.append(f"upcoming_matches_http_error:{error}")
    except Exception as error:
        diagnostics.append(f"upcoming_matches_parse_error:{error}")

    # 2. Dedicated endpoint, using provider IDs captured from Fixtures whenever
    # available. Its documented response is odds[].k1 and odds[].k2.
    filters: dict[str, Any] = {}
    for key, value in (
        ("tournamentId", tournament_id),
        ("roundId", round_id),
        ("player1Id", provider_player_1_id),
        ("player2Id", provider_player_2_id),
    ):
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric):
            filters[key] = int(numeric)

    if filters:
        try:
            payload = _request_json(
                f"{base}/matchodds/{str(tour_type).lower()}",
                api_key,
                host,
                params=filters,
            )
            items = _odds_candidates(payload)
            for item in items:
                first, second = _extract_match_prices(item)
                returned_p1 = str(item.get("id1_o", ""))
                returned_p2 = str(item.get("id2_o", ""))
                requested_p1 = str(int(float(provider_player_1_id))) if pd.notna(pd.to_numeric(provider_player_1_id, errors="coerce")) else ""
                requested_p2 = str(int(float(provider_player_2_id))) if pd.notna(pd.to_numeric(provider_player_2_id, errors="coerce")) else ""
                if requested_p1 and returned_p1 == requested_p2 and returned_p2 == requested_p1:
                    first, second = second, first
                if np.isfinite(first) or np.isfinite(second):
                    return {
                        "available": True,
                        "odds_player_1": first,
                        "odds_player_2": second,
                        "source": "ms_api_matchodds",
                        "bookmaker_id": item.get("id_b_o"),
                        "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                        "provider_odds": item,
                    }
            diagnostics.append("matchodds_returned_no_prices")
        except PermissionError as error:
            return {"available": False, "reason": "odds_plan_restricted", "error": str(error)}
        except requests.RequestException as error:
            diagnostics.append(f"matchodds_http_error:{error}")
        except Exception as error:
            diagnostics.append(f"matchodds_parse_error:{error}")

    return {
        "available": False,
        "reason": "market_odds_unavailable",
        "diagnostics": diagnostics,
        "filters": filters,
    }


# ---------------------------------------------------------------------------
# Tennis-API score, extend-event and odds-movement integration
# ---------------------------------------------------------------------------
def _api_objects(payload: Any) -> list[dict[str, Any]]:
    """Flatten common Tennis-API response envelopes into record dictionaries."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "result", "results", "items", "matches", "events", "odds"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = _api_objects(value)
            if nested:
                return nested
    return [payload] if payload else []


def _api_first(record: dict[str, Any], paths: tuple[str, ...], default: Any = None) -> Any:
    for path in paths:
        value = _nested_value(record, path, None)
        if value not in (None, ""):
            return value
    return default


def fetch_tournament_results(
    *,
    season_id: Any,
    api_key: str,
    host: str,
    tour_type: str = "atp",
) -> list[dict[str, Any]]:
    """Return the completed-result archive for one tournament edition."""
    numeric = pd.to_numeric(season_id, errors="coerce")
    if pd.isna(numeric):
        raise ValueError(f"Invalid tournament season id: {season_id!r}")
    payload = _request_json(
        f"https://{host}/tennis/v2/{tour_type.lower()}/tournament/results/{int(numeric)}",
        api_key,
        host,
    )
    return _api_objects(payload)


def _score_component(value: Any) -> str:
    """Render scalar or structured score values without leaking Python dicts."""
    if value in (None, ""):
        return ""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return str(int(value)) if float(value).is_integer() else str(value)
    if isinstance(value, str):
        return value.strip()
    return ""


def _score_from_result_record(item: dict[str, Any]) -> str:
    direct = _api_first(
        item,
        ("result", "score", "finalScore", "final_score", "displayScore"),
        "",
    )
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    home_score = _api_first(item, ("homeScore", "home_score"), {})
    away_score = _api_first(item, ("awayScore", "away_score"), {})
    if isinstance(home_score, dict) and isinstance(away_score, dict):
        home_periods = home_score.get("periods", {})
        away_periods = away_score.get("periods", {})
        if isinstance(home_periods, dict) and isinstance(away_periods, dict):
            sets: list[str] = []
            period_keys = sorted(
                {
                    key for key in [*home_periods.keys(), *away_periods.keys()]
                    if re.fullmatch(r"period\d+", str(key))
                },
                key=lambda key: int(re.search(r"\d+", str(key)).group()),
            )
            for key in period_keys:
                home = _score_component(home_periods.get(key))
                away = _score_component(away_periods.get(key))
                if home and away:
                    token = f"{home}-{away}"
                    tie_break = _score_component(
                        away_periods.get(f"{key}_tie_break")
                        or home_periods.get(f"{key}_tie_break")
                    )
                    if tie_break:
                        token += f"({tie_break})"
                    sets.append(token)
            if sets:
                return " ".join(sets)

    # Some Tennis API payloads expose sets as a list of objects.
    sets_value = _api_first(item, ("sets", "periods", "score.sets"), [])
    if isinstance(sets_value, list):
        sets: list[str] = []
        for set_item in sets_value:
            if not isinstance(set_item, dict):
                continue
            left = _api_first(
                set_item,
                ("player1", "player_1", "home", "homeScore", "score1", "first"),
                None,
            )
            right = _api_first(
                set_item,
                ("player2", "player_2", "away", "awayScore", "score2", "second"),
                None,
            )
            left_text, right_text = _score_component(left), _score_component(right)
            if left_text and right_text:
                sets.append(f"{left_text}-{right_text}")
        if sets:
            return " ".join(sets)
    return ""


def normalize_tournament_result(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize one tournament result across Tennis API response versions."""
    if not isinstance(item, dict):
        return {
            "player_1_name": "",
            "player_2_name": "",
            "winner_name": "",
            "status": "Unknown",
            "score": "",
            "round": "",
            "provider_result": item,
        }

    player_1 = _api_first(
        item,
        (
            "player1.name", "player_1.name", "homeTeam.name", "home_team.name",
            "home.name", "participant1.name", "participant1", "player1",
        ),
        "",
    )
    player_2 = _api_first(
        item,
        (
            "player2.name", "player_2.name", "awayTeam.name", "away_team.name",
            "away.name", "participant2.name", "participant2", "player2",
        ),
        "",
    )
    if isinstance(player_1, dict):
        player_1 = _api_first(player_1, ("name", "shortName", "fullName"), "")
    if isinstance(player_2, dict):
        player_2 = _api_first(player_2, ("name", "shortName", "fullName"), "")

    winner = _api_first(
        item,
        ("winner.name", "winnerName", "winner_name"),
        "",
    )
    if isinstance(winner, dict):
        winner = _api_first(winner, ("name", "shortName"), "")
    winner_code = pd.to_numeric(
        _api_first(item, ("winnerCode", "winner_code", "winner"), None),
        errors="coerce",
    )
    if not str(winner or "").strip() and pd.notna(winner_code):
        if int(winner_code) == 1:
            winner = player_1
        elif int(winner_code) == 2:
            winner = player_2

    status_value = _api_first(
        item,
        ("status.description", "status.type", "status", "matchStatus", "state"),
        "Ended",
    )
    if isinstance(status_value, dict):
        status_value = _api_first(status_value, ("description", "type", "name"), "Ended")
    status = str(status_value or "Ended")

    return {
        "player_1_name": str(player_1 or "").strip(),
        "player_2_name": str(player_2 or "").strip(),
        "winner_name": str(winner or "").strip(),
        "status": status,
        "status_type": str(
            _api_first(item, ("status.type", "statusType", "state"), status)
        ).casefold(),
        "score": _score_from_result_record(item),
        "round": str(
            _api_first(item, ("round.name", "round", "roundName"), "")
        ),
        "event_id": _api_first(item, ("id", "eventId", "event_id"), None),
        "provider_result": item,
    }

def fetch_event_details_by_players_date(
    *, player_1_name: str, player_2_name: str, match_date: Any,
    api_key: str, host: str,
) -> dict[str, Any]:
    """Resolve the Extend event and its event id using players and date."""
    from urllib.parse import quote
    date_text = pd.to_datetime(match_date, errors="raise").date().isoformat()
    payload = _request_json(
        f"https://{host}/tennis/v2/extend/api/event/get/"
        f"{quote(str(player_1_name), safe='')}/{quote(str(player_2_name), safe='')}/{date_text}",
        api_key,
        host,
    )
    if not isinstance(payload, dict):
        return _provider_result(
            payload,
            reason="event_payload_not_object",
            source="tennis_api_event",
        )
    result = payload.get("result", payload)
    if not isinstance(result, dict) or not result:
        return {
            "available": False,
            "reason": "event_not_found",
            "source": "tennis_api_event",
            "event_id": None,
            "event": result if isinstance(result, dict) else {},
        }
    event_id = _api_first(result, ("id", "eventId", "event_id"), None)
    return {
        "available": event_id not in (None, ""),
        "reason": None if event_id not in (None, "") else "event_id_missing",
        "source": "tennis_api_event",
        "event_id": str(event_id) if event_id not in (None, "") else None,
        "status": _api_first(result, ("status", "state"), None),
        "score": _api_first(result, ("score", "result"), None),
        "event": result,
    }


def _walk_api_values(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_api_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_api_values(child)


def _best_match_winner_prices(payload: Any, player_1_name: str, player_2_name: str) -> dict[str, Any]:
    p1 = _odds_normalize_text(player_1_name)
    p2 = _odds_normalize_text(player_2_name)
    first: list[tuple[float, str]] = []
    second: list[tuple[float, str]] = []
    for item in _walk_api_values(payload):
        label = _api_first(item, ("name", "label", "selection", "outcome", "participant", "runner"), "")
        if isinstance(label, dict):
            label = label.get("name", label.get("label", ""))
        price = np.nan
        for key in ("odds", "odd", "price", "decimal", "decimalOdds", "value"):
            if key in item:
                price = _odds_number(item.get(key))
                if np.isfinite(price):
                    break
        if not np.isfinite(price):
            continue
        bookmaker = str(_api_first(item, ("bookmaker.name", "bookmakerName", "bookmaker", "provider"), ""))
        normalized = _odds_normalize_text(label)
        if p1 and (normalized == p1 or p1 in normalized or normalized in p1):
            first.append((float(price), bookmaker))
        elif p2 and (normalized == p2 or p2 in normalized or normalized in p2):
            second.append((float(price), bookmaker))
    best1=max(first, default=(np.nan,""), key=lambda x:x[0])
    best2=max(second, default=(np.nan,""), key=lambda x:x[0])
    return {"available": bool(np.isfinite(best1[0]) or np.isfinite(best2[0])),
            "odds_player_1": best1[0], "odds_player_2": best2[0],
            "bookmaker_player_1": best1[1], "bookmaker_player_2": best2[1]}


def fetch_prematch_odds_by_event(
    *, event_id: str, player_1_name: str, player_2_name: str,
    api_key: str, host: str,
) -> dict[str, Any]:
    payload = _request_json(
        f"https://{host}/tennis/v2/extend/api/odds/pre-match/{event_id}",
        api_key, host,
    )
    parsed = _provider_result(
        _best_match_winner_prices(payload, player_1_name, player_2_name),
        reason="match_winner_parser_returned_none",
        source="extend_pre_match_odds",
        event_id=str(event_id),
    )
    return {
        **parsed,
        "event_id": str(event_id),
        "source": "extend_pre_match_odds",
        "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "provider_payload": payload,
    }


def fetch_odds_movements_by_event(
    *, event_id: str, api_key: str, host: str,
) -> dict[str, Any]:
    payload = _request_json(
        f"https://{host}/tennis/v2/extend/api/odds/summary/movements/last-10/{event_id}",
        api_key, host,
    )
    records = _api_objects(payload)
    return {
        "available": bool(records),
        "reason": None if records else "odds_movements_unavailable",
        "source": "extend_odds_movements",
        "event_id": str(event_id),
        "movements": records,
        "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "provider_payload": payload,
    }
