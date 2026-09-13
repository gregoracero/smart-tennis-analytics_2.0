#!/usr/bin/env python3
"""Canonical filesystem paths used by the Streamlit application.

This module is the single source of truth for runtime data paths. Application
pages, data-access modules and inference services should import paths from here
instead of defining local candidate lists or hard-coded v4/v5 fallbacks.

Environment variables may override the defaults for local, container or cloud
execution. Relative override values are resolved from the project root.
No file is read when this module is imported.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Iterable


# -----------------------------------------------------------------------------
# Project directories
# -----------------------------------------------------------------------------

STREAMLIT_APP_DIR: Final[Path] = Path(__file__).resolve().parents[1]
PROJECT_ROOT: Final[Path] = STREAMLIT_APP_DIR.parent

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
STAGING_DATA_DIR: Final[Path] = DATA_DIR / "staging"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
UPCOMING_DATA_DIR: Final[Path] = DATA_DIR / "upcoming"

MODELING_DIR: Final[Path] = PROJECT_ROOT / "modeling"
ARTIFACTS_DIR: Final[Path] = MODELING_DIR / "artifacts"
PRODUCTION_ARTIFACTS_DIR: Final[Path] = ARTIFACTS_DIR / "production"
ACTIVE_MODEL_PATH: Final[Path] = PRODUCTION_ARTIFACTS_DIR / "active_model.json"


# -----------------------------------------------------------------------------
# Path resolution
# -----------------------------------------------------------------------------

def _resolve_override(variable: str, default: Path) -> Path:
    """Return an environment override or the canonical default path.

    Absolute overrides are retained. Relative overrides are interpreted from
    PROJECT_ROOT so behavior is independent of the process working directory.
    User-home markers and environment variables are expanded.
    """
    raw_value = os.getenv(variable)
    if raw_value is None or not raw_value.strip():
        return default

    expanded = os.path.expandvars(os.path.expanduser(raw_value.strip()))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve(strict=False)


# -----------------------------------------------------------------------------
# Canonical inference inputs
# -----------------------------------------------------------------------------

MATCH_HISTORY_PATH: Final[Path] = _resolve_override(
    "SMART_TENNIS_MATCH_HISTORY_PATH",
    PROCESSED_DATA_DIR / "tennis_matches_with_player_stats.sackmann_v5.parquet",
)

PLAYER_CURRENT_STATE_PATH: Final[Path] = _resolve_override(
    "SMART_TENNIS_PLAYER_CURRENT_STATE_PATH",
    PROCESSED_DATA_DIR / "player_current_state.sackmann_v5.parquet",
)

PLAYER_PAIR_CURRENT_STATE_PATH: Final[Path] = _resolve_override(
    "SMART_TENNIS_PLAYER_PAIR_CURRENT_STATE_PATH",
    PROCESSED_DATA_DIR / "player_pair_current_state.sackmann_v5.parquet",
)

MATCH_FEATURE_MANIFEST_PATH: Final[Path] = _resolve_override(
    "SMART_TENNIS_MATCH_FEATURE_MANIFEST_PATH",
    PROCESSED_DATA_DIR / "tennis_match_feature_manifest.sackmann_v5.json",
)


# -----------------------------------------------------------------------------
# Canonical training datasets
# -----------------------------------------------------------------------------

WINNER_TRAINING_DATASET_PATH: Final[Path] = _resolve_override(
    "SMART_TENNIS_WINNER_TRAINING_DATASET_PATH",
    PROCESSED_DATA_DIR / "tennis_ml_no_market.sackmann_v5.parquet",
)

WINNER_TRAINING_METADATA_PATH: Final[Path] = _resolve_override(
    "SMART_TENNIS_WINNER_TRAINING_METADATA_PATH",
    PROCESSED_DATA_DIR / "tennis_ml_dataset_metadata_sackmann_v5.json",
)

WINNER_TRAINING_REPORT_PATH: Final[Path] = _resolve_override(
    "SMART_TENNIS_WINNER_TRAINING_REPORT_PATH",
    PROCESSED_DATA_DIR / "tennis_ml_dataset_report_sackmann_v5.csv",
)

WINNER_FEATURE_MANIFEST_PATH: Final[Path] = _resolve_override(
    "SMART_TENNIS_WINNER_FEATURE_MANIFEST_PATH",
    PROCESSED_DATA_DIR / "tennis_ml_feature_manifest_sackmann_v5.csv",
)

V6_ALL_TRAINING_DATASET_PATH: Final[Path] = _resolve_override(
    "SMART_TENNIS_V6_ALL_TRAINING_DATASET_PATH",
    PROCESSED_DATA_DIR / "tennis_ml_no_market.v6_all.sackmann_v5.parquet",
)

V6_ALL_MARKET_DATASET_PATH: Final[Path] = _resolve_override(
    "SMART_TENNIS_V6_ALL_MARKET_DATASET_PATH",
    PROCESSED_DATA_DIR / "tennis_ml_with_market.v6_all.sackmann_v5.parquet",
)

V6_ALL_METADATA_PATH: Final[Path] = _resolve_override(
    "SMART_TENNIS_V6_ALL_METADATA_PATH",
    PROCESSED_DATA_DIR / "tennis_ml_dataset_metadata.v6_all.sackmann_v5.json",
)


# -----------------------------------------------------------------------------
# Upcoming fixtures, runtime caches and generated state
# -----------------------------------------------------------------------------

UPCOMING_MATCHES_CACHE_PATH: Final[Path] = _resolve_override(
    "SMART_TENNIS_UPCOMING_CACHE_PATH",
    PROCESSED_DATA_DIR / "upcoming_matches_cache.parquet",
)

UPCOMING_CLASSIFICATION_AUDIT_PATH: Final[Path] = _resolve_override(
    "SMART_TENNIS_UPCOMING_AUDIT_PATH",
    PROCESSED_DATA_DIR / "upcoming_matches_classification_audit.parquet",
)

UPCOMING_PROVIDER_MATCHES_PATH: Final[Path] = _resolve_override(
    "SMART_TENNIS_PROVIDER_MATCHES_PATH",
    UPCOMING_DATA_DIR / "provider_matches.csv",
)

UPCOMING_MATCHES_PATH: Final[Path] = _resolve_override(
    "SMART_TENNIS_UPCOMING_MATCHES_PATH",
    UPCOMING_DATA_DIR / "upcoming_matches.parquet",
)

IA_CANDIDATES_DOSSIERS_PATH: Final[Path] = _resolve_override(
    "SMART_TENNIS_IA_DOSSIERS_PATH",
    DATA_DIR / "cache" / "ia_candidates_dossiers.json",
)


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

REQUIRED_INFERENCE_PATHS: Final[dict[str, Path]] = {
    "active_model": ACTIVE_MODEL_PATH,
    "match_history": MATCH_HISTORY_PATH,
    "player_current_state": PLAYER_CURRENT_STATE_PATH,
    "player_pair_current_state": PLAYER_PAIR_CURRENT_STATE_PATH,
    "match_feature_manifest": MATCH_FEATURE_MANIFEST_PATH,
}

OPTIONAL_RUNTIME_PATHS: Final[dict[str, Path]] = {
    "upcoming_cache": UPCOMING_MATCHES_CACHE_PATH,
    "upcoming_classification_audit": UPCOMING_CLASSIFICATION_AUDIT_PATH,
    "provider_matches": UPCOMING_PROVIDER_MATCHES_PATH,
    "upcoming_matches": UPCOMING_MATCHES_PATH,
    "ia_candidates_dossiers": IA_CANDIDATES_DOSSIERS_PATH,
}


def first_existing(paths: Iterable[Path], *, label: str) -> Path:
    """Return the first existing path or raise a detailed error.

    This helper is intended only for external provider alternatives. Canonical
    project datasets should normally use their explicit path constants.
    """
    candidates = tuple(paths)
    for path in candidates:
        if path.exists():
            return path
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No se encontro {label}. Rutas comprobadas: {checked}")


def validate_inference_paths(*, raise_on_missing: bool = False) -> dict[str, object]:
    """Inspect the canonical inference contract without loading large files."""
    status = {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file(),
        }
        for name, path in REQUIRED_INFERENCE_PATHS.items()
    }
    missing = [name for name, value in status.items() if not value["is_file"]]
    result: dict[str, object] = {
        "available": not missing,
        "missing": missing,
        "paths": status,
    }
    if missing and raise_on_missing:
        details = ", ".join(
            f"{name}={REQUIRED_INFERENCE_PATHS[name]}" for name in missing
        )
        raise FileNotFoundError(
            f"Faltan archivos del contrato canonico de inferencia: {details}"
        )
    return result


def ensure_runtime_directories() -> None:
    """Create only directories used for generated runtime state."""
    for directory in (
        UPCOMING_DATA_DIR,
        IA_CANDIDATES_DOSSIERS_PATH.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ACTIVE_MODEL_PATH",
    "ARTIFACTS_DIR",
    "DATA_DIR",
    "IA_CANDIDATES_DOSSIERS_PATH",
    "MATCH_FEATURE_MANIFEST_PATH",
    "MATCH_HISTORY_PATH",
    "MODELING_DIR",
    "OPTIONAL_RUNTIME_PATHS",
    "PLAYER_CURRENT_STATE_PATH",
    "PLAYER_PAIR_CURRENT_STATE_PATH",
    "PROCESSED_DATA_DIR",
    "PRODUCTION_ARTIFACTS_DIR",
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "REQUIRED_INFERENCE_PATHS",
    "STAGING_DATA_DIR",
    "STREAMLIT_APP_DIR",
    "UPCOMING_CLASSIFICATION_AUDIT_PATH",
    "UPCOMING_DATA_DIR",
    "UPCOMING_MATCHES_CACHE_PATH",
    "UPCOMING_MATCHES_PATH",
    "UPCOMING_PROVIDER_MATCHES_PATH",
    "V6_ALL_MARKET_DATASET_PATH",
    "V6_ALL_METADATA_PATH",
    "V6_ALL_TRAINING_DATASET_PATH",
    "WINNER_FEATURE_MANIFEST_PATH",
    "WINNER_TRAINING_DATASET_PATH",
    "WINNER_TRAINING_METADATA_PATH",
    "WINNER_TRAINING_REPORT_PATH",
    "ensure_runtime_directories",
    "first_existing",
    "validate_inference_paths",
]
