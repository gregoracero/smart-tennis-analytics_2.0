#!/usr/bin/env python3
"""Match Prediction optimizada y compatible con el pipeline surface/context v1.


La pagina:
- usa player_current_state.sackmann_v5.parquet para el estado global y por superficie;
- usa get_player_inference_snapshot para nivel, ronda y best-of;
- construye exactamente las features declaradas en run_metadata.json;
- calcula features derivadas con la misma definicion que el dataset ML;
- utiliza orientacion determinista por player_id, igual que training;
- mantiene H2H, forma, ultimos partidos y mercado como analisis secundarios;
- no carga el historico completo al abrir la pagina;
- recibe partidos y contexto desde Upcoming Matches;
- mantiene un modo manual simplificado;
- muestra frescura, procedencia y cobertura de los datos.
"""
from __future__ import annotations


import __main__
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


import joblib
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import streamlit as st
from catboost import CatBoostClassifier, Pool


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from streamlit_app.config.data_paths import (
    ACTIVE_MODEL_PATH,
    MATCH_HISTORY_PATH,
    PLAYER_CURRENT_STATE_PATH,
    PRODUCTION_ARTIFACTS_DIR,
)


from modeling.calibration import PlattCalibrator


__main__.PlattCalibrator = PlattCalibrator


from streamlit_app.data_access import (
    players_adapted as players_data_access,
)
from streamlit_app.services.match_prediction_service import predict_upcoming_fixture


get_players = (
    players_data_access.get_players
)

get_player_matches = (
    players_data_access.get_player_matches
)
get_players_tournament_history = getattr(
    players_data_access,
    "get_players_tournament_history",
    None,
)
get_players_current_tournament_statistics = getattr(
    players_data_access,
    "get_players_current_tournament_statistics",
    None,
)
get_players_recent_tournament_statistics = getattr(
    players_data_access,
    "get_players_recent_tournament_statistics",
    None,
)

def get_player_inference_snapshot(
    player_id: int,
    surface: str,
    tournament_level: str,
    round_name: str,
    best_of: int,
    competition_type: str | None = None,
    indoor: bool | None = None,
    prediction_date: Any | None = None,
):
    """Obtiene el snapshot utilizando la API consolidada de players_adapted."""

    specialized_function = getattr(
        players_data_access,
        "get_player_inference_snapshot",
        None,
    )

    if not callable(specialized_function):
        raise AttributeError(
            "players_adapted.py must provide get_player_inference_snapshot"
        )

    return specialized_function(
        player_id=player_id,
        surface=surface,
        tourney_level=tournament_level,
        round_name=round_name,
        best_of=best_of,
        competition_type=competition_type,
        indoor=indoor,
        prediction_date=prediction_date,
    )


def get_data_freshness() -> dict[str, Any]:
    external_function = getattr(
        players_data_access,
        "get_data_freshness",
        None,
    )

    if callable(external_function):
        try:
            result = external_function()

            if isinstance(result, dict):
                return result

        except Exception:
            pass

    state_path = PLAYER_CURRENT_STATE_PATH
    if not state_path.exists():
        return {
            "available": False,
            "status": "MISSING",
            "state_as_of_date": None,
            "freshness_days": None,
        }

    try:
        import pyarrow.parquet as pq

        schema = set(
            pq.ParquetFile(
                state_path
            ).schema_arrow.names
        )

        date_column = next(
            (
                column
                for column in [
                    "inference_state_as_of_date",
                    "state_as_of_date",
                    "snapshot_date",
                    "last_match_date",
                    "match_date",
                ]
                if column in schema
            ),
            None,
        )

        if date_column is not None:
            frame = pd.read_parquet(
                state_path,
                columns=[date_column],
                engine="pyarrow",
            )

            dates = pd.to_datetime(
                frame[date_column],
                errors="coerce",
            ).dropna()

        else:
            dates = pd.Series(
                dtype="datetime64[ns]"
            )

        if dates.empty:
            state_date = (
                pd.Timestamp.fromtimestamp(
                    state_path.stat().st_mtime
                )
                .normalize()
            )
        else:
            state_date = (
                pd.Timestamp(dates.max())
                .normalize()
            )

        today = (
            pd.Timestamp.today()
            .normalize()
        )

        freshness_days = max(
            int(
                (
                    today
                    - state_date
                ).days
            ),
            0,
        )

        if freshness_days <= 1:
            status = "FRESH"
        elif freshness_days <= 3:
            status = "WARNING"
        else:
            status = "STALE"

        return {
            "available": True,
            "status": status,
            "state_as_of_date": (
                state_date.isoformat()
            ),
            "freshness_days": freshness_days,
        }

    except Exception as error:
        return {
            "available": False,
            "status": "ERROR",
            "state_as_of_date": None,
            "freshness_days": None,
            "error": str(error),
        }

def local_data_freshness() -> dict[str, Any]:
    """Obtiene la frescura del estado actual de jugadores.

    Primero utiliza get_data_freshness de players_adapted cuando
    esté disponible. Si la función todavía no existe, calcula la
    frescura directamente desde la ruta canonica v5.
    """

    external_function = getattr(
        players_data_access,
        "get_data_freshness",
        None,
    )

    if callable(external_function):
        try:
            result = external_function()

            if isinstance(result, dict):
                return result

        except Exception:
            pass

    if not PLAYER_CURRENT_STATE_PATH.exists():
        return {
            "available": False,
            "status": "MISSING",
            "state_as_of_date": None,
            "freshness_days": None,
            "source": str(
                PLAYER_CURRENT_STATE_PATH
            ),
        }

    try:
        parquet_file = pq.ParquetFile(
            PLAYER_CURRENT_STATE_PATH
        )

        available_columns = set(
            parquet_file.schema_arrow.names
        )

        preferred_columns = [
            "inference_state_as_of_date",
            "state_as_of_date",
            "snapshot_date",
            "last_match_date",
            "match_date",
        ]

        selected_column = next(
            (
                column
                for column in preferred_columns
                if column in available_columns
            ),
            None,
        )

        if selected_column is None:
            state_date = pd.Timestamp(
                PLAYER_CURRENT_STATE_PATH
                .stat()
                .st_mtime,
                unit="s",
            ).normalize()

            source = "file_modification_time"

        else:
            frame = pd.read_parquet(
                PLAYER_CURRENT_STATE_PATH,
                columns=[
                    selected_column,
                ],
                engine="pyarrow",
            )

            values = pd.to_datetime(
                frame[selected_column],
                errors="coerce",
            ).dropna()

            if values.empty:
                state_date = pd.Timestamp(
                    PLAYER_CURRENT_STATE_PATH
                    .stat()
                    .st_mtime,
                    unit="s",
                ).normalize()

                source = "file_modification_time"

            else:
                state_date = (
                    values.max()
                    .normalize()
                )

                source = selected_column

        today = (
            pd.Timestamp.today()
            .normalize()
        )

        freshness_days = max(
            int(
                (
                    today
                    - state_date
                ).days
            ),
            0,
        )

        if freshness_days <= 1:
            status = "FRESH"

        elif freshness_days <= 3:
            status = "WARNING"

        else:
            status = "STALE"

        return {
            "available": True,
            "status": status,
            "state_as_of_date": (
                state_date.isoformat()
            ),
            "freshness_days": freshness_days,
            "source": source,
            "state_path": str(
                PLAYER_CURRENT_STATE_PATH
            ),
        }

    except Exception as error:
        return {
            "available": False,
            "status": "ERROR",
            "state_as_of_date": None,
            "freshness_days": None,
            "source": str(
                PLAYER_CURRENT_STATE_PATH
            ),
            "error": str(error),
        }


get_data_freshness = (
    local_data_freshness
)


# Backward-compatible local names retained by the existing page logic.
PRODUCTION_ROOT = PRODUCTION_ARTIFACTS_DIR
MATCH_FEATURES_PATH = MATCH_HISTORY_PATH
SYNTHETIC_ID_MIN = 90_000_000


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


@st.cache_resource(show_spinner=False)
def load_production_bundle(active_mtime_ns: int) -> dict[str, Any]:
    # active_mtime_ns forma parte de la cache key y fuerza recarga tras promocion.
    del active_mtime_ns
    active = read_json(ACTIVE_MODEL_PATH)
    required = {
        "active_model_version", "mode", "champion", "probability_policy",
        "artifact_directory", "model_file", "metadata_file",
    }
    missing = sorted(required - set(active))
    if missing:
        raise ValueError(f"active_model.json incompleto: {missing}")

    artifact_dir = (PRODUCTION_ROOT / active["artifact_directory"]).resolve()
    production_resolved = PRODUCTION_ROOT.resolve()
    if artifact_dir != production_resolved and production_resolved not in artifact_dir.parents:
        raise ValueError("active_model.json apunta fuera de production")

    model_path = artifact_dir / active["model_file"]
    metadata_path = artifact_dir / active["metadata_file"]
    manifest_path = artifact_dir / active.get(
        "registry_manifest_file", "registry_manifest.json"
    )
    metadata = read_json(metadata_path)
    manifest = read_json(manifest_path)

    for relative_name, expected_hash in manifest.get("hashes_sha256", {}).items():
        artifact = artifact_dir / relative_name
        if not artifact.exists():
            raise FileNotFoundError(f"Falta artefacto registrado: {artifact}")
        if sha256(artifact) != str(expected_hash).upper():
            raise RuntimeError(f"SHA-256 no coincide: {relative_name}")

    champion = str(active["champion"]).strip().lower()
    policy = str(active["probability_policy"]).strip().lower()
    if champion != str(metadata.get("champion", "")).strip().lower():
        raise RuntimeError("Champion inconsistente entre active_model y metadata")
    if policy != str(metadata.get("probability_policy", "")).strip().lower():
        raise RuntimeError("Politica inconsistente entre active_model y metadata")
    if policy not in {"raw", "calibrated"}:
        raise ValueError("La politica debe ser raw o calibrated")

    if champion in {"xgboost", "logistic"}:
        model = joblib.load(model_path)
        model_kind = "sklearn_pipeline"
    elif champion.startswith("catboost"):
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        model_kind = "catboost"
    else:
        raise ValueError(f"Champion no soportado: {champion}")

    calibrator = None
    if policy == "calibrated":
        calibrator_path = artifact_dir / "platt_calibrator.joblib"
        if not calibrator_path.exists():
            raise FileNotFoundError(
                "La politica solicita calibracion pero falta platt_calibrator.joblib"
            )
        calibrator = joblib.load(calibrator_path)

    features = list(metadata.get("features", metadata.get("feature_columns", [])))
    categoricals = list(
        metadata.get("categoricals", metadata.get("categorical_columns", []))
    )
    if not features:
        raise ValueError("model_metadata.json no declara features")
    if model_kind == "catboost" and list(model.feature_names_) != features:
        raise RuntimeError("El esquema CatBoost no coincide con metadata")

    return {
        "active": active,
        "artifact_dir": artifact_dir,
        "metadata": metadata,
        "manifest": manifest,
        "model": model,
        "model_kind": model_kind,
        "calibrator": calibrator,
        "features": features,
        "categoricals": categoricals,
    }


try:
    if not ACTIVE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "No existe production/active_model.json. Promocione un modelo primero."
        )
    production_bundle = load_production_bundle(
        ACTIVE_MODEL_PATH.stat().st_mtime_ns
    )
except Exception as error:
    st.error(f"No se pudo cargar el modelo activo: {error}")
    st.stop()

active_model = production_bundle["active"]
model = production_bundle["model"]
model_kind = production_bundle["model_kind"]
calibrator = production_bundle["calibrator"]
model_metadata = production_bundle["metadata"]
registry_manifest = production_bundle["manifest"]
MODEL_FEATURES = production_bundle["features"]
CATEGORICAL_FEATURES = production_bundle["categoricals"]
SELECTED_PROBABILITY = active_model["probability_policy"]
APPLY_CALIBRATION = SELECTED_PROBABILITY == "calibrated"
MODEL_VERSION = active_model["active_model_version"]

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]
TOURNAMENT_LEVELS = ["A", "G", "M", "F", "D", "C"]
TOURNAMENT_LEVEL_LABELS = {
    "A": "ATP Tour", "G": "Grand Slam", "M": "ATP Masters 1000",
    "F": "ATP Finals", "D": "Davis Cup / Team Event", "C": "ATP Challenger",
}
ROUND_LABELS = {
    "Q1": "Clasificacion, primera ronda (Q1)",
    "Q2": "Clasificacion, segunda ronda (Q2)",
    "Q3": "Clasificacion, ronda final (Q3)",
    "R128": "Ronda de 128 (R128)", "R64": "Ronda de 64 (R64)",
    "R32": "Dieciseisavos de final (R32)", "R16": "Octavos de final (R16)",
    "QF": "Cuartos de final (QF)", "SF": "Semifinal (SF)",
    "F": "Final (F)", "RR": "Fase de grupos (RR)",
}
ROUNDS = list(ROUND_LABELS)
ANALYSIS_SURFACES = ["Overall", *SURFACES]
ROUND_ORDER = {
    "PRE-Q": 0, "Q1": 5, "Q2": 6, "Q3": 7, "RR": 8,
    "R128": 10, "R64": 20, "R32": 30, "R16": 40,
    "QF": 50, "SF": 60, "F": 70, "BR": 75,
}


def add_round_order(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rounds = result.get("round", pd.Series("", index=result.index))
    result["_round_order"] = (
        rounds.astype("string").str.strip().str.upper()
        .map(ROUND_ORDER).fillna(-1).astype("Int16")
    )
    return result


def sort_match_history(frame: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    result = add_round_order(frame)
    columns = [
        column for column in
        ("match_date", "tourney_id", "_round_order", "match_num")
        if column in result
    ]
    result = result.sort_values(
        columns,
        ascending=[ascending] * len(columns),
        kind="mergesort",
        na_position="last",
    )
    return result.drop(columns=["_round_order"], errors="ignore").reset_index(drop=True)


st.set_page_config(page_title="Match Prediction", page_icon="🎾", layout="wide")
st.title("🎾 Match Prediction")
st.caption("Prediccion con estado actual, contexto de partido y estadisticas especificas de superficie.")
st.info(
    "Las probabilidades son estimaciones estadisticas. La fecha de estado indica "
    "hasta que dia estan actualizados los partidos conocidos."
)




@st.cache_data(show_spinner=False)
def load_players_catalog() -> pd.DataFrame:
    players = get_players()
    if players is None or players.empty:
        return pd.DataFrame(columns=["player_id", "player_name"])
    return (
        players.dropna(subset=["player_id", "player_name"])
        .drop_duplicates("player_id")
        .sort_values(["player_name", "player_id"], kind="mergesort")
        .reset_index(drop=True)
    )






def safe_numeric(value: Any) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan




def safe_number(value: Any, decimals: int = 2, default: str = "-") -> str:
    number = safe_numeric(value)
    return default if pd.isna(number) else f"{number:,.{decimals}f}"




def safe_integer(value: Any, default: str = "-") -> str:
    number = safe_numeric(value)
    return default if pd.isna(number) else f"{int(round(number)):,}"




def safe_percentage(value: Any, decimals: int = 1, default: str = "-") -> str:
    number = safe_numeric(value)
    return default if pd.isna(number) else f"{number:.{decimals}%}"




def market_status_label(value: Any) -> str:
    return {
        "validated_candidate": "Validated candidate",
        "provisional": "Provisional",
        "experimental": "Experimental",
        "not_validated": "Not validated",
    }.get(str(value or ""), str(value or "-").replace("_", " ").title())


def direct_market_view(market: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(market.get("available")),
        "player_1_first_set_probability": market.get("player_1_first_set_probability"),
        "player_2_first_set_probability": market.get("player_2_first_set_probability"),
        "player_1_win_any_set_probability": market.get(
            "player_1_win_any_set_probability", market.get("player_1_win_set_probability")
        ),
        "player_2_win_any_set_probability": market.get(
            "player_2_win_any_set_probability", market.get("player_2_win_set_probability")
        ),
        "over_probabilities": market.get("over_probabilities", {}),
        "model_status": market.get("model_status", {}),
        "over_model_status": market.get("over_model_status", {}),
        "structural_baseline": market.get("structural_baseline", {}),
        "selected_over_line": market.get("over_games_line"),
        "selected_over_probability": market.get("over_games_probability"),
        "selected_under_probability": market.get("under_games_probability"),
        "method": market.get("method"),
        "reason": market.get("reason"),
    }


def snapshot_value(snapshot: pd.Series | None, column: str, default: Any = np.nan) -> Any:
    return default if snapshot is None or column not in snapshot.index else snapshot[column]




def difference(left: Any, right: Any) -> float:
    left_value, right_value = safe_numeric(left), safe_numeric(right)
    return left_value - right_value if pd.notna(left_value) and pd.notna(right_value) else np.nan




def ratio(left: Any, right: Any) -> float:
    left_value, right_value = safe_numeric(left), safe_numeric(right)
    if pd.isna(left_value) or pd.isna(right_value) or right_value == 0:
        return np.nan
    return float(np.clip(left_value / right_value, 0, 20))




def elo_probability(difference_value: Any) -> float:
    diff = safe_numeric(difference_value)
    return np.nan if pd.isna(diff) else float(1.0 / (1.0 + 10.0 ** (-diff / 400.0)))




@st.cache_data(show_spinner=False, ttl=3600)
def load_player_history(player_id: int) -> pd.DataFrame:
    history = get_player_matches(
        player_id=player_id, season="CAREER", surface="ALL", last_n_matches=None
    )
    if history is None or history.empty:
        return pd.DataFrame()
    result = history.copy()
    result["match_date"] = pd.to_datetime(result["match_date"], errors="coerce")
    return sort_match_history(result, ascending=False)




def filter_history(
    history: pd.DataFrame,
    period: str,
    surface: str,
    limit: int | None = None,
) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    result = history.copy()
    if period == "Current Season" and "match_date" in result:
        years = result["match_date"].dropna().dt.year
        if not years.empty:
            result = result.loc[result["match_date"].dt.year.eq(int(years.max()))]
    if surface != "Overall" and "surface" in result:
        result = result.loc[result["surface"].astype(str).str.casefold().eq(surface.casefold())]
    result = sort_match_history(result, ascending=False)
    if limit is not None:
        result = result.head(int(limit))
    return result.reset_index(drop=True)




def win_rate(history: pd.DataFrame) -> float:
    if history is None or history.empty:
        return np.nan
    if "won" in history:
        values = pd.to_numeric(history["won"], errors="coerce").dropna()
    else:
        values = history["result"].astype("string").str.upper().map({"W": 1.0, "L": 0.0}).dropna()
    return float(values.mean()) if not values.empty else np.nan




def weighted_percentage(history: pd.DataFrame, numerator: str, denominator: str, fallback: str) -> float:
    if history is None or history.empty:
        return np.nan
    if numerator in history and denominator in history:
        num = pd.to_numeric(history[numerator], errors="coerce")
        den = pd.to_numeric(history[denominator], errors="coerce")
        valid = num.notna() & den.gt(0)
        if valid.any() and den.loc[valid].sum() > 0:
            return float(num.loc[valid].sum() / den.loc[valid].sum())
    if fallback in history:
        values = pd.to_numeric(history[fallback], errors="coerce").dropna()
        return float(values.mean()) if not values.empty else np.nan
    return np.nan




def comparison_values(
    player_id: int,
    snapshot: pd.Series,
    period: str,
    surface: str,
    recent_match_count: int,
) -> dict[str, Any]:
    full = load_player_history(player_id)
    filtered = filter_history(full, period, surface)
    recent = filtered.head(int(recent_match_count))
    last_5, last_10 = filtered.head(5), filtered.head(10)
    surface_key = surface if surface != "Overall" else snapshot_value(snapshot, "selected_surface", "Overall")
    return {
        "rank": snapshot_value(snapshot, "player_rank"),
        "rank_points": snapshot_value(snapshot, "player_rank_points"),
        "age": snapshot_value(snapshot, "player_age"),
        "height": snapshot_value(snapshot, "player_height"),
        "career_matches": snapshot_value(snapshot, "career_matches_before"),
        "career_win_rate": snapshot_value(snapshot, "career_win_rate_before"),
        "global_elo": snapshot_value(snapshot, "elo_before"),
        "surface_matches": len(filtered) if surface != "Overall" else snapshot_value(snapshot, "career_matches_before"),
        "surface_win_rate": win_rate(filtered) if surface != "Overall" else snapshot_value(snapshot, "career_win_rate_before"),
        "surface_elo": snapshot_value(snapshot, "surface_elo_before") if surface != "Overall" else snapshot_value(snapshot, "elo_before"),
        "last_5_win_rate": win_rate(last_5),
        "last_10_win_rate": win_rate(last_10),
        "recent_win_rate": win_rate(recent),
        "recent_matches_used": int(len(recent)),
        "days_since_last_match": snapshot_value(snapshot, "days_since_last_match"),
        "service_points_won": weighted_percentage(
            recent, "service_points_won", "service_points", "service_points_won_pct"
        ),
        "return_points_won": weighted_percentage(
            recent, "return_points_won", "return_points_played", "return_points_won_pct"
        ),
        "surface_label": surface_key,
    }




@st.cache_data(show_spinner=False, ttl=3600)
def get_h2h_matches(player_a_id: int, player_b_id: int) -> pd.DataFrame:
    if not MATCH_FEATURES_PATH.exists():
        return pd.DataFrame()
    dataset = ds.dataset(str(MATCH_FEATURES_PATH), format="parquet")
    names = set(dataset.schema.names)
    required = {"player_1_id", "player_2_id", "target_player_1_win"}
    if not required.issubset(names):
        return pd.DataFrame()
    columns = [
        column for column in (
            "player_1_id", "player_1_name", "player_2_id", "player_2_name",
            "target_player_1_win", "match_date", "tourney_id", "tourney_name",
            "tourney_level", "competition_type", "surface", "round",
            "score", "match_num",
        ) if column in names
    ]
    expression = (
        ((ds.field("player_1_id") == int(player_a_id)) & (ds.field("player_2_id") == int(player_b_id)))
        | ((ds.field("player_1_id") == int(player_b_id)) & (ds.field("player_2_id") == int(player_a_id)))
    )
    result = dataset.to_table(columns=columns, filter=expression).to_pandas()
    if result.empty:
        return result
    result["match_date"] = pd.to_datetime(result["match_date"], errors="coerce")
    p1_won = pd.to_numeric(result["target_player_1_win"], errors="coerce").eq(1)
    if {"player_1_name", "player_2_name"}.issubset(result.columns):
        result["winner_name"] = result["player_1_name"].where(p1_won, result["player_2_name"])
        result["loser_name"] = result["player_2_name"].where(p1_won, result["player_1_name"])
    return sort_match_history(result, ascending=False)

def calculate_h2h(player_a_id: int, player_b_id: int) -> dict[str, Any]:
    history = get_h2h_matches(player_a_id, player_b_id)
    if history.empty:
        return {
            "matches": 0, "player_1_wins": 0, "player_2_wins": 0,
            "player_1_win_rate": np.nan, "history": history,
        }
    target = pd.to_numeric(history["target_player_1_win"], errors="coerce")
    a_is_side_1 = pd.to_numeric(history["player_1_id"], errors="coerce").eq(player_a_id)
    a_won = np.where(a_is_side_1, target.eq(1), target.eq(0))
    wins = int(np.asarray(a_won).sum())
    return {
        "matches": len(history), "player_1_wins": wins,
        "player_2_wins": len(history) - wins,
        "player_1_win_rate": wins / len(history), "history": history,
    }




def reverse_h2h(h2h: dict[str, Any]) -> dict[str, Any]:
    matches = h2h["matches"]
    return {
        "matches": matches,
        "player_1_wins": h2h["player_2_wins"],
        "player_2_wins": h2h["player_1_wins"],
        "player_1_win_rate": h2h["player_2_wins"] / matches if matches else np.nan,
        "history": h2h["history"],
    }




def confidence_components(player_a: pd.Series, player_b: pd.Series, h2h_matches: int) -> dict[str, Any]:
    career = max(min(
        safe_numeric(snapshot_value(player_a, "career_matches_before", 0)),
        safe_numeric(snapshot_value(player_b, "career_matches_before", 0)),
    ), 0)
    surface = max(min(
        safe_numeric(snapshot_value(player_a, "surface_matches_before", 0)),
        safe_numeric(snapshot_value(player_b, "surface_matches_before", 0)),
    ), 0)
    stats = max(min(
        safe_numeric(snapshot_value(player_a, "stat_matches_before", 0)),
        safe_numeric(snapshot_value(player_b, "stat_matches_before", 0)),
    ), 0)
    score = round(
        min(career / 150, 1) * 40 + min(surface / 60, 1) * 30
        + min(stats / 40, 1) * 20 + min(h2h_matches / 10, 1) * 10
    )
    return {
        "score": score, "career_matches_min": career,
        "surface_matches_min": surface, "stat_matches_min": stats,
        "h2h_matches": h2h_matches,
    }




def confidence_label(score: int) -> str:
    return "HIGH" if score >= 80 else "MEDIUM" if score >= 60 else "LOW"




def feature_from_snapshot(snapshot: pd.Series, feature: str) -> Any:
    generic = feature
    for prefix in ("player_1_", "player_2_"):
        if generic.startswith(prefix):
            generic = generic[len(prefix):]
            break
    aliases = {
        "height": "player_height", "age": "player_age", "rank": "player_rank",
        "rank_points": "player_rank_points", "hand": "player_hand", "ioc": "player_ioc",
    }
    return snapshot_value(snapshot, aliases.get(generic, generic))




def quality_feature(feature: str, player_a: pd.Series, player_b: pd.Series) -> tuple[Any, str]:
    mappings = {
        "min_career_matches_before": "career_matches_before",
        "min_stat_matches_before": "stat_matches_before",
        "min_surface_matches_before": "surface_matches_before",
        "min_surface_stat_matches_before": "surface_stat_matches_before",
    }
    if feature in mappings:
        column = mappings[feature]
        values = [safe_numeric(snapshot_value(player_a, column)), safe_numeric(snapshot_value(player_b, column))]
        return (min(values) if all(pd.notna(value) for value in values) else np.nan), "minimum coverage"
    if feature == "cold_start_any_player":
        return int(min(
            safe_numeric(snapshot_value(player_a, "career_matches_before", 0)),
            safe_numeric(snapshot_value(player_b, "career_matches_before", 0)),
        ) < 5), "career coverage flag"
    if feature == "surface_cold_start_any_player":
        return int(min(
            safe_numeric(snapshot_value(player_a, "surface_matches_before", 0)),
            safe_numeric(snapshot_value(player_b, "surface_matches_before", 0)),
        ) < 5), "surface coverage flag"
    if feature == "stats_reliable_both":
        return int(min(
            safe_numeric(snapshot_value(player_a, "stat_matches_before", 0)),
            safe_numeric(snapshot_value(player_b, "stat_matches_before", 0)),
        ) >= 10), "global statistics coverage flag"
    if feature == "surface_stats_reliable_both":
        return int(min(
            safe_numeric(snapshot_value(player_a, "surface_stat_matches_before", 0)),
            safe_numeric(snapshot_value(player_b, "surface_stat_matches_before", 0)),
        ) >= 10), "surface statistics coverage flag"
    if feature == "synthetic_player_any":
        ids = [safe_numeric(snapshot_value(player_a, "player_id")), safe_numeric(snapshot_value(player_b, "player_id"))]
        return int(any(pd.notna(value) and value >= SYNTHETIC_ID_MIN for value in ids)), "identity coverage flag"
    return None, ""




def derived_feature(feature: str, player_a: pd.Series, player_b: pd.Series) -> tuple[Any, str]:
    diff_elo = difference(snapshot_value(player_a, "elo_before"), snapshot_value(player_b, "elo_before"))
    diff_surface_elo = difference(
        snapshot_value(player_a, "surface_elo_before"), snapshot_value(player_b, "surface_elo_before")
    )
    values = {
        "elo_expected_p1": elo_probability(diff_elo),
        "surface_elo_expected_p1": elo_probability(diff_surface_elo),
        "absolute_elo_difference": abs(diff_elo) if pd.notna(diff_elo) else np.nan,
        "absolute_surface_elo_difference": abs(diff_surface_elo) if pd.notna(diff_surface_elo) else np.nan,
        "rank_ratio_p1_p2": ratio(snapshot_value(player_a, "player_rank"), snapshot_value(player_b, "player_rank")),
        "rank_points_ratio_p1_p2": ratio(
            safe_numeric(snapshot_value(player_a, "player_rank_points")) + 1,
            safe_numeric(snapshot_value(player_b, "player_rank_points")) + 1,
        ),
    }
    return (values.get(feature), "stable derived feature") if feature in values else (None, "")




def build_model_features(
    player_a: pd.Series,
    player_b: pd.Series,
    surface: str,
    tournament_level: str,
    round_name: str,
    best_of: int,
    competition_type: str,
    indoor: bool,
    h2h: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    context = {
        "surface": surface, "tourney_level": tournament_level,
        "competition_type": competition_type, "indoor": str(bool(indoor)),
        "round": round_name, "best_of": str(best_of),
    }
    row: dict[str, Any] = {}
    audit: list[dict[str, Any]] = []
    missing: list[str] = []


    for feature in MODEL_FEATURES:
        value, source = None, "unsupported"
        if feature in context:
            value, source = context[feature], "selected match context"
        elif feature == "player_1_h2h_wins_before":
            value, source = h2h["player_1_wins"], "historical H2H"
        elif feature == "player_2_h2h_wins_before":
            value, source = h2h["player_2_wins"], "historical H2H"
        elif feature == "player_1_h2h_win_rate_before":
            value, source = h2h["player_1_win_rate"], "historical H2H"
        elif feature.startswith("diff_"):
            suffix = feature[len("diff_"):]
            left = feature_from_snapshot(player_a, f"player_1_{suffix}")
            right = feature_from_snapshot(player_b, f"player_2_{suffix}")
            value, source = difference(left, right), "player_1 minus player_2"
        elif feature.startswith("player_1_"):
            value, source = feature_from_snapshot(player_a, feature), "player 1 current state"
        elif feature.startswith("player_2_"):
            value, source = feature_from_snapshot(player_b, feature), "player 2 current state"
        else:
            value, source = quality_feature(feature, player_a, player_b)
            if value is None:
                value, source = derived_feature(feature, player_a, player_b)


        if feature in CATEGORICAL_FEATURES:
            final = "__MISSING__" if value is None or pd.isna(value) else str(value)
        else:
            final = safe_numeric(value)
        is_missing = final == "__MISSING__" if isinstance(final, str) else pd.isna(final)
        if is_missing:
            missing.append(feature)
        row[feature] = final
        audit.append({"feature": feature, "value": final, "source": source, "missing": is_missing})


    frame = pd.DataFrame([row], columns=MODEL_FEATURES)
    for feature in CATEGORICAL_FEATURES:
        frame[feature] = frame[feature].astype("string").fillna("__MISSING__").astype(str)
    return frame, pd.DataFrame(audit), sorted(set(missing))




def raw_model_probability(frame: pd.DataFrame) -> float:
    """Predice usando el artefacto completo promocionado."""
    if model_kind == "catboost":
        pool = Pool(
            frame[MODEL_FEATURES],
            cat_features=CATEGORICAL_FEATURES,
            feature_names=MODEL_FEATURES,
        )
        probability = model.predict_proba(pool)[0, 1]
    else:
        # XGBoost y logistica incluyen imputacion y one-hot dentro del Pipeline.
        probability = model.predict_proba(frame[MODEL_FEATURES])[0, 1]
    probability = float(probability)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeError("El modelo devolvio una probabilidad invalida")
    return probability


def apply_probability_policy(raw_probability: float) -> float:
    if not APPLY_CALIBRATION:
        return float(raw_probability)
    values = np.asarray([raw_probability], dtype=float)
    if hasattr(calibrator, "predict"):
        calibrated = np.asarray(calibrator.predict(values), dtype=float)[0]
    elif hasattr(calibrator, "predict_proba"):
        result = np.asarray(calibrator.predict_proba(values), dtype=float)
        calibrated = result[0, -1]
    else:
        raise TypeError("API de calibrador no soportada")
    return float(np.clip(calibrated, 1e-7, 1.0 - 1e-7))


def predict_symmetrically(
    player_a: pd.Series,
    player_b: pd.Series,
    surface: str,
    tournament_level: str,
    round_name: str,
    best_of: int,
    competition_type: str,
    indoor: bool,
    h2h_ab: dict[str, Any],
) -> tuple[float, float, pd.DataFrame, list[str], dict[str, float]]:
    """Mantiene la interfaz legacy, pero reproduce el contrato de training.

    El dataset canonico v5 orienta cada partido por player_1_id < player_2_id. El modelo
    promocionado fue seleccionado y evaluado con esa orientacion determinista.
    Promediar ambas orientaciones seria una transformacion no validada. Por eso
    se construye una sola fila en la orientacion de entrenamiento y la
    probabilidad se devuelve al orden visual A/B.
    """
    player_a_id = int(safe_numeric(snapshot_value(player_a, "player_id")))
    player_b_id = int(safe_numeric(snapshot_value(player_b, "player_id")))
    a_is_model_side_1 = player_a_id < player_b_id

    if a_is_model_side_1:
        model_side_1, model_side_2 = player_a, player_b
        model_h2h = h2h_ab
    else:
        model_side_1, model_side_2 = player_b, player_a
        model_h2h = reverse_h2h(h2h_ab)

    frame, audit, missing = build_model_features(
        model_side_1,
        model_side_2,
        surface,
        tournament_level,
        round_name,
        best_of,
        competition_type,
        indoor,
        model_h2h,
    )
    raw_model_side_1 = raw_model_probability(frame)
    selected_model_side_1 = apply_probability_policy(raw_model_side_1)

    raw_a = raw_model_side_1 if a_is_model_side_1 else 1.0 - raw_model_side_1
    selected_a = (
        selected_model_side_1
        if a_is_model_side_1
        else 1.0 - selected_model_side_1
    )
    diagnostics = {
        "raw_model_side_1": float(raw_model_side_1),
        "model_side_1_player_id": float(min(player_a_id, player_b_id)),
        "raw_orientation_gap": 0.0,
        "orientation_policy": "ascending_player_id",
    }
    return (
        float(np.clip(raw_a, 1e-7, 1.0 - 1e-7)),
        float(np.clip(selected_a, 1e-7, 1.0 - 1e-7)),
        audit,
        missing,
        diagnostics,
    )

def option_index(
    options: list[Any],
    selected_value: Any,
    default: int = 0,
) -> int:
    try:
        return options.index(selected_value)
    except (ValueError, TypeError):
        return default


def player_label_index(
    labels: list[str],
    options: dict[str, int],
    target_player_id: Any,
    default: int,
) -> int:
    target = safe_numeric(target_player_id)
    if pd.isna(target):
        return default
    for index, label in enumerate(labels):
        if int(options[label]) == int(target):
            return index
    return default


def infer_competition_type(
    tournament_level: str,
    round_name: str,
) -> str:
    if str(tournament_level).upper() == "C":
        return "CHALLENGER"
    if str(round_name).upper() in {"Q1", "Q2", "Q3"}:
        return "ATP_QUALIFYING"
    return "ATP"


def infer_best_of(
    tournament_level: str,
    round_name: str,
) -> int:
    grand_slam_main_draw = (
        str(tournament_level).upper() == "G"
        and str(round_name).upper() not in {"Q1", "Q2", "Q3"}
    )
    return 5 if grand_slam_main_draw else 3


def valid_upcoming_selection(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    required = {
        "player_1_id",
        "player_2_id",
        "surface",
        "tournament_level",
        "round",
        "best_of",
        "competition_type",
        "indoor",
    }
    return required.issubset(value)


players = load_players_catalog()
if players.empty:
    st.error("No players were found.")
    st.stop()


@st.cache_data(show_spinner=False)
def build_player_options(ids: tuple, names: tuple) -> dict[str, int]:
    return {
        f"{name} [{player_id}]": player_id
        for player_id, name in zip(ids, names)
    }


player_options = build_player_options(
    tuple(players["player_id"].tolist()),
    tuple(players["player_name"].tolist()),
)
player_labels = list(player_options)

upcoming_selection = st.session_state.get("upcoming_match_selection")
has_upcoming_selection = valid_upcoming_selection(upcoming_selection)

if has_upcoming_selection:
    st.success(
        "Scheduled match loaded: "
        f"{upcoming_selection.get('player_1_name', 'Player 1')} vs "
        f"{upcoming_selection.get('player_2_name', 'Player 2')}"
    )
    fixture_details = " · ".join(
        str(value)
        for value in (
            upcoming_selection.get("match_date"),
            upcoming_selection.get("scheduled_time"),
            upcoming_selection.get("tournament"),
        )
        if value not in (None, "", "TBD")
    )
    if fixture_details:
        st.caption(fixture_details)

setup_mode = st.radio(
    "Prediction mode",
    ["Scheduled Match", "Manual Match"],
    index=0 if has_upcoming_selection else 1,
    horizontal=True,
    key="match_prediction_mode",
)

scheduled_mode = setup_mode == "Scheduled Match" and has_upcoming_selection
if setup_mode == "Scheduled Match" and not has_upcoming_selection:
    st.info(
        "Select a fixture in Upcoming Matches first, or use Manual Match."
    )

if scheduled_mode:
    default_surface = upcoming_selection["surface"]
    default_level = upcoming_selection["tournament_level"]
    default_round = upcoming_selection["round"]
    default_best_of = int(upcoming_selection["best_of"])
    default_competition = upcoming_selection["competition_type"]
    default_indoor = bool(upcoming_selection["indoor"])
    default_player_1_id = upcoming_selection["player_1_id"]
    default_player_2_id = upcoming_selection["player_2_id"]
else:
    default_surface = "Hard"
    default_level = "A"
    default_round = "R16"
    default_best_of = 3
    default_competition = "ATP"
    default_indoor = False
    default_player_1_id = None
    default_player_2_id = None

with st.form("match_prediction_form", clear_on_submit=False):
    st.subheader("Match Setup")

    if scheduled_mode:
        schedule_1, schedule_2, schedule_3, schedule_4 = st.columns(4)
        schedule_1.metric("Surface", default_surface)
        schedule_2.metric(
            "Tournament Level",
            TOURNAMENT_LEVEL_LABELS.get(default_level, default_level),
        )
        schedule_3.metric(
            "Round",
            ROUND_LABELS.get(default_round, default_round),
        )
        schedule_4.metric(
            "Format",
            f"{default_competition} · Best of {default_best_of} · "
            f"{'Indoor' if default_indoor else 'Outdoor'}",
        )

        surface_input = default_surface
        tournament_level_input = default_level
        round_input = default_round
        best_of_input = default_best_of
        competition_type_input = default_competition
        indoor_input = default_indoor

    else:
        context_1, context_2, context_3, context_4 = st.columns(4)
        with context_1:
            surface_input = st.selectbox(
                "Surface",
                SURFACES,
                index=option_index(SURFACES, default_surface),
            )
        with context_2:
            tournament_level_input = st.selectbox(
                "Tournament Level",
                TOURNAMENT_LEVELS,
                index=option_index(TOURNAMENT_LEVELS, default_level),
                format_func=lambda value: TOURNAMENT_LEVEL_LABELS.get(
                    value,
                    value,
                ),
            )
        with context_3:
            round_input = st.selectbox(
                "Round",
                ROUNDS,
                index=option_index(ROUNDS, default_round, 6),
                format_func=lambda value: ROUND_LABELS.get(value, value),
            )
        with context_4:
            indoor_input = st.selectbox(
                "Environment",
                [False, True],
                index=1 if default_indoor else 0,
                format_func=lambda value: "Indoor" if value else "Outdoor",
            )

        competition_type_input = infer_competition_type(
            tournament_level_input,
            round_input,
        )
        best_of_input = infer_best_of(
            tournament_level_input,
            round_input,
        )
        st.caption(
            "Automatically inferred context: "
            f"{competition_type_input} · Best of {best_of_input}"
        )

    player_column_1, player_column_2 = st.columns(2)
    with player_column_1:
        player_1_label = st.selectbox(
            "Player 1",
            player_labels,
            index=player_label_index(
                player_labels,
                player_options,
                default_player_1_id,
                0,
            ),
            disabled=scheduled_mode,
        )
    with player_column_2:
        player_2_label = st.selectbox(
            "Player 2",
            player_labels,
            index=player_label_index(
                player_labels,
                player_options,
                default_player_2_id,
                1 if len(player_labels) > 1 else 0,
            ),
            disabled=scheduled_mode,
        )

    action_1, action_2 = st.columns([3, 1])
    with action_1:
        predict_clicked = st.form_submit_button(
            "Predict Scheduled Match" if scheduled_mode else "Predict Match",
            type="primary",
            use_container_width=True,
            disabled=setup_mode == "Scheduled Match" and not scheduled_mode,
        )
    with action_2:
        clear_clicked = st.form_submit_button(
            "Clear",
            use_container_width=True,
        )


if clear_clicked:
    st.session_state.pop("prediction", None)
    st.session_state.pop("upcoming_match_selection", None)
    st.session_state.pop("match_prediction_mode", None)
    st.rerun()


if predict_clicked:
    selected_player_1_id = int(player_options[player_1_label])
    selected_player_2_id = int(player_options[player_2_label])
    if selected_player_1_id == selected_player_2_id:
        st.error("Select two different players.")
        st.stop()
    selected_player_1_name = str(players.loc[players["player_id"].eq(selected_player_1_id), "player_name"].iloc[0])
    selected_player_2_name = str(players.loc[players["player_id"].eq(selected_player_2_id), "player_name"].iloc[0])


    with st.spinner("Building current player states and calculating prediction..."):
        prediction_date_input = (
            upcoming_selection.get("match_date")
            if scheduled_mode and upcoming_selection
            else None
        )
        player_1 = get_player_inference_snapshot(
            selected_player_1_id,
            surface_input,
            tournament_level_input,
            round_input,
            best_of_input,
            competition_type=competition_type_input,
            indoor=indoor_input,
            prediction_date=prediction_date_input,
        )
        player_2 = get_player_inference_snapshot(
            selected_player_2_id,
            surface_input,
            tournament_level_input,
            round_input,
            best_of_input,
            competition_type=competition_type_input,
            indoor=indoor_input,
            prediction_date=prediction_date_input,
        )
        if player_1 is None or player_2 is None:
            st.error("A current state could not be found for one or both players.")
            st.stop()
        player_1 = player_1.copy()
        player_2 = player_2.copy()
        player_1["player_id"] = selected_player_1_id
        player_2["player_id"] = selected_player_2_id
        h2h = calculate_h2h(selected_player_1_id, selected_player_2_id)
        raw_probability, final_probability, feature_audit, missing_features, symmetry = predict_symmetrically(
            player_1, player_2, surface_input, tournament_level_input,
            round_input, best_of_input, competition_type_input, indoor_input, h2h,
        )


    probability_player_2 = 1.0 - final_probability
    market_prediction: dict[str, Any] = {}
    market_prediction_error: str | None = None
    try:
        requested_over_line = 38.5 if int(best_of_input) == 5 else 21.5
        canonical_prediction = predict_upcoming_fixture(
            player_1_id=selected_player_1_id,
            player_2_id=selected_player_2_id,
            surface=surface_input,
            tournament_level=tournament_level_input,
            round_name=round_input,
            best_of=int(best_of_input),
            competition_type=competition_type_input,
            indoor=bool(indoor_input),
            prediction_date=prediction_date_input,
            tournament_id=str(
                upcoming_selection.get("tournament_id", upcoming_selection.get("tournament", ""))
                if scheduled_mode and upcoming_selection else ""
            ),
            over_games_line=requested_over_line,
            market_simulations=1_000,
        )
        if canonical_prediction.get("available", True):
            # Use the canonical service result so all prediction pages share the
            # exact same winner and market inference contract.
            final_probability = float(canonical_prediction["probability_player_1"])
            probability_player_2 = float(canonical_prediction["probability_player_2"])
            market_prediction = canonical_prediction.get("market_probabilities", {}) or {}
        else:
            market_prediction_error = str(canonical_prediction.get("reason", "unavailable"))
    except Exception as error:
        # Winner prediction from the local production pipeline remains usable if
        # an optional market artifact fails.
        market_prediction_error = str(error)
    confidence = confidence_components(player_1, player_2, h2h["matches"])
    st.session_state["prediction"] = {
        "player_1_id": selected_player_1_id, "player_2_id": selected_player_2_id,
        "player_1_name": selected_player_1_name, "player_2_name": selected_player_2_name,
        "player_1": player_1, "player_2": player_2,
        "surface": surface_input, "tournament_level": tournament_level_input,
        "round": round_input, "best_of": best_of_input,
        "competition_type": competition_type_input, "indoor": indoor_input,
        "probability_player_1": final_probability,
        "probability_player_2": probability_player_2,
        "raw_probability_player_1": raw_probability,
        "fair_odds_player_1": 1.0 / final_probability,
        "fair_odds_player_2": 1.0 / probability_player_2,
        "confidence": confidence, "h2h": h2h,
        "feature_audit": feature_audit, "missing_features": missing_features,
        "synthetic_player_any": selected_player_1_id >= SYNTHETIC_ID_MIN or selected_player_2_id >= SYNTHETIC_ID_MIN,
        "selected_probability_policy": SELECTED_PROBABILITY,
        "symmetry_diagnostics": symmetry,
        "prediction_source": "upcoming_matches" if scheduled_mode else "manual",
        "scheduled_fixture": upcoming_selection.copy() if scheduled_mode else None,
        "tournament_history_name": (
            upcoming_selection.get("tournament_history_name", upcoming_selection.get("tournament"))
            if scheduled_mode and upcoming_selection else None
        ),
        "tournament_history_id": (
            upcoming_selection.get("tournament_history_id", upcoming_selection.get("tournament_id"))
            if scheduled_mode and upcoming_selection else None
        ),
        "tournament_history_years": int(
            upcoming_selection.get("tournament_history_years", 10)
            if scheduled_mode and upcoming_selection else 10
        ),
        "prediction_date": prediction_date_input,
        "market_probabilities": market_prediction,
        "market_prediction_error": market_prediction_error,
    }


if "prediction" not in st.session_state:
    st.stop()


prediction = st.session_state["prediction"]
player_1_id, player_2_id = prediction["player_1_id"], prediction["player_2_id"]
player_1_name, player_2_name = prediction["player_1_name"], prediction["player_2_name"]
player_1, player_2 = prediction["player_1"], prediction["player_2"]
prob_1, prob_2 = prediction["probability_player_1"], prediction["probability_player_2"]
fair_odds_1, fair_odds_2 = prediction["fair_odds_player_1"], prediction["fair_odds_player_2"]
confidence, h2h = prediction["confidence"], prediction["h2h"]
feature_audit, missing_features = prediction["feature_audit"], prediction["missing_features"]
surface, tournament_level = prediction["surface"], prediction["tournament_level"]
round_name, best_of = prediction["round"], prediction["best_of"]
competition_type, indoor = prediction["competition_type"], prediction["indoor"]


st.divider()
st.subheader("Prediction Summary")
if prediction.get("prediction_source") == "upcoming_matches":
    scheduled_fixture = prediction.get("scheduled_fixture") or {}
    st.caption(
        "Scheduled fixture · "
        + " · ".join(
            str(value)
            for value in (
                scheduled_fixture.get("match_date"),
                scheduled_fixture.get("scheduled_time"),
                scheduled_fixture.get("tournament"),
            )
            if value not in (None, "", "TBD")
        )
    )
st.caption(
    f"{TOURNAMENT_LEVEL_LABELS.get(tournament_level, tournament_level)} · "
    f"{ROUND_LABELS.get(round_name, round_name)} · {surface} · Best of {best_of} · "
    f"{competition_type} · {'Indoor' if indoor else 'Outdoor'}"
)
summary_1, summary_center, summary_2 = st.columns([2, 3, 2])
with summary_1:
    st.metric(player_1_name, safe_percentage(prob_1))
    st.caption(f"Fair odds: {safe_number(fair_odds_1)}")
with summary_center:
    st.progress(float(prob_1))
    st.metric("Data Confidence", f"{confidence['score']}/100")
    st.caption(f"Coverage: {confidence_label(confidence['score'])}")
with summary_2:
    st.metric(player_2_name, safe_percentage(prob_2))
    st.caption(f"Fair odds: {safe_number(fair_odds_2)}")


market_probabilities = prediction.get("market_probabilities", {}) or {}
direct_markets = direct_market_view(market_probabilities)
st.subheader("Set and Total-Game Markets")
if direct_markets.get("available"):
    first_status = market_status_label(
        direct_markets.get("model_status", {}).get("first_set")
    )
    any_status = market_status_label(
        direct_markets.get("model_status", {}).get("win_any_set")
    )
    first_1, first_2, any_1, any_2 = st.columns(4)
    first_1.metric(
        f"{player_1_name} first set",
        safe_percentage(direct_markets.get("player_1_first_set_probability")),
        help=first_status,
    )
    first_2.metric(
        f"{player_2_name} first set",
        safe_percentage(direct_markets.get("player_2_first_set_probability")),
        help=first_status,
    )
    any_1.metric(
        f"{player_1_name} wins >=1 set",
        safe_percentage(direct_markets.get("player_1_win_any_set_probability")),
        help=any_status,
    )
    any_2.metric(
        f"{player_2_name} wins >=1 set",
        safe_percentage(direct_markets.get("player_2_win_any_set_probability")),
        help=any_status,
    )
    st.caption(
        "First-set probabilities are complementary. At-least-one-set probabilities "
        "are not complementary because both players can win a set."
    )
    if direct_markets.get("over_probabilities"):
        over_rows = [
            {
                "Market": f"Over {line}",
                "Probability": safe_percentage(probability),
                "Fair odds": safe_number(
                    1.0 / float(probability) if float(probability) > 0 else np.nan, 2
                ),
                "Status": market_status_label(
                    direct_markets.get("over_model_status", {}).get(str(line))
                ),
            }
            for line, probability in direct_markets["over_probabilities"].items()
        ]
        st.dataframe(pd.DataFrame(over_rows), use_container_width=True, hide_index=True)
    else:
        over_probability = safe_numeric(direct_markets.get("selected_over_probability"))
        if pd.notna(over_probability):
            line = direct_markets.get("selected_over_line")
            over_1, over_2 = st.columns(2)
            over_1.metric(f"Structural Over {line}", safe_percentage(over_probability))
            over_2.metric(f"Structural Under {line}", safe_percentage(1.0 - over_probability))
            st.caption("BO5 totals use the structural serve-point simulation.")
    structural = direct_markets.get("structural_baseline", {})
    if structural.get("available"):
        with st.expander("Structural serve-point simulation", expanded=False):
            structural_1, structural_2 = st.columns(2)
            structural_1.metric(
                "Expected total games",
                safe_number(structural.get("expected_total_games"), 1),
            )
            structural_2.metric(
                "Simulation P1 match win",
                safe_percentage(structural.get("match_win_player_1")),
            )
            scores = structural.get("set_score_probabilities", {})
            if scores:
                st.dataframe(
                    pd.DataFrame([
                        {"Set score": score, "Probability": safe_percentage(probability)}
                        for score, probability in scores.items()
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )
else:
    error = prediction.get("market_prediction_error") or direct_markets.get("reason")
    st.warning(
        "The optional direct-market bundle is unavailable for this prediction. "
        f"Match-winner output remains available. Detail: {error or 'not provided'}"
    )

freshness = get_data_freshness()
if freshness.get("available"):
    state_date = pd.to_datetime(freshness["state_as_of_date"]).date()
    st.caption(
        f"Player state data through {state_date}. Data age: "
        f"{freshness['freshness_days']} day(s). Status: {freshness['status']}."
    )
    if freshness["status"] == "STALE":
        st.warning("Recent activity may be incomplete because the current-state data is stale.")


with st.expander("Confidence Drivers", expanded=False):
    st.dataframe(pd.DataFrame({
        "Driver": ["Minimum Career Matches", "Minimum Surface Matches", "Minimum Statistical Matches", "Head-to-Head Matches"],
        "Value": [confidence["career_matches_min"], confidence["surface_matches_min"], confidence["stat_matches_min"], confidence["h2h_matches"]],
    }), use_container_width=True, hide_index=True)



# ---------------------------------------------------------------------------
# Immediate explanation card
# ---------------------------------------------------------------------------
def first_snapshot_numeric(snapshot: pd.Series, names: list[str]) -> float:
    for name in names:
        value = safe_numeric(snapshot_value(snapshot, name))
        if pd.notna(value):
            return value
    return np.nan


def metric_row(
    label: str,
    left: float,
    right: float,
    formatter,
    higher_is_better: bool = True,
    note: str = "",
) -> dict[str, Any]:
    left_number = safe_numeric(left)
    right_number = safe_numeric(right)
    if pd.isna(left_number) or pd.isna(right_number):
        advantage = "Insufficient data"
    elif np.isclose(left_number, right_number, rtol=0.0, atol=1e-12):
        advantage = "Balanced"
    else:
        left_better = left_number > right_number if higher_is_better else left_number < right_number
        advantage = player_1_name if left_better else player_2_name
    return {
        "Metric": label,
        player_1_name: formatter(left_number),
        player_2_name: formatter(right_number),
        "Immediate comparison": advantage,
        "Why it matters": note,
    }


def advantage_strength(probability: float) -> str:
    favorite = max(float(probability), 1.0 - float(probability))
    if favorite < 0.55:
        return "Very balanced"
    if favorite < 0.60:
        return "Slight advantage"
    if favorite < 0.70:
        return "Moderate advantage"
    if favorite < 0.80:
        return "Clear favorite"
    return "Strong favorite"


scheduled_fixture = prediction.get("scheduled_fixture") or {}
reliability = scheduled_fixture.get("reliability") or st.session_state.get(
    "upcoming_match_selection", {}
).get("reliability")

favorite_name = player_1_name if prob_1 >= prob_2 else player_2_name
favorite_probability = max(prob_1, prob_2)
edge_pp = abs(favorite_probability - 0.5) * 100.0

st.subheader("How to read this prediction")
with st.container(border=True):
    top_1, top_2, top_3, top_4 = st.columns(4)
    top_1.metric("Estimated favorite", favorite_name)
    top_2.metric("Win probability", safe_percentage(favorite_probability))
    top_3.metric("Advantage strength", advantage_strength(favorite_probability))
    top_4.metric("Edge over 50/50", f"+{edge_pp:.1f} pp")

    if isinstance(reliability, dict) and reliability.get("available"):
        observed_rate = float(reliability["observed_win_rate"])
        sample_size = int(reliability["sample_size"])
        gap = observed_rate - float(reliability["mean_predicted_probability"])
        st.markdown(
            f"**Historical comparison.** The model gives **{favorite_name} "
            f"{favorite_probability:.1%}**. In **{sample_size:,}** out-of-sample "
            f"matches with a similar probability band and the best available "
            f"context (**{reliability['profile_scope']}**), the model favorite won "
            f"**{observed_rate:.1%}** of the time. The historical calibration gap "
            f"was **{gap:+.1%}**."
        )
        st.caption(
            f"Band {reliability['probability_band']} · 95% observed interval "
            f"{reliability['confidence_lower']:.1%}–{reliability['confidence_upper']:.1%} · "
            f"Reliability {reliability['reliability_label']}. Reliability measures "
            "historical calibration, not certainty for this individual match."
        )
    else:
        st.info(
            "No historical reliability profile was transferred for this prediction. "
            "The probability remains valid, but there is no contextual observed-rate "
            "comparison on this screen."
        )

    if favorite_probability < 0.60:
        st.warning(
            "This is a close match. A high reliability label can mean that probabilities "
            "around this level were historically well calibrated; it does not turn a "
            "55%-45% estimate into a strong favorite."
        )

# Values are current pre-match snapshots. They explain the matchup context but
# are not a SHAP attribution of the model output.
overall_elo_1 = first_snapshot_numeric(player_1, ["elo_before"])
overall_elo_2 = first_snapshot_numeric(player_2, ["elo_before"])
surface_elo_1 = first_snapshot_numeric(player_1, ["surface_elo_before"])
surface_elo_2 = first_snapshot_numeric(player_2, ["surface_elo_before"])
serve_1 = first_snapshot_numeric(player_1, [
    "service_points_won_ewma_long_before", "service_points_won_pct_before"
])
serve_2 = first_snapshot_numeric(player_2, [
    "service_points_won_ewma_long_before", "service_points_won_pct_before"
])
return_1 = first_snapshot_numeric(player_1, [
    "return_points_won_ewma_long_before", "return_points_won_pct_before"
])
return_2 = first_snapshot_numeric(player_2, [
    "return_points_won_ewma_long_before", "return_points_won_pct_before"
])
momentum_30_1 = first_snapshot_numeric(player_1, ["momentum_ewma_30d_before"])
momentum_30_2 = first_snapshot_numeric(player_2, ["momentum_ewma_30d_before"])
momentum_90_1 = first_snapshot_numeric(player_1, ["momentum_ewma_90d_before"])
momentum_90_2 = first_snapshot_numeric(player_2, ["momentum_ewma_90d_before"])
game_share_1 = first_snapshot_numeric(player_1, ["games_won_share_last_10_before"])
game_share_2 = first_snapshot_numeric(player_2, ["games_won_share_last_10_before"])
performance_1 = first_snapshot_numeric(player_1, ["performance_vs_expected_ewma_before"])
performance_2 = first_snapshot_numeric(player_2, ["performance_vs_expected_ewma_before"])
rest_1 = first_snapshot_numeric(player_1, ["rest_days_before", "days_since_last_match"])
rest_2 = first_snapshot_numeric(player_2, ["rest_days_before", "days_since_last_match"])
matches_7_1 = first_snapshot_numeric(player_1, ["matches_last_7d_before"])
matches_7_2 = first_snapshot_numeric(player_2, ["matches_last_7d_before"])

immediate_rows = [
    metric_row("Overall Elo", overall_elo_1, overall_elo_2, lambda value: safe_number(value, 0), True, "Overall opponent-adjusted strength."),
    metric_row(f"{surface} Elo", surface_elo_1, surface_elo_2, lambda value: safe_number(value, 0), True, "Strength on the selected surface."),
    metric_row("Service points won, long-horizon EWMA", serve_1, serve_2, lambda value: safe_percentage(value), True, "Exponentially weighted long-horizon service efficiency from the current-state pipeline; this is not a fixed 5- or 10-match window."),
    metric_row("Return points won, long-horizon EWMA", return_1, return_2, lambda value: safe_percentage(value), True, "Exponentially weighted long-horizon return efficiency from the current-state pipeline; this is not a fixed 5- or 10-match window."),
    metric_row("Momentum, 30 days", momentum_30_1, momentum_30_2, lambda value: safe_number(value, 3), True, "Short-term performance trend."),
    metric_row("Momentum, 90 days", momentum_90_1, momentum_90_2, lambda value: safe_number(value, 3), True, "More stable medium-term trend."),
    metric_row("Games won share, last 10", game_share_1, game_share_2, lambda value: safe_percentage(value), True, "Dominance beyond simple wins and losses."),
    metric_row("Performance vs expected", performance_1, performance_2, lambda value: safe_number(value, 3), True, "Recent performance adjusted for opponent strength."),
    metric_row("Rest days", rest_1, rest_2, lambda value: safe_number(value, 0), True, "Recovery before the scheduled match."),
    metric_row("Matches in last 7 days", matches_7_1, matches_7_2, lambda value: safe_number(value, 0), False, "Recent workload; fewer is not always better, but high load can increase uncertainty."),
]

st.subheader("Immediate player comparison")
st.dataframe(pd.DataFrame(immediate_rows), use_container_width=True, hide_index=True)
st.caption(
    "These indicators summarize the most interpretable pre-match differences. "
    "They are contextual comparisons, not a causal decomposition of the XGBoost prediction."
)

# ---------------------------------------------------------------------------
# Tipster analysis
# ---------------------------------------------------------------------------
def tipster_metric(
    block: str,
    metric: str,
    left: Any,
    right: Any,
    formatter,
    higher_is_better: bool | None,
    interpretation: str,
    scope: str,
) -> dict[str, Any]:
    left_number = safe_numeric(left)
    right_number = safe_numeric(right)
    if higher_is_better is None or pd.isna(left_number) or pd.isna(right_number):
        advantage = "Context only" if higher_is_better is None else "Insufficient data"
    elif np.isclose(left_number, right_number, rtol=0.0, atol=1e-12):
        advantage = "Balanced"
    else:
        left_better = left_number > right_number if higher_is_better else left_number < right_number
        advantage = player_1_name if left_better else player_2_name
    return {
        "Block": block,
        "Metric": metric,
        player_1_name: formatter(left_number),
        player_2_name: formatter(right_number),
        "Advantage": advantage,
        "Window / scope": scope,
        "Interpretation": interpretation,
    }


def momentum_confidence_label(value: Any) -> str:
    number = safe_numeric(value)
    if pd.isna(number):
        return "-"
    if number >= 0.70:
        return f"High ({number:.2f})"
    if number >= 0.40:
        return f"Moderate ({number:.2f})"
    return f"Limited ({number:.2f})"


def signed_number(value: Any, decimals: int = 3) -> str:
    number = safe_numeric(value)
    return "-" if pd.isna(number) else f"{number:+.{decimals}f}"


def surface_history_value(snapshot: pd.Series, generic: str, surface_specific: str) -> float:
    return first_snapshot_numeric(snapshot, [surface_specific, generic])


# Momentum v6 remains optional until a v6 state file has been rebuilt. The
# section degrades gracefully and clearly labels unavailable evidence.
result_residual_30_1 = first_snapshot_numeric(player_1, ["result_residual_ewma_30d_before"])
result_residual_30_2 = first_snapshot_numeric(player_2, ["result_residual_ewma_30d_before"])
result_residual_90_1 = first_snapshot_numeric(player_1, ["result_residual_ewma_90d_before"])
result_residual_90_2 = first_snapshot_numeric(player_2, ["result_residual_ewma_90d_before"])
surface_residual_1 = first_snapshot_numeric(player_1, ["surface_result_residual_ewma_90d_before"])
surface_residual_2 = first_snapshot_numeric(player_2, ["surface_result_residual_ewma_90d_before"])
game_residual_1 = first_snapshot_numeric(player_1, ["game_share_residual_ewma_30d_before"])
game_residual_2 = first_snapshot_numeric(player_2, ["game_share_residual_ewma_30d_before"])
serve_residual_1 = first_snapshot_numeric(player_1, ["serve_residual_ewma_30d_before"])
serve_residual_2 = first_snapshot_numeric(player_2, ["serve_residual_ewma_30d_before"])
return_residual_1 = first_snapshot_numeric(player_1, ["return_residual_ewma_30d_before"])
return_residual_2 = first_snapshot_numeric(player_2, ["return_residual_ewma_30d_before"])
dominance_1 = first_snapshot_numeric(player_1, ["dominance_ratio_ewma_30d_before"])
dominance_2 = first_snapshot_numeric(player_2, ["dominance_ratio_ewma_30d_before"])
momentum_neff_1 = first_snapshot_numeric(player_1, ["momentum_effective_sample_before"])
momentum_neff_2 = first_snapshot_numeric(player_2, ["momentum_effective_sample_before"])
momentum_confidence_1 = first_snapshot_numeric(player_1, ["momentum_confidence_before"])
momentum_confidence_2 = first_snapshot_numeric(player_2, ["momentum_confidence_before"])
elo_change_90_1 = first_snapshot_numeric(player_1, ["elo_change_90d_before", "elo_change_last_5_before"])
elo_change_90_2 = first_snapshot_numeric(player_2, ["elo_change_90d_before", "elo_change_last_5_before"])
avg_opponent_elo_1 = first_snapshot_numeric(player_1, ["avg_opponent_elo_last_5_before"])
avg_opponent_elo_2 = first_snapshot_numeric(player_2, ["avg_opponent_elo_last_5_before"])
second_serve_1 = first_snapshot_numeric(player_1, ["second_serve_win_ewma_before", "second_serve_win_pct_before"])
second_serve_2 = first_snapshot_numeric(player_2, ["second_serve_win_ewma_before", "second_serve_win_pct_before"])
minutes_7_1 = first_snapshot_numeric(player_1, ["minutes_last_7d_before"])
minutes_7_2 = first_snapshot_numeric(player_2, ["minutes_last_7d_before"])
sets_7_1 = first_snapshot_numeric(player_1, ["sets_last_7d_before"])
sets_7_2 = first_snapshot_numeric(player_2, ["sets_last_7d_before"])
rank_1 = first_snapshot_numeric(player_1, ["player_rank"])
rank_2 = first_snapshot_numeric(player_2, ["player_rank"])
rank_points_1 = first_snapshot_numeric(player_1, ["player_rank_points"])
rank_points_2 = first_snapshot_numeric(player_2, ["player_rank_points"])
career_high_1 = first_snapshot_numeric(player_1, ["career_high_rank_before"])
career_high_2 = first_snapshot_numeric(player_2, ["career_high_rank_before"])
surface_win_1 = first_snapshot_numeric(player_1, ["surface_win_rate_before"])
surface_win_2 = first_snapshot_numeric(player_2, ["surface_win_rate_before"])

st.subheader("Tipster Analysis")
st.caption(
    "Professional-style pre-match checklist based on structural strength, opponent-adjusted "
    "form, surface, serve and return production, workload, matchup and price. This is "
    "probabilistic analysis. No combination removes uncertainty or guarantees a result or return."
)

available_v6 = sum(
    pd.notna(value)
    for value in (
        result_residual_30_1, result_residual_30_2, serve_residual_1,
        serve_residual_2, momentum_confidence_1, momentum_confidence_2,
    )
)
if available_v6 == 0:
    st.info(
        "Momentum v6 is not yet available in the active player state. The section uses the "
        "validated v5 indicators and will automatically populate opponent-adjusted residuals "
        "after the v6 Gold and current-state files are rebuilt."
    )

with st.container(border=True):
    overview_1, overview_2, overview_3, overview_4 = st.columns(4)
    overview_1.metric("Model favorite", favorite_name)
    overview_2.metric("Model probability", safe_percentage(favorite_probability))
    overview_3.metric(
        "Surface Elo edge",
        signed_number(surface_elo_1 - surface_elo_2, 0)
        if pd.notna(surface_elo_1) and pd.notna(surface_elo_2) else "-",
        help=f"Positive favors {player_1_name}; negative favors {player_2_name}.",
    )
    confidence_min = min(
        [value for value in (momentum_confidence_1, momentum_confidence_2) if pd.notna(value)],
        default=np.nan,
    )
    overview_4.metric("Momentum evidence", momentum_confidence_label(confidence_min))

    tipster_rows = [
        tipster_metric("1. Structural level", "Overall Elo", overall_elo_1, overall_elo_2, lambda v: safe_number(v, 0), True, "Opponent-adjusted long-run strength.", "Current pre-match state"),
        tipster_metric("1. Structural level", f"{surface} Elo", surface_elo_1, surface_elo_2, lambda v: safe_number(v, 0), True, "Primary structural rating for the selected surface.", f"Current {surface} state"),
        tipster_metric("1. Structural level", "Elo change", elo_change_90_1, elo_change_90_2, signed_number, True, "Recent movement in latent strength.", "90 days; fallback to last 5 matches"),
        tipster_metric("1. Structural level", "ATP ranking", rank_1, rank_2, lambda v: safe_number(v, 0), False, "Ranking is slower-moving than Elo but remains informative.", "Current state"),
        tipster_metric("1. Structural level", "Ranking points", rank_points_1, rank_points_2, lambda v: safe_number(v, 0), True, "Current ranking-point strength.", "Current state"),
        tipster_metric("1. Structural level", "Career-high ranking", career_high_1, career_high_2, lambda v: safe_number(v, 0), False, "Historical peak ranking; context rather than current form.", "Career"),
        tipster_metric("2. Opponent-adjusted form", "Result residual", result_residual_30_1, result_residual_30_2, signed_number, True, "Actual result minus pre-match Elo expectation. Positive means over-performance.", "EWMA, 30-day half-life"),
        tipster_metric("2. Opponent-adjusted form", "Result residual, stable", result_residual_90_1, result_residual_90_2, signed_number, True, "More stable opponent-adjusted result signal.", "EWMA, 90-day half-life"),
        tipster_metric("2. Opponent-adjusted form", "Game-share residual", game_residual_1, game_residual_2, signed_number, True, "Game dominance relative to Elo-derived expectation.", "EWMA, 30-day half-life"),
        tipster_metric("2. Opponent-adjusted form", "Average recent opponent Elo", avg_opponent_elo_1, avg_opponent_elo_2, lambda v: safe_number(v, 0), None, "Schedule quality is context, not an automatic advantage.", "Last 5 opponents"),
        tipster_metric("2. Opponent-adjusted form", "Effective sample", momentum_neff_1, momentum_neff_2, lambda v: safe_number(v, 1), None, "Recency-weighted amount of evidence behind the momentum signal.", "30-day EWMA weights"),
        tipster_metric("2. Opponent-adjusted form", "Momentum confidence", momentum_confidence_1, momentum_confidence_2, momentum_confidence_label, None, "Higher confidence means more evidence, coverage and stability, not better form.", "0 to 1"),
        tipster_metric("3. Surface", f"{surface} result residual", surface_residual_1, surface_residual_2, signed_number, True, "Opponent-adjusted form on the selected surface.", "EWMA, 90-day half-life"),
        tipster_metric("3. Surface", f"{surface} win rate", surface_win_1, surface_win_2, safe_percentage, True, "Historical win rate in the selected surface state.", "Current state"),
        tipster_metric("4. Serve", "Service points won", serve_1, serve_2, safe_percentage, True, "Core service efficiency.", "Long-horizon EWMA"),
        tipster_metric("4. Serve", "Serve residual vs opponent", serve_residual_1, serve_residual_2, signed_number, True, "Actual service production relative to opponent return quality.", "EWMA, 30-day half-life"),
        tipster_metric("4. Serve", "Second-serve points won", second_serve_1, second_serve_2, safe_percentage, True, "Resilience when the first serve misses.", "Long-horizon EWMA"),
        tipster_metric("5. Return", "Return points won", return_1, return_2, safe_percentage, True, "Core return efficiency.", "Long-horizon EWMA"),
        tipster_metric("5. Return", "Return residual vs opponent", return_residual_1, return_residual_2, signed_number, True, "Actual return production relative to opponent service quality.", "EWMA, 30-day half-life"),
        tipster_metric("6. Point and game dominance", "Dominance ratio", dominance_1, dominance_2, lambda v: safe_number(v, 3), True, "Return points won divided by service points lost. Above 1 is positive.", "EWMA, 30-day half-life"),
        tipster_metric("6. Point and game dominance", "Games won share", game_share_1, game_share_2, safe_percentage, True, "Captures score dominance beyond wins and losses.", "Last 10 matches"),
        tipster_metric("7. Fatigue and readiness", "Rest days", rest_1, rest_2, lambda v: safe_number(v, 0), None, "Very little rest can imply fatigue; long inactivity can imply lack of rhythm.", "Before prediction date"),
        tipster_metric("7. Fatigue and readiness", "Matches in last 7 days", matches_7_1, matches_7_2, lambda v: safe_number(v, 0), False, "High match density may increase fatigue; zero can indicate inactivity.", "Trailing 7 days"),
        tipster_metric("7. Fatigue and readiness", "Minutes in last 7 days", minutes_7_1, minutes_7_2, lambda v: safe_number(v, 0), False, "Recorded recent physical workload.", "Trailing 7 days"),
        tipster_metric("7. Fatigue and readiness", "Sets in last 7 days", sets_7_1, sets_7_2, lambda v: safe_number(v, 0), False, "Recent set workload.", "Trailing 7 days"),
        tipster_metric("9. Head-to-head", "H2H wins", h2h["player_1_wins"], h2h["player_2_wins"], lambda v: safe_number(v, 0), True, "Use as secondary context; small or old samples should receive limited weight.", "Completed direct meetings"),
        tipster_metric("12. Model benchmark", "Model probability", prob_1, prob_2, safe_percentage, True, "Calibrated model estimate when the active policy is calibrated.", SELECTED_PROBABILITY),
        tipster_metric("12. Model benchmark", "Model fair odds", fair_odds_1, fair_odds_2, lambda v: safe_number(v, 2), False, "Inverse model probability before bookmaker margin.", "Current prediction"),
    ]
    st.dataframe(pd.DataFrame(tipster_rows), use_container_width=True, hide_index=True)

    with st.expander("Interpretation and limitations", expanded=False):
        st.markdown(
            f"""
            **Practical priority**

            1. {surface} Elo and structural strength.
            2. Service and return point production.
            3. Opponent-adjusted recent form and its evidence level.
            4. Surface suitability.
            5. Workload, rest and inactivity.
            6. Contextual H2H, conditions and tournament history.
            7. Market no-vig probability as an external benchmark when odds are supplied.

            **How to read negative Momentum v6 values**

            A negative result, game, serve or return residual means the player performed below
            the pre-match expectation in that dimension. It does not mean the player is weak in
            absolute terms. A high-Elo player can have negative recent residual form and still be
            the structural favorite.

            **Data not inferred by this application**

            Injuries, medical timeouts, travel distance, time-zone changes, weather, altitude,
            ball type, tactical style and motivation are not inferred unless a verified data source
            is present. Missing qualitative context is shown as a limitation rather than invented.
            """
        )

st.subheader("Recent Tournament Form")
st.caption(
    "The last three completed tournaments for each player, ordered from most recent. "
    "Percentages are weighted from recorded point totals. Totals and per-match values "
    "describe tournament workload and execution, not career averages."
)
if not callable(get_players_recent_tournament_statistics):
    st.warning("Install the updated players_adapted.py to enable recent-tournament form.")
else:
    recent_tournaments = get_players_recent_tournament_statistics(
        player_1_id=player_1_id,
        player_2_id=player_2_id,
        last_n_tournaments=3,
        as_of_date=prediction.get("prediction_date"),
    )
    if recent_tournaments.empty:
        st.info("No recent tournament records were found for either player.")
    else:
        percent_columns = {
            "service_points_won_pct", "return_points_won_pct", "first_serve_in_pct",
            "first_serve_win_pct", "second_serve_win_pct", "service_hold_rate",
            "score_coverage", "point_stat_coverage", "minutes_coverage",
        }
        integer_columns = {
            "matches", "wins", "break_points_saved", "break_points_faced",
            "breaks_conceded", "service_games_played", "service_games_held",
            "aces", "double_faults", "games_played", "games_won", "games_lost",
            "sets_played", "sets_won", "sets_lost", "minutes_accumulated",
        }
        display_columns = [
            "recency_rank", "tournament", "surface", "first_match_date", "last_match_date",
            "matches", "wins", "service_points_won_pct", "return_points_won_pct",
            "first_serve_in_pct", "first_serve_win_pct", "second_serve_win_pct",
            "break_points_faced", "break_points_saved", "break_points_saved_per_match",
            "breaks_conceded", "service_games_played", "service_games_held", "service_hold_rate",
            "aces", "aces_per_match", "double_faults", "double_faults_per_match",
            "games_played", "games_played_per_match", "games_won", "games_won_per_match",
            "games_lost", "games_lost_per_match", "sets_played", "sets_played_per_match",
            "sets_won", "sets_won_per_match", "sets_lost", "sets_lost_per_match",
            "minutes_accumulated", "minutes_per_match", "score_coverage",
            "point_stat_coverage", "minutes_coverage",
        ]
        labels = {
            "recency_rank": "Recent #", "tournament": "Tournament", "surface": "Surface",
            "first_match_date": "First date", "last_match_date": "Last date",
            "matches": "Matches", "wins": "Wins",
            "service_points_won_pct": "Service points won", "return_points_won_pct": "Return points won",
            "first_serve_in_pct": "First serve in", "first_serve_win_pct": "First serve points won",
            "second_serve_win_pct": "Second serve points won", "break_points_faced": "BP faced",
            "break_points_saved": "BP saved", "break_points_saved_per_match": "BP saved / match",
            "breaks_conceded": "Breaks conceded", "service_games_played": "Service games",
            "service_games_held": "Service games held", "service_hold_rate": "Service hold rate",
            "aces": "Aces", "aces_per_match": "Aces / match", "double_faults": "Double faults",
            "double_faults_per_match": "DF / match", "games_played": "Games played",
            "games_played_per_match": "Games / match", "games_won": "Games won",
            "games_won_per_match": "Games won / match", "games_lost": "Games lost",
            "games_lost_per_match": "Games lost / match", "sets_played": "Sets played",
            "sets_played_per_match": "Sets / match", "sets_won": "Sets won",
            "sets_won_per_match": "Sets won / match", "sets_lost": "Sets lost",
            "sets_lost_per_match": "Sets lost / match", "minutes_accumulated": "Minutes",
            "minutes_per_match": "Minutes / match", "score_coverage": "Score coverage",
            "point_stat_coverage": "Point-stat coverage", "minutes_coverage": "Minutes coverage",
        }
        recent_columns = st.columns(2)
        for container, pid, pname in (
            (recent_columns[0], player_1_id, player_1_name),
            (recent_columns[1], player_2_id, player_2_name),
        ):
            with container:
                st.markdown(f"**{pname}**")
                player_recent = recent_tournaments.loc[
                    pd.to_numeric(recent_tournaments["player_id"], errors="coerce").eq(int(pid))
                ].copy()
                if player_recent.empty:
                    st.info("No recent tournaments found.")
                    continue
                shown = player_recent[[column for column in display_columns if column in player_recent]].copy()
                for column in shown.columns:
                    if column in percent_columns:
                        shown[column] = shown[column].map(safe_percentage)
                    elif column in integer_columns:
                        shown[column] = shown[column].map(safe_integer)
                    elif column.endswith("_per_match"):
                        shown[column] = shown[column].map(lambda value: safe_number(value, 2))
                    elif column in {"first_match_date", "last_match_date"}:
                        shown[column] = pd.to_datetime(shown[column], errors="coerce").dt.date
                shown = shown.rename(columns=labels)
                st.dataframe(shown, use_container_width=True, hide_index=True)

st.subheader("Current Tournament Comparison")
current_tournament_name = prediction.get("tournament_history_name")
current_tournament_id = prediction.get("tournament_history_id")
if not current_tournament_name:
    st.info(
        "Current-tournament statistics are available for fixtures opened from "
        "Upcoming Matches. Manual mode does not currently select a tournament."
    )
elif not callable(get_players_current_tournament_statistics):
    st.warning(
        "Install the updated players_adapted.py to enable current-tournament statistics."
    )
else:
    current_tournament = get_players_current_tournament_statistics(
        player_1_id=player_1_id,
        player_2_id=player_2_id,
        tournament_name=str(current_tournament_name),
        tournament_id=current_tournament_id,
        as_of_date=prediction.get("prediction_date"),
    )
    if current_tournament.empty:
        st.info(
            f"No completed matches were found for either player in the current "
            f"edition of {current_tournament_name}."
        )
    else:
        tournament_by_player = {
            int(row["player_id"]): row
            for _, row in current_tournament.iterrows()
        }
        row_1 = tournament_by_player.get(int(player_1_id), pd.Series(dtype="object"))
        row_2 = tournament_by_player.get(int(player_2_id), pd.Series(dtype="object"))
        def tournament_value(row: pd.Series, key: str) -> Any:
            return row.get(key, np.nan) if not row.empty else np.nan
        integer_formatter = lambda value: safe_number(value, 0)
        decimal_formatter = lambda value: safe_number(value, 2)
        current_rows = [
            metric_row("Matches completed", tournament_value(row_1, "matches"), tournament_value(row_2, "matches"), integer_formatter, True, "Completed matches in this tournament edition."),
            metric_row("Wins", tournament_value(row_1, "wins"), tournament_value(row_2, "wins"), integer_formatter, True, "Victories already recorded in the current tournament."),
            metric_row("Service points won", tournament_value(row_1, "service_points_won_pct"), tournament_value(row_2, "service_points_won_pct"), safe_percentage, True, "Weighted service efficiency from recorded point totals."),
            metric_row("Return points won", tournament_value(row_1, "return_points_won_pct"), tournament_value(row_2, "return_points_won_pct"), safe_percentage, True, "Weighted return efficiency from recorded point totals."),
            metric_row("First serve in", tournament_value(row_1, "first_serve_in_pct"), tournament_value(row_2, "first_serve_in_pct"), safe_percentage, True, "First-serve consistency in completed matches."),
            metric_row("First serve points won", tournament_value(row_1, "first_serve_win_pct"), tournament_value(row_2, "first_serve_win_pct"), safe_percentage, True, "Effectiveness behind the first serve."),
            metric_row("Second serve points won", tournament_value(row_1, "second_serve_win_pct"), tournament_value(row_2, "second_serve_win_pct"), safe_percentage, True, "Second-serve resilience under pressure."),
            metric_row("Break points faced, total", tournament_value(row_1, "break_points_faced"), tournament_value(row_2, "break_points_faced"), integer_formatter, False, "Total break points faced. Lower exposure is generally preferable."),
            metric_row("Break points saved, total", tournament_value(row_1, "break_points_saved"), tournament_value(row_2, "break_points_saved"), integer_formatter, True, "Total break points saved, shown as volume rather than a potentially misleading percentage."),
            metric_row("Break points saved per match", tournament_value(row_1, "break_points_saved_per_match"), tournament_value(row_2, "break_points_saved_per_match"), decimal_formatter, True, "Average number of break points saved per completed match."),
            metric_row("Breaks conceded, total", tournament_value(row_1, "breaks_conceded"), tournament_value(row_2, "breaks_conceded"), integer_formatter, False, "Break points faced minus break points saved."),
            metric_row("Service games played", tournament_value(row_1, "service_games_played"), tournament_value(row_2, "service_games_played"), integer_formatter, False, "Recorded service games played; a larger value can reflect greater workload."),
            metric_row("Service games held", tournament_value(row_1, "service_games_held"), tournament_value(row_2, "service_games_held"), integer_formatter, True, "Estimated holds: service games played minus breaks conceded."),
            metric_row("Service hold rate", tournament_value(row_1, "service_hold_rate"), tournament_value(row_2, "service_hold_rate"), safe_percentage, True, "Estimated share of recorded service games held."),
            metric_row("Aces, total", tournament_value(row_1, "aces"), tournament_value(row_2, "aces"), integer_formatter, True, "Total aces in completed matches."),
            metric_row("Aces per match", tournament_value(row_1, "aces_per_match"), tournament_value(row_2, "aces_per_match"), decimal_formatter, True, "Average aces per completed match."),
            metric_row("Double faults, total", tournament_value(row_1, "double_faults"), tournament_value(row_2, "double_faults"), integer_formatter, False, "Total double faults; lower is preferable."),
            metric_row("Double faults per match", tournament_value(row_1, "double_faults_per_match"), tournament_value(row_2, "double_faults_per_match"), decimal_formatter, False, "Average double faults per completed match; lower is preferable."),
            metric_row("Games played, total", tournament_value(row_1, "games_played"), tournament_value(row_2, "games_played"), integer_formatter, False, "Total games recorded in all parsed scores; a larger value indicates more match workload."),
            metric_row("Games played per match", tournament_value(row_1, "games_played_per_match"), tournament_value(row_2, "games_played_per_match"), decimal_formatter, False, "Average games played per completed match."),
            metric_row("Games won, total", tournament_value(row_1, "games_won"), tournament_value(row_2, "games_won"), integer_formatter, True, "Total games won from recorded scores."),
            metric_row("Games won per match", tournament_value(row_1, "games_won_per_match"), tournament_value(row_2, "games_won_per_match"), decimal_formatter, True, "Average games won per completed match."),
            metric_row("Games lost, total", tournament_value(row_1, "games_lost"), tournament_value(row_2, "games_lost"), integer_formatter, False, "Total games lost from recorded scores; lower is preferable."),
            metric_row("Games lost per match", tournament_value(row_1, "games_lost_per_match"), tournament_value(row_2, "games_lost_per_match"), decimal_formatter, False, "Average games lost per completed match; lower is preferable."),
            metric_row("Sets played, total", tournament_value(row_1, "sets_played"), tournament_value(row_2, "sets_played"), integer_formatter, False, "Total played set-score tokens, including unfinished retirement sets."),
            metric_row("Sets played per match", tournament_value(row_1, "sets_played_per_match"), tournament_value(row_2, "sets_played_per_match"), decimal_formatter, False, "Average sets played per completed match."),
            metric_row("Sets won, total", tournament_value(row_1, "sets_won"), tournament_value(row_2, "sets_won"), integer_formatter, True, "Total sets won from recorded scores."),
            metric_row("Sets won per match", tournament_value(row_1, "sets_won_per_match"), tournament_value(row_2, "sets_won_per_match"), decimal_formatter, True, "Average sets won per completed match."),
            metric_row("Sets lost, total", tournament_value(row_1, "sets_lost"), tournament_value(row_2, "sets_lost"), integer_formatter, False, "Total sets lost from recorded scores; lower is preferable."),
            metric_row("Sets lost per match", tournament_value(row_1, "sets_lost_per_match"), tournament_value(row_2, "sets_lost_per_match"), decimal_formatter, False, "Average sets lost per completed match; lower is preferable."),
            metric_row("Accumulated minutes", tournament_value(row_1, "minutes_accumulated"), tournament_value(row_2, "minutes_accumulated"), integer_formatter, False, "Recorded match workload in minutes; lower can indicate less accumulated fatigue."),
            metric_row("Minutes per match", tournament_value(row_1, "minutes_per_match"), tournament_value(row_2, "minutes_per_match"), lambda value: safe_number(value, 1), False, "Average duration where minutes are available."),
        ]
        st.caption(
            f"Completed matches in the current edition of {current_tournament_name}. "
            "Break-point figures are shown as totals and per-match volumes. "
            "Games and sets are parsed from recorded scores."
        )
        st.dataframe(
            pd.DataFrame(current_rows),
            use_container_width=True,
            hide_index=True,
        )
        coverage_rows = []
        for player_id, player_name in (
            (player_1_id, player_1_name),
            (player_2_id, player_2_name),
        ):
            row = tournament_by_player.get(
                int(player_id), pd.Series(dtype="object")
            )
            coverage_rows.append({
                "Player": player_name,
                "Matches": safe_integer(tournament_value(row, "matches")),
                "Matches with score": safe_integer(tournament_value(row, "matches_with_score")),
                "Matches with point stats": safe_integer(tournament_value(row, "stat_matches")),
                "Matches with service-game stats": safe_integer(tournament_value(row, "service_game_stat_matches")),
                "Matches with minutes": safe_integer(tournament_value(row, "matches_with_minutes")),
                "Score coverage": safe_percentage(tournament_value(row, "score_coverage")),
                "Point-stat coverage": safe_percentage(tournament_value(row, "point_stat_coverage")),
                "Minutes coverage": safe_percentage(tournament_value(row, "minutes_coverage")),
            })
        st.dataframe(
            pd.DataFrame(coverage_rows),
            use_container_width=True,
            hide_index=True,
        )

st.subheader("Players Form and Comparison")
with st.container(border=True):
    scope_col, surface_col, count_col = st.columns([2, 3, 1])
    with scope_col:
        analysis_period = st.radio("Period", ["Current Season", "Career"], horizontal=True, key="prediction_analysis_period")
    with surface_col:
        analysis_surface = st.radio("Analysis Surface", ANALYSIS_SURFACES, horizontal=True, key="prediction_analysis_surface")
    with count_col:
        recent_match_count = st.selectbox("Recent Matches", [5, 10, 15, 20], index=1, key="prediction_recent_count")


comparison_1 = comparison_values(
    player_1_id, player_1, analysis_period, analysis_surface, recent_match_count
)
comparison_2 = comparison_values(
    player_2_id, player_2, analysis_period, analysis_surface, recent_match_count
)
comparison_definitions = [
    ("Rank", "rank", "number", "Current state"),
    ("ATP Points", "rank_points", "number", "Current state"),
    ("Age", "age", "number", "Current state"),
    ("Height", "height", "number", "Current state"),
    ("Career Matches", "career_matches", "number", "Current state"),
    ("Career Win Rate", "career_win_rate", "percentage", "Current state"),
    ("Global Elo", "global_elo", "number", "Current state"),
    (f"{analysis_surface} Matches", "surface_matches", "number", f"{analysis_period} · {analysis_surface}"),
    (f"{analysis_surface} Win Rate", "surface_win_rate", "percentage", f"{analysis_period} · {analysis_surface}"),
    (f"{analysis_surface} Elo", "surface_elo", "number", "Current selected-surface state"),
    ("Last 5 Win Rate", "last_5_win_rate", "percentage", f"Last 5 · {analysis_period} · {analysis_surface}"),
    ("Last 10 Win Rate", "last_10_win_rate", "percentage", f"Last 10 · {analysis_period} · {analysis_surface}"),
    (f"Last {recent_match_count} Win Rate", "recent_win_rate", "percentage", f"Selected recent window · {analysis_period} · {analysis_surface}"),
    ("Days Since Last Recorded Match", "days_since_last_match", "number", "Calculated on prediction date"),
    ("Service Points Won", "service_points_won", "percentage", f"Last {recent_match_count} matches · {analysis_period} · {analysis_surface}"),
    ("Return Points Won", "return_points_won", "percentage", f"Last {recent_match_count} matches · {analysis_period} · {analysis_surface}"),
]
comparison_rows = []
for label, key, kind, source in comparison_definitions:
    formatter = safe_percentage if kind == "percentage" else lambda value: safe_number(value, 1)
    comparison_rows.append({
        "Metric": label, "Scope / Source": source,
        player_1_name: formatter(comparison_1.get(key)),
        player_2_name: formatter(comparison_2.get(key)),
    })
st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)


st.subheader(f"Last {recent_match_count} Matches")
history_1 = filter_history(load_player_history(player_1_id), analysis_period, analysis_surface, recent_match_count)
history_2 = filter_history(load_player_history(player_2_id), analysis_period, analysis_surface, recent_match_count)
last_columns = [
    "match_date", "tourney_name", "tourney_level", "surface", "round",
    "opponent_name", "score", "result",
]
last_1, last_2 = st.columns(2)
for container, name, history in ((last_1, player_1_name, history_1), (last_2, player_2_name, history_2)):
    with container:
        st.write(name)
        if history.empty:
            st.info("No matches found for the selected filters.")
        else:
            st.dataframe(history[[column for column in last_columns if column in history]], use_container_width=True, hide_index=True)


st.subheader("Head-to-Head History")
h1, h2, h3 = st.columns(3)
h1.metric("Meetings", h2h["matches"])
h2.metric(f"{player_1_name} Wins", h2h["player_1_wins"])
h3.metric(f"{player_2_name} Wins", h2h["player_2_wins"])
if h2h["history"].empty:
    st.info("No previous meetings found.")
else:
    h2h_columns = [
        column for column in (
            "match_date", "tourney_name", "surface", "round",
            "winner_name", "loser_name", "score",
        ) if column in h2h["history"]
    ]
    st.dataframe(
        sort_match_history(h2h["history"], ascending=False)[h2h_columns],
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Recent History at This Tournament")
tournament_history_name = prediction.get("tournament_history_name")
tournament_history_id = prediction.get("tournament_history_id")
tournament_history_years = int(prediction.get("tournament_history_years", 10))
if not tournament_history_name:
    st.info(
        "Tournament history is available for fixtures opened from Upcoming Matches. "
        "Manual predictions do not currently select a specific tournament."
    )
elif not callable(get_players_tournament_history):
    st.warning(
        "players_adapted.py does not provide get_players_tournament_history. "
        "Install the updated data-access script."
    )
else:
    tournament_history = get_players_tournament_history(
        player_1_id=player_1_id,
        player_2_id=player_2_id,
        tournament_name=str(tournament_history_name),
        tournament_id=tournament_history_id,
        years=tournament_history_years,
        as_of_date=prediction.get("prediction_date"),
    )
    st.caption(
        f"Furthest recorded round by season at {tournament_history_name}. "
        f"Window: last {tournament_history_years} years."
    )
    if tournament_history.empty:
        st.info("No previous appearances were found for either player at this tournament.")
    else:
        tournament_columns = [
            "season", "player_name", "surface", "matches", "wins", "losses",
            "best_round", "outcome", "last_opponent", "last_result", "last_score",
        ]
        st.dataframe(
            tournament_history[[
                column for column in tournament_columns
                if column in tournament_history
            ]],
            use_container_width=True,
            hide_index=True,
        )
        summary_rows = []
        for player_id, player_name in (
            (player_1_id, player_1_name),
            (player_2_id, player_2_name),
        ):
            player_history = tournament_history.loc[
                pd.to_numeric(tournament_history["player_id"], errors="coerce").eq(player_id)
            ]
            if player_history.empty:
                summary_rows.append({
                    "Player": player_name,
                    "Appearances": 0,
                    "Best historical round": "-",
                    "Best outcome": "No recorded appearance",
                })
                continue
            best = player_history.sort_values(
                ["best_round_order", "season"],
                ascending=[False, False],
                kind="mergesort",
            ).iloc[0]
            summary_rows.append({
                "Player": player_name,
                "Appearances": int(player_history["season"].nunique()),
                "Best historical round": best.get("best_round", "-"),
                "Best outcome": best.get("outcome", "-"),
            })
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )
with st.expander("Market Comparison", expanded=False):
    with st.form("market_comparison_form"):
        market_1, market_2 = st.columns(2)
        with market_1:
            odds_1 = st.number_input(f"{player_1_name} Market Odds", min_value=1.01, value=float(round(fair_odds_1, 2)), step=0.01)
        with market_2:
            odds_2 = st.number_input(f"{player_2_name} Market Odds", min_value=1.01, value=float(round(fair_odds_2, 2)), step=0.01)
        compare_market = st.form_submit_button("Compare with Market", use_container_width=True)
    if compare_market:
        raw_1, raw_2 = 1 / odds_1, 1 / odds_2
        market_p1 = raw_1 / (raw_1 + raw_2)
        market_p2 = 1 - market_p1
        st.dataframe(pd.DataFrame({
            "Metric": ["Model Probability", "Market Probability, No Vig", "Difference", "Fair Odds", "Entered Market Odds"],
            player_1_name: [safe_percentage(prob_1), safe_percentage(market_p1), safe_percentage(prob_1 - market_p1), safe_number(fair_odds_1), safe_number(odds_1)],
            player_2_name: [safe_percentage(prob_2), safe_percentage(market_p2), safe_percentage(prob_2 - market_p2), safe_number(fair_odds_2), safe_number(odds_2)],
        }), use_container_width=True, hide_index=True)


st.subheader("Model Information")
symmetry = prediction["symmetry_diagnostics"]
st.dataframe(pd.DataFrame({
    "Item": ["Active Model Version", "Champion", "Model Mode", "Feature Count", "Categorical Features", "Probability Policy", "Calibration Applied", "Direct Markets", "Structural Simulator", "Missing Inference Features", "Orientation Consistency Gap"],
    "Value": [MODEL_VERSION, active_model["champion"], model_metadata.get("mode", "no_market"), len(MODEL_FEATURES), len(CATEGORICAL_FEATURES), SELECTED_PROBABILITY, APPLY_CALIBRATION, bool(direct_markets.get("available")), bool(direct_markets.get("structural_baseline", {}).get("available")), len(missing_features), safe_number(symmetry["raw_orientation_gap"], 5)],
}), use_container_width=True, hide_index=True)
if missing_features:
    st.warning("Missing inference features: " + ", ".join(missing_features))
with st.expander("Model Feature Audit"):
    st.dataframe(feature_audit, use_container_width=True, hide_index=True)


st.subheader("Prediction Context")
st.dataframe(pd.DataFrame({
    "Player": [player_1_name, player_2_name],
    "State As Of": [snapshot_value(player_1, "inference_state_as_of_date"), snapshot_value(player_2, "inference_state_as_of_date")],
    "Last Recorded Match": [snapshot_value(player_1, "last_match_date"), snapshot_value(player_2, "last_match_date")],
    "Days Since Last Recorded Match": [snapshot_value(player_1, "days_since_last_match"), snapshot_value(player_2, "days_since_last_match")],
    "Selected Surface": [surface, surface],
    "Surface Matches": [snapshot_value(player_1, "surface_matches_before"), snapshot_value(player_2, "surface_matches_before")],
    "Tournament Level": [TOURNAMENT_LEVEL_LABELS.get(tournament_level, tournament_level)] * 2,
    "Round": [ROUND_LABELS.get(round_name, round_name)] * 2,
    "Best Of": [best_of, best_of],
}), use_container_width=True, hide_index=True)


with st.expander("ℹ️ Prediction methodology"):
    st.markdown(
        """
        - Global and selected-surface features come from `player_current_state.sackmann_v5.parquet`.
        - Context-history features use the selected tournament level, round and best-of format.
        - Days since last match is calculated from the last recorded match to the prediction date.
        - Player sides use ascending player ID, exactly as in the validated training dataset; the probability is then mapped back to the selected screen order.
        - H2H only uses completed historical matches.
        - Data confidence measures coverage, not certainty of the result.
        - XGBoost and logistic champions use their complete saved preprocessing Pipeline. CatBoost receives numeric missing values directly. Categorical missing values use `__MISSING__`.
        - The page loads `production/active_model.json` and verifies registered SHA-256 hashes before inference.
        - Match-winner inference and optional market inference are reconciled through `match_prediction_service.py` so Match Prediction, Upcoming Matches and IA Candidates use the same contract.
        - First-set and at-least-one-set probabilities come from calibrated supervised direct-market models.
        - BO3 Over probabilities use cumulative direct models with monotonic correction. BO5 totals use the structural serve-point simulation.
        - Direct market probabilities and structural simulation are separate methodologies and are displayed separately.
        """
    )