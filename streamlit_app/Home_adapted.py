#!/usr/bin/env python3
"""Home y navegación explícita de Smart Tennis Analytics 2.0.

Soluciona dos problemas de la Home anterior:
1. Localiza la raíz real del repositorio antes de construir las rutas a data/ y modeling/.
2. Usa st.navigation para mostrar únicamente las páginas adaptadas seleccionadas.

Ejecutar desde la raíz del repositorio:
    streamlit run streamlit_app/Home_adapted.py

Ajusta PAGE_CANDIDATES si los nombres reales de las páginas son distintos.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st


# ==========================================================
# PROJECT AND APPLICATION PATHS
# ==========================================================

def find_project_root(start: Path) -> Path:
    """Localiza la raíz que contiene data/, modeling/ y streamlit_app/."""
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (
            (candidate / "data" / "processed").exists()
            and (candidate / "modeling").exists()
            and (candidate / "streamlit_app").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "No se pudo localizar la raíz del repositorio. Se esperaba encontrar "
        "data/processed, modeling y streamlit_app."
    )


THIS_FILE = Path(__file__).resolve()
APP_ROOT = THIS_FILE.parent
PROJECT_ROOT = find_project_root(APP_ROOT)
PAGES_DIR = APP_ROOT / "pages"

for import_path in (PROJECT_ROOT, APP_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


from streamlit_app.config.data_paths import (
    ACTIVE_MODEL_PATH,
    ARTIFACTS_DIR,
    MATCH_HISTORY_PATH,
    PROCESSED_DATA_DIR,
    PRODUCTION_ARTIFACTS_DIR,
)


# ==========================================================
# DATA AND MODEL PATHS
# ==========================================================

# Backward-compatible local names retained by the existing Home logic.
SOURCE_PARQUET = MATCH_HISTORY_PATH
MODEL_ARTIFACTS_ROOT = ARTIFACTS_DIR
PRODUCTION_ROOT = PRODUCTION_ARTIFACTS_DIR
UPCOMING_MARKET_ODDS_PATH = (
    PROCESSED_DATA_DIR / "upcoming_market_odds.oddsharvester.parquet"
)


def resolve_model_directory(
    mode: str,
) -> tuple[Path, str]:
    """Devuelve exclusivamente el modelo promocionado."""

    production = (
        PRODUCTION_ROOT
        / mode
    )

    if model_is_available(
        production
    ):
        return (
            production,
            "Production",
        )

    return (
        production,
        "Unavailable",
    )


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Smart Tennis Analytics",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# GENERAL HELPERS
# ==========================================================

def safe_integer(value: Any, default: str = "-") -> str:
    if value is None or pd.isna(value):
        return default
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return default


def safe_number(value: Any, decimals: int = 3, default: str = "-") -> str:
    if value is None or pd.isna(value):
        return default
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return default


def safe_percentage(value: Any, decimals: int = 1, default: str = "-") -> str:
    if value is None or pd.isna(value):
        return default
    try:
        return f"{float(value):.{decimals}%}"
    except (TypeError, ValueError):
        return default


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

def load_active_model() -> dict[str, Any]:
    """Carga el puntero de producción activo."""

    active = load_json_file(
        ACTIVE_MODEL_PATH
    )

    if not active:
        return {
            "available": False,
            "champion": None,
            "mode": None,
            "probability_policy": None,
            "active_model_version": None,
        }

    required = {
        "active_model_version",
        "mode",
        "champion",
        "probability_policy",
        "artifact_directory",
        "model_file",
        "metadata_file",
    }

    missing = sorted(
        required
        - set(active)
    )

    if missing:
        return {
            "available": False,
            "error": (
                "active_model.json incompleto: "
                + ", ".join(missing)
            ),
        }

    artifact_directory = (
        PRODUCTION_ROOT
        / active[
            "artifact_directory"
        ]
    )

    model_path = (
        artifact_directory
        / active["model_file"]
    )

    metadata_path = (
        artifact_directory
        / active["metadata_file"]
    )

    active["available"] = bool(
        model_path.exists()
        and metadata_path.exists()
    )
    optional_artifacts = {
        "structural_market_model": active.get("market_model_file"),
        "structural_market_metadata": active.get("market_metadata_file"),
        "direct_market_model": active.get("direct_market_model_file"),
        "direct_market_metadata": active.get("direct_market_metadata_file"),
    }
    active["optional_artifacts"] = {
        key: {
            "file": value,
            "available": bool(value and (artifact_directory / str(value)).exists()),
        }
        for key, value in optional_artifacts.items()
    }
    active["direct_markets_available"] = all(
        active["optional_artifacts"][key]["available"]
        for key in ("direct_market_model", "direct_market_metadata")
    )
    active["structural_markets_available"] = all(
        active["optional_artifacts"][key]["available"]
        for key in ("structural_market_model", "structural_market_metadata")
    )

    active[
        "artifact_directory_path"
    ] = str(
        artifact_directory
    )

    return active


def model_is_available(
    model_directory: Path,
) -> bool:
    """Valida una carpeta de producción del pipeline v6."""

    metadata_path = (
        model_directory
        / "model_metadata.json"
    )

    report_path = (
        model_directory
        / "training_report.json"
    )

    manifest_path = (
        model_directory
        / "registry_manifest.json"
    )

    if not all(
        path.exists()
        for path in (
            metadata_path,
            report_path,
            manifest_path,
        )
    ):
        return False

    metadata = load_json_file(
        metadata_path
    )

    champion = str(
        metadata.get(
            "champion",
            "",
        )
    ).strip().lower()

    model_files = {
        "xgboost": (
            "xgboost_pipeline.joblib"
        ),
        "logistic": (
            "logistic_pipeline.joblib"
        ),
        "catboost": (
            "catboost_model.cbm"
        ),
        "catboost_surface_blend": (
            "catboost_model.cbm"
        ),
    }

    model_file = model_files.get(
        champion
    )

    if model_file is None:
        return False

    return (
        model_directory
        / model_file
    ).exists()



def model_status_label(available: bool, stage: str) -> str:
    if available and stage == "Production":
        return "✅ Production"
    if available and stage == "Candidate":
        return "🧪 Candidate"
    return "⚠️ Not available"


def get_metric(
    metrics: dict[str, Any],
    evaluation: str,
    metric: str,
    default: Any = None,
) -> Any:
    return metrics.get("model", {}).get(evaluation, {}).get(metric, default)

def get_production_test_metrics(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Obtiene las métricas de la probabilidad promovida."""

    champion = str(
        report.get(
            "champion_selected_on_tune",
            "",
        )
    )

    policy = str(
        report.get(
            "probability_policy_selected_on_calibration",
            {},
        ).get(
            "selected",
            "raw",
        )
    )

    test_metrics = report.get(
        "test_metrics",
        {},
    )

    if policy == "calibrated":
        selected_name = (
            f"{champion}_selected"
        )

        return test_metrics.get(
            selected_name,
            {},
        )

    return test_metrics.get(
        champion,
        {},
    )


# ==========================================================
# PLATFORM SUMMARY
# ==========================================================

@st.cache_data(show_spinner="Loading platform summary...")
def load_platform_summary(parquet_path_as_text: str) -> dict[str, Any]:
    parquet_path = Path(parquet_path_as_text)
    empty = {
        "source_available": False,
        "error": f"Dataset not found: {parquet_path}",
        "matches": 0,
        "players": 0,
        "tournament_editions": 0,
        "tournament_names": 0,
        "seasons": 0,
        "surfaces": 0,
        "matches_with_odds": 0,
        "odds_coverage": None,
        "first_match_date": None,
        "last_match_date": None,
        "atp_matches": 0,
        "qualifying_matches": 0,
        "challenger_matches": 0,
        "futures_matches": 0,
        "grand_slam_matches": 0,
        "tml_matches": 0,
        "eligible_matches": 0,
    }

    if not parquet_path.exists():
        return empty

    try:
        schema = set(pq.ParquetFile(parquet_path).schema_arrow.names)
        requested = [
            "tourney_id",
            "tourney_name",
            "tourney_level",
            "competition_type",
            "match_date",
            "tourney_date",
            "surface",
            "player_1_id",
            "player_2_id",
            "player_1_name",
            "player_2_name",
            "source_origin",
            "eligible_for_model",
            "odds_matched",
            "player_1_avg_odds",
            "player_2_avg_odds",
            "player_1_ps_odds",
            "player_2_ps_odds",
            "player_1_b365_odds",
            "player_2_b365_odds",
            "player_1_bfe_odds",
            "player_2_bfe_odds",
        ]
        selected = [column for column in requested if column in schema]
        frame = pd.read_parquet(
            parquet_path,
            columns=selected,
            engine="pyarrow",
        )
    except Exception as error:
        empty["error"] = f"Could not read dataset: {error}"
        return empty

    matches = len(frame)

    if "match_date" in frame.columns:
        match_date = pd.to_datetime(frame["match_date"], errors="coerce")
    elif "tourney_date" in frame.columns:
        match_date = pd.to_datetime(
            pd.to_numeric(frame["tourney_date"], errors="coerce")
            .astype("Int64")
            .astype(str),
            format="%Y%m%d",
            errors="coerce",
        )
    else:
        match_date = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")

    ids = [
        frame[column]
        for column in ("player_1_id", "player_2_id")
        if column in frame.columns
    ]
    if ids:
        players = int(pd.concat(ids, ignore_index=True).dropna().nunique())
    else:
        names = [
            frame[column]
            for column in ("player_1_name", "player_2_name")
            if column in frame.columns
        ]
        players = int(
            pd.concat(names, ignore_index=True).dropna().astype(str).nunique()
        ) if names else 0

    if "competition_type" in frame.columns:
        competition = (
            frame["competition_type"]
            .astype("string")
            .str.strip()
            .str.upper()
        )
    else:
        competition = pd.Series("UNKNOWN", index=frame.index, dtype="string")

    if "tourney_level" in frame.columns:
        level = frame["tourney_level"].astype("string").str.upper()
    else:
        level = pd.Series(pd.NA, index=frame.index, dtype="string")

    if "odds_matched" in frame.columns:
        valid_odds = frame["odds_matched"].fillna(False).astype(bool)
    else:
        valid_odds = pd.Series(False, index=frame.index, dtype=bool)
        for left, right in (
            ("player_1_avg_odds", "player_2_avg_odds"),
            ("player_1_ps_odds", "player_2_ps_odds"),
            ("player_1_b365_odds", "player_2_b365_odds"),
            ("player_1_bfe_odds", "player_2_bfe_odds"),
        ):
            if left in frame.columns and right in frame.columns:
                left_odds = pd.to_numeric(frame[left], errors="coerce")
                right_odds = pd.to_numeric(frame[right], errors="coerce")
                valid_odds |= left_odds.gt(1.0) & right_odds.gt(1.0)

    matches_with_odds = int(valid_odds.sum())
    source = (
        frame["source_origin"].astype("string").str.upper()
        if "source_origin" in frame.columns
        else pd.Series("UNKNOWN", index=frame.index, dtype="string")
    )
    eligible = (
        frame["eligible_for_model"].fillna(False).astype(bool)
        if "eligible_for_model" in frame.columns
        else pd.Series(True, index=frame.index, dtype=bool)
    )

    # competition_type is part of the tournament edition key.
    tournament_editions = int(
        frame[["tourney_id"]]
        .assign(competition_type=competition)
        .dropna(subset=["tourney_id"])
        .drop_duplicates()
        .shape[0]
    ) if "tourney_id" in frame.columns else 0

    return {
        "source_available": True,
        "error": None,
        "matches": int(matches),
        "players": players,
        "tournament_editions": tournament_editions,
        "tournament_names": int(
            frame["tourney_name"].dropna().astype(str).nunique()
        ) if "tourney_name" in frame.columns else 0,
        "seasons": int(match_date.dt.year.dropna().nunique()),
        "surfaces": int(
            frame["surface"].dropna().astype(str).nunique()
        ) if "surface" in frame.columns else 0,
        "matches_with_odds": matches_with_odds,
        "odds_coverage": matches_with_odds / matches if matches else None,
        "first_match_date": match_date.min(),
        "last_match_date": match_date.max(),
        "atp_matches": int(competition.eq("ATP").sum()),
        "qualifying_matches": int(competition.eq("ATP_QUALIFYING").sum()),
        "challenger_matches": int(competition.eq("CHALLENGER").sum()),
        "futures_matches": int(competition.eq("FUTURES").sum()),
        "grand_slam_matches": int(level.eq("G").sum()),
        "tml_matches": int(source.eq("TML").sum()),
        "eligible_matches": int(eligible.sum()),
    }


# ==========================================================
# HOME CONTENT
# ==========================================================

def render_model_card(
    title: str,
    subtitle: str,
    model_dir: Path,
    stage: str,
) -> None:
    available = model_is_available(
        model_dir
    )

    metadata = load_json_file(
        model_dir
        / "model_metadata.json"
    )

    report = load_json_file(
        model_dir
        / "training_report.json"
    )

    active = load_active_model()

    metrics = get_production_test_metrics(
        report
    )

    champion = str(
        metadata.get(
            "champion",
            active.get(
                "champion",
                "-",
            ),
        )
    )

    probability_policy = str(
        metadata.get(
            "probability_policy",
            active.get(
                "probability_policy",
                "raw",
            ),
        )
    )

    features = metadata.get(
        "features",
        metadata.get(
            "feature_columns",
            [],
        ),
    )

    st.subheader(
        title
    )

    st.caption(
        subtitle
    )

    st.metric(
        "Status",
        model_status_label(
            available,
            stage,
        ),
    )

    left, right = st.columns(
        2
    )

    with left:
        st.metric(
            "Test Log Loss",
            safe_number(
                metrics.get(
                    "log_loss"
                ),
                4,
            ),
        )

        st.metric(
            "Accuracy",
            safe_percentage(
                metrics.get(
                    "accuracy"
                )
            ),
        )

    with right:
        st.metric(
            "ROC AUC",
            safe_number(
                metrics.get(
                    "roc_auc"
                ),
                4,
            ),
        )

        st.metric(
            "Calibration Error",
            safe_percentage(
                metrics.get(
                    "ece_15_bins"
                )
            ),
        )

    st.write(
        "Champion:",
        champion.upper()
        if champion
        else "-",
    )

    st.write(
        "Features:",
        safe_integer(
            len(features)
        ),
    )

    st.write(
        "Probability policy:",
        (
            probability_policy.title()
            if available
            else "-"
        ),
    )

    st.write(
        "Model version:",
        active.get(
            "active_model_version",
            "-",
        ),
    )

    st.write(
        "Artifact stage:",
        stage,
    )
    if stage == "Production":
        st.write(
            "Direct market models:",
            "Available" if active.get("direct_markets_available") else "Not available",
        )
        st.write(
            "Structural simulator:",
            "Available" if active.get("structural_markets_available") else "Not available",
        )


def render_home() -> None:
    summary = load_platform_summary(str(SOURCE_PARQUET))

    no_market_dir, no_market_stage = resolve_model_directory("no_market")
    with_market_dir, with_market_stage = resolve_model_directory("with_market")
    no_market_available = model_is_available(no_market_dir)
    with_market_available = model_is_available(with_market_dir)
    active = load_active_model()
    direct_markets_available = bool(active.get("direct_markets_available"))
    structural_markets_available = bool(active.get("structural_markets_available"))
    available_models = (
        int(no_market_available)
        + int(with_market_available)
        + int(direct_markets_available)
        + int(structural_markets_available)
    )

    st.title("🎾 Smart Tennis Analytics 2.0")
    st.markdown(
        "**ATP, qualifying and Challenger analytics, historical market "
        "analysis and professional match prediction.**"
    )

    if not summary["source_available"]:
        st.error(summary["error"])
        st.code(str(SOURCE_PARQUET))
        st.stop()

    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
    kpi_1.metric("Players", safe_integer(summary["players"]))
    kpi_2.metric("Matches", safe_integer(summary["matches"]))
    kpi_3.metric("Tournament Editions", safe_integer(summary["tournament_editions"]))
    kpi_4.metric("Available Models", safe_integer(available_models))

    st.divider()
    st.subheader("📚 Data Coverage")

    coverage_1, coverage_2, coverage_3, coverage_4 = st.columns(4)
    first_date = summary["first_match_date"]
    last_date = summary["last_match_date"]
    coverage_1.metric(
        "First Match",
        str(first_date.date()) if pd.notna(first_date) else "-",
    )
    coverage_2.metric(
        "Last Match",
        str(last_date.date()) if pd.notna(last_date) else "-",
    )
    coverage_3.metric("Seasons", safe_integer(summary["seasons"]))
    coverage_4.metric("Surfaces", safe_integer(summary["surfaces"]))

    coverage_5, coverage_6, coverage_7, coverage_8 = st.columns(4)
    coverage_5.metric("ATP Main Draw", safe_integer(summary["atp_matches"]))
    coverage_6.metric("ATP Qualifying", safe_integer(summary["qualifying_matches"]))
    coverage_7.metric("Challenger", safe_integer(summary["challenger_matches"]))
    coverage_8.metric("TML Matches", safe_integer(summary["tml_matches"]))

    coverage_9, coverage_10, coverage_11, coverage_12 = st.columns(4)
    coverage_9.metric("Grand Slam", safe_integer(summary["grand_slam_matches"]))
    coverage_10.metric("Futures History", safe_integer(summary["futures_matches"]))
    coverage_11.metric("Eligible Matches", safe_integer(summary["eligible_matches"]))
    coverage_12.metric("Odds Coverage", safe_percentage(summary["odds_coverage"]))

    st.divider()
    st.header("🤖 Model Artifacts")
    model_1, model_2 = st.columns(2)
    with model_1:
        with st.container(border=True):
            render_model_card(
                "🎾 Sports Model",
                (
                    "Active production champion "
                    "without betting-market features."
                ),
                no_market_dir,
                no_market_stage,
            )

    with model_2:
        with st.container(border=True):
            render_model_card(
                "💹 Sports + Market Model",
                (
                    "Optional production champion "
                    "with betting-market features."
                ),
                with_market_dir,
                with_market_stage,
            )

    st.divider()
    st.header("📊 Analytics and Prediction")
    card_1, card_2, card_3 = st.columns(3)

    with card_1:
        with st.container(border=True):
            st.subheader("👤 Player Analysis")
            st.write("Career, surface, Elo, recent form and match history.")
            st.page_link(
                "pages/01_Player_Analysis_adapted.py",
                label="Open Player Analysis",
                icon="👤",
                use_container_width=True,
            )

    with card_2:
        with st.container(border=True):
            st.subheader("🏆 Tournament Analysis")
            st.write("Tournament editions, field strength and historical results.")
            st.page_link(
                "pages/03_Tournament_Analysis_adapted.py",
                label="Open Tournament Analysis",
                icon="🏆",
                use_container_width=True,
            )

    with card_3:
        with st.container(border=True):
            st.subheader("🎯 Match Prediction")
            st.write(
                "Production match-winner probability, calibrated first-set and "
                "at-least-one-set markets, BO3 totals and structural simulation."
            )
            if no_market_available:
                st.page_link(
                    "pages/05_Match_Prediction_adapted.py",
                    label="Open Match Prediction",
                    icon="🎯",
                    use_container_width=True,
                )
            else:
                st.warning("No sports-model artifact is available.")

    upcoming_card, ai_card_1, ai_card_2 = st.columns(3)
    with upcoming_card:
        with st.container(border=True):
            st.subheader("📅 Upcoming Matches")
            st.write(
                "Scheduled fixtures with match winner, first set, at least one set, "
                "calibrated BO3 totals and structural diagnostics."
            )
            if (PAGES_DIR / "06_Upcoming_Matches.py").exists():
                st.page_link(
                    "pages/06_Upcoming_Matches.py",
                    label="Open Upcoming Matches",
                    icon="📅",
                    use_container_width=True,
                )
            else:
                st.warning("Upcoming Matches page is not installed.")
    ai_card_1, ai_card_2, ai_card_3 = st.columns(3)
    with ai_card_1:
        with st.container(border=True):
            st.subheader("🤖 IA Candidates")
            st.write(
                "Evidence-first dossiers for upcoming matches: surface changes, "
                "inactivity, H2H, score trends and tournament history."
            )
            if (PAGES_DIR / "07_IA_Candidates.py").exists():
                st.page_link(
                    "pages/07_IA_Candidates.py",
                    label="Open IA Candidates",
                    icon="🤖",
                    use_container_width=True,
                )
            else:
                st.warning("IA Candidates page is not installed.")
    with ai_card_2:
        st.empty()
    with ai_card_3:
        st.empty()
    betting_1, betting_2, betting_3 = st.columns(3)

    with betting_1:
        with st.container(border=True):
            st.subheader("💰 Player Betting")
            st.write(
                "Historical odds, ROI, favourites, "
                "underdogs and drawdown."
            )
            st.page_link(
                "pages/02_Player_Betting_adapted.py",
                label="Open Player Betting",
                icon="💰",
                use_container_width=True,
            )

    with betting_2:
        with st.container(border=True):
            st.subheader("📈 Tournament Betting")
            st.write(
                "Historical market performance by "
                "tournament and surface."
            )
            st.page_link(
                "pages/04_Tournament_Betting_adapted.py",
                label="Open Tournament Betting",
                icon="📈",
                use_container_width=True,
            )

    with betting_3:
        with st.container(border=True):
            st.subheader("📊 Upcoming Odds Analysis")
            st.write(
                "Run OddsHarvester and inspect upcoming "
                "match-winner prices and market margins."
            )
            st.page_link(
                "pages/07_upcoming_odds_analysis.py",
                label="Open Upcoming Odds Analysis",
                icon="📊",
                use_container_width=True,
            )

    st.divider()
    st.header("⚙️ Platform Status")
    status_1, status_2, status_3 = st.columns(3)
    status_1.metric("Dataset", "✅ Online")
    status_2.metric("Data Max Date", str(last_date.date()) if pd.notna(last_date) else "-")
    status_3.metric("Match Winner", model_status_label(no_market_available, no_market_stage))
    status_4, status_5, status_6 = st.columns(3)
    status_4.metric(
        "Direct Markets",
        "✅ Production" if direct_markets_available else "⚠️ Not available",
    )
    status_5.metric(
        "Structural Simulator",
        "✅ Production" if structural_markets_available else "⚠️ Not available",
    )
    status_6.metric("Optional With-Market", model_status_label(with_market_available, with_market_stage))

    with st.expander("🔎 Paths and technical details"):
        details = pd.DataFrame(
            [
                {
                    "Component": "Project root",
                    "Path": str(PROJECT_ROOT),
                    "Available": PROJECT_ROOT.exists(),
                },
                {
                    "Component": "Feature dataset",
                    "Path": str(SOURCE_PARQUET),
                    "Available": SOURCE_PARQUET.exists(),
                },
                {
                    "Component": f"Sports model ({no_market_stage})",
                    "Path": str(no_market_dir),
                    "Available": no_market_available,
                },
                {
                    "Component": f"Market model ({with_market_stage})",
                    "Path": str(with_market_dir),
                    "Available": with_market_available,
                },
                {
                    "Component": "Direct market model",
                    "Path": str(
                        no_market_dir / str(active.get("direct_market_model_file", ""))
                    ),
                    "Available": direct_markets_available,
                },
                {
                    "Component": "Structural market simulator",
                    "Path": str(
                        no_market_dir / str(active.get("market_model_file", ""))
                    ),
                    "Available": structural_markets_available,
                },
                {
                    "Component": "Upcoming OddsHarvester cache",
                    "Path": str(
                        UPCOMING_MARKET_ODDS_PATH
                    ),
                    "Available": (
                        UPCOMING_MARKET_ODDS_PATH
                    ).exists(),
                },
                
            ]
        )
        st.dataframe(details, use_container_width=True, hide_index=True)

    st.caption(
        "Smart Tennis Analytics 2.0 | Sackmann + TennisMyLife v2 | "
        "ATP, qualifying and Challenger analytics"
    )


# ==========================================================
# EXPLICIT NAVIGATION
# ==========================================================

PAGE_CANDIDATES = [
    {
        "path": PAGES_DIR / "01_Player_Analysis_adapted.py",
        "title": "Player Analysis",
        "icon": "👤",
        "section": "Analytics",
    },
    {
        "path": PAGES_DIR / "03_Tournament_Analysis_adapted.py",
        "title": "Tournament Analysis",
        "icon": "🏆",
        "section": "Analytics",
    },
    {
        "path": PAGES_DIR / "02_Player_Betting_adapted.py",
        "title": "Player Betting",
        "icon": "💰",
        "section": "Market Analysis",
    },
    {
        "path": PAGES_DIR / "04_Tournament_Betting_adapted.py",
        "title": "Tournament Betting",
        "icon": "📈",
        "section": "Market Analysis",
    },
    {
        "path": PAGES_DIR / "07_upcoming_odds_analysis.py",
        "title": "Upcoming Odds Analysis",
        "icon": "📊",
        "section": "Market Analysis",
    },
    {
        "path": PAGES_DIR / "05_Match_Prediction_adapted.py",
        "title": "Match Prediction",
        "icon": "🎯",
        "section": "Prediction",
    },
    {
        "path": PAGES_DIR / "06_Upcoming_Matches.py",
        "title": "Upcoming Matches",
        "icon": "📅",
        "section": "Prediction",
    },
    {
        "path": PAGES_DIR / "07_IA_Candidates.py",
        "title": "IA Candidates",
        "icon": "🤖",
        "section": "Prediction",
    },
]

navigation: dict[str, list[st.Page]] = {
    "Home": [
        st.Page(
            render_home,
            title="Home",
            icon="🎾",
            default=True,
        )
    ]
}

for definition in PAGE_CANDIDATES:
    if definition["path"].exists():
        navigation.setdefault(
            definition["section"],
            [],
        ).append(
            st.Page(
                definition["path"],
                title=definition["title"],
                icon=definition["icon"],
            )
        )

selected_page = st.navigation(
    navigation,
    position="sidebar",
    expanded=True,
)

selected_page.run()
