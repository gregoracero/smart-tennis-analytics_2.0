#!/usr/bin/env python3
"""Entrena, valida, registra y promociona el champion del pipeline de tenis.

Este es el punto de entrada de produccion y del estudio de ablacion. Ejecuta el entrenador profesional v5
como motor de benchmark y, cuando se indica --promote, valida y promociona de
forma atomica el champion elegido en tune y su politica raw/calibrated elegida
en calibration.

El script NO vuelve a elegir el modelo usando test. Test se conserva como
informe final y como alarma, nunca como criterio de seleccion o promocion.

Flujo
-----
1. Ejecuta train_professional_tennis_model_professional_v5.py como motor de benchmark temporal.
2. Lee training_report.json y model_metadata.json.
3. Comprueba que el champion fue elegido en tune y la politica en calibration.
4. Valida artefacto, features, probabilidades de calibration y hashes.
5. Aplica gates basados exclusivamente en tune/calibration.
6. Copia el candidate a un registro inmutable versionado.
7. Promociona atomicamente a modeling/artifacts/production/<mode>.
8. Actualiza modeling/artifacts/production/active_model.json.

Streamlit debe leer active_model.json y no rutas experimentales directas.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

MODEL_FILE_BY_CHAMPION = {
    "xgboost": "xgboost_pipeline.joblib",
    "logistic": "logistic_pipeline.joblib",
    "catboost": "catboost_model.cbm",
    "catboost_surface_blend": "catboost_model.cbm",
}
REQUIRED_CANDIDATE_FILES = {
    "training_report.json",
    "model_metadata.json",
    "metrics.csv",
    "feature_audit.csv",
    "calibration_predictions.parquet",
    "test_predictions.parquet",
    "test_segment_metrics.csv",
}
TARGET = "target_player_1_win"
MARKET_MODEL_FILE = "tennis_market_models.joblib"
MARKET_METADATA_FILE = "tennis_market_metadata.json"
MARKET_VALIDATION_FILE = "tennis_market_validation.csv"
MARKET_REPORT_DIRECTORY = "market_validation_reports"
MARKET_SUMMARY_FILE = "market_validation_summary.json"
MARKET_SEGMENT_METRICS_FILE = "market_segment_metrics.csv"
MARKET_SEGMENT_CALIBRATION_FILE = "market_segment_calibration.csv"
MARKET_RELIABILITY_FILE = "market_reliability_summary.csv"
DIRECT_MARKET_MODEL_FILE = "tennis_market_direct_models.joblib"
DIRECT_MARKET_METADATA_FILE = "tennis_market_direct_metadata.json"
DIRECT_MARKET_VALIDATION_FILE = "tennis_market_direct_validation.parquet"
DIRECT_MARKET_REQUIRED_FILES = {
    DIRECT_MARKET_MODEL_FILE,
    DIRECT_MARKET_METADATA_FILE,
    DIRECT_MARKET_VALIDATION_FILE,
    "market_model_metrics.csv",
    "market_over_monotonicity.csv",
    "market_calibration_bands.csv",
    "market_segment_metrics.csv",
    "market_symmetry_validation.csv",
}
MARKET_PRIMARY_OVER_LINE = 21.5
MARKET_BO3_OVER_LINES = (18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5)
MARKET_BO5_OVER_LINES = (36.5, 38.5, 40.5)
MARKET_REQUIRED_STAT_COLUMNS = (
    "w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon", "score",
)
MARKET_DEFAULT_OVER_LINES = (18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 36.5, 38.5, 40.5)
DEFAULT_MIN_CALIBRATION_GAIN_VS_ELO = 0.001
DEFAULT_MAX_CALIBRATION_LOGLOSS = 0.69
DEFAULT_MAX_SEGMENT_LOGLOSS = 0.75
DEFAULT_MAX_SEGMENT_ECE = 0.20
ABLATION_PROFILES = (
    "optimized_v4", "momentum", "workload", "dynamic_stats",
    "ranking_trends", "matchup", "momentum_v6", "v5_all", "v6_all",
)
PROFILE_DATASET_NAMES = {
    "baseline": "tennis_ml_no_market.baseline.sackmann_v5.parquet",
    "optimized_v4": "tennis_ml_no_market.optimized_v4.sackmann_v5.parquet",
    "momentum": "tennis_ml_no_market.momentum.sackmann_v5.parquet",
    "workload": "tennis_ml_no_market.workload.sackmann_v5.parquet",
    "dynamic_stats": "tennis_ml_no_market.dynamic_stats.sackmann_v5.parquet",
    "ranking_trends": "tennis_ml_no_market.ranking_trends.sackmann_v5.parquet",
    "matchup": "tennis_ml_no_market.matchup.sackmann_v5.parquet",
    "momentum_v6": "tennis_ml_no_market.momentum_v6.sackmann_v5.parquet",
    "v5_all": "tennis_ml_no_market.sackmann_v5.parquet",
    "v6_all": "tennis_ml_no_market.v6_all.sackmann_v5.parquet",
}
PROFILE_METADATA_NAMES = {
    "baseline": "tennis_ml_dataset_metadata.baseline.sackmann_v5.json",
    "optimized_v4": "tennis_ml_dataset_metadata.optimized_v4.sackmann_v5.json",
    "momentum": "tennis_ml_dataset_metadata.momentum.sackmann_v5.json",
    "workload": "tennis_ml_dataset_metadata.workload.sackmann_v5.json",
    "dynamic_stats": "tennis_ml_dataset_metadata.dynamic_stats.sackmann_v5.json",
    "ranking_trends": "tennis_ml_dataset_metadata.ranking_trends.sackmann_v5.json",
    "matchup": "tennis_ml_dataset_metadata.matchup.sackmann_v5.json",
    "momentum_v6": "tennis_ml_dataset_metadata.momentum_v6.sackmann_v5.json",
    "v5_all": "tennis_ml_dataset_metadata_sackmann_v5.json",
    "v6_all": "tennis_ml_dataset_metadata.v6_all.sackmann_v5.json",
}


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    reasons: list[str]
    warnings: list[str]
    champion: str
    policy: str
    candidate_calibration_logloss: float
    elo_surface_calibration_logloss: float | None


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (
            (candidate / "data" / "processed").exists()
            and (candidate / "modeling").exists()
        ):
            return candidate
    raise FileNotFoundError("No se pudo localizar la raiz del proyecto")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def directory_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)).replace("\\", "/"): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "registry_manifest.json"
    }


def finite_probability(values: pd.Series, label: str) -> np.ndarray:
    probability = pd.to_numeric(values, errors="coerce").to_numpy(float)
    if not np.isfinite(probability).all():
        raise RuntimeError(f"{label}: probabilidades nulas o no finitas")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise RuntimeError(f"{label}: probabilidades fuera de [0,1]")
    return probability


def selected_metric_from_report(
    report: dict[str, Any],
    period: str,
) -> tuple[str, dict[str, Any]]:
    champion = str(report["champion_selected_on_tune"])
    policy = str(report["probability_policy_selected_on_calibration"]["selected"])
    metrics_key = "calibration_metrics" if period == "calibration" else "test_metrics"
    metrics = report[metrics_key]
    if policy == "raw":
        return champion, metrics[champion]
    policy_block = report["probability_policy_selected_on_calibration"]
    if period == "calibration":
        return f"{champion}_selected", policy_block["calibrated"]
    selected_name = f"{champion}_selected"
    return selected_name, metrics[selected_name]


def verify_candidate_files(candidate: Path, champion: str) -> str:
    missing = sorted(
        name for name in REQUIRED_CANDIDATE_FILES
        if not (candidate / name).exists()
    )
    if missing:
        raise RuntimeError(f"Candidate incompleto. Faltan: {missing}")
    if champion not in MODEL_FILE_BY_CHAMPION:
        raise RuntimeError(f"Champion no soportado para produccion: {champion}")
    model_file = MODEL_FILE_BY_CHAMPION[champion]
    if not (candidate / model_file).exists():
        raise RuntimeError(f"No existe artefacto del champion: {model_file}")
    return model_file


def verify_model_reload(
    candidate: Path,
    champion: str,
    model_file: str,
    expected_features: list[str],
) -> None:
    model_path = candidate / model_file
    if champion in {"xgboost", "logistic"}:
        pipeline = joblib.load(model_path)
        if not hasattr(pipeline, "predict_proba"):
            raise RuntimeError("El pipeline recargado no implementa predict_proba")
        preprocess = pipeline.named_steps.get("preprocess")
        if preprocess is None:
            raise RuntimeError("El pipeline no contiene el paso preprocess")
        configured: list[str] = []
        for _, _, columns in preprocess.transformers:
            if isinstance(columns, (list, tuple, np.ndarray, pd.Index)):
                configured.extend(str(column) for column in columns)
        if configured and (
            len(configured) != len(expected_features)
            or set(configured) != set(expected_features)
        ):
            missing = sorted(set(expected_features) - set(configured))
            unexpected = sorted(set(configured) - set(expected_features))
            raise RuntimeError(
                "Las features del pipeline no coinciden con metadata. "
                f"Faltan={missing}; inesperadas={unexpected}"
            )
    elif champion.startswith("catboost"):
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        names = list(model.feature_names_)
        if names != expected_features:
            raise RuntimeError(
                "Las features CatBoost no coinciden exactamente con metadata"
            )


def verify_prediction_artifacts(
    candidate: Path,
    champion: str,
    policy: str,
) -> None:
    calibration = pd.read_parquet(candidate / "calibration_predictions.parquet")
    test = pd.read_parquet(candidate / "test_predictions.parquet")
    selected_column = "probability_selected"
    raw_column = f"probability_{champion}"
    for label, frame in (("calibration", calibration), ("test", test)):
        if TARGET not in frame:
            raise RuntimeError(f"{label}: falta target de auditoria")
        if raw_column not in frame:
            raise RuntimeError(f"{label}: falta {raw_column}")
        if selected_column not in frame:
            raise RuntimeError(f"{label}: falta {selected_column}")
        raw = finite_probability(frame[raw_column], f"{label} raw")
        selected = finite_probability(frame[selected_column], f"{label} selected")
        if policy == "raw" and not np.allclose(raw, selected, rtol=0, atol=1e-6):
            raise RuntimeError(
                f"{label}: metadata indica raw pero probability_selected difiere"
            )


def segment_warnings(
    candidate: Path,
    maximum_logloss: float,
    maximum_ece: float,
) -> list[str]:
    frame = pd.read_csv(candidate / "test_segment_metrics.csv")
    warnings: list[str] = []
    if frame.empty:
        return ["No hay metricas segmentadas"]
    for row in frame.itertuples(index=False):
        logloss = float(row.log_loss)
        ece = float(row.ece_15_bins)
        if logloss > maximum_logloss:
            warnings.append(
                f"Segmento {row.segment}={row.value}: log_loss={logloss:.6f}"
            )
        if ece > maximum_ece:
            warnings.append(
                f"Segmento {row.segment}={row.value}: ece={ece:.6f}"
            )
    return warnings


def evaluate_promotion(
    candidate: Path,
    report: dict[str, Any],
    minimum_gain_vs_elo: float,
    maximum_calibration_logloss: float,
    maximum_segment_logloss: float,
    maximum_segment_ece: float,
) -> PromotionDecision:
    reasons: list[str] = []
    champion = str(report.get("champion_selected_on_tune", ""))
    policy = str(
        report.get("probability_policy_selected_on_calibration", {}).get(
            "selected", ""
        )
    )
    if report.get("test_used_for_selection") is not False:
        reasons.append("training_report no garantiza test_used_for_selection=false")
    if policy not in {"raw", "calibrated"}:
        reasons.append(f"Politica de probabilidad invalida: {policy}")

    _, candidate_metrics = selected_metric_from_report(report, "calibration")
    candidate_logloss = float(candidate_metrics["log_loss"])
    if not math.isfinite(candidate_logloss):
        reasons.append("Log Loss de calibration no finito")
    if candidate_logloss >= maximum_calibration_logloss:
        reasons.append(
            f"Log Loss calibration {candidate_logloss:.6f} >= "
            f"{maximum_calibration_logloss:.6f}"
        )

    calibration_metrics = report.get("calibration_metrics", {})
    elo_metrics = calibration_metrics.get("elo_surface")
    elo_logloss = float(elo_metrics["log_loss"]) if elo_metrics else None
    if elo_logloss is None:
        reasons.append("No existe baseline elo_surface en calibration")
    else:
        gain = elo_logloss - candidate_logloss
        if gain < minimum_gain_vs_elo:
            reasons.append(
                f"Ganancia vs Elo superficie insuficiente: {gain:.6f} < "
                f"{minimum_gain_vs_elo:.6f}"
            )

    warnings = segment_warnings(
        candidate, maximum_segment_logloss, maximum_segment_ece
    )
    return PromotionDecision(
        approved=not reasons,
        reasons=reasons,
        warnings=warnings,
        champion=champion,
        policy=policy,
        candidate_calibration_logloss=candidate_logloss,
        elo_surface_calibration_logloss=elo_logloss,
    )


def model_version(champion: str, mode: str, report: dict[str, Any]) -> str:
    fingerprint = str(
        report.get("dataset_fingerprint", {}).get("edge_sha256", "unknown")
    )[:10].lower()
    return (
        f"{champion}_{mode}_{time.strftime('%Y%m%d_%H%M%S')}_{fingerprint}"
    )


def copy_candidate_to_registry(
    candidate: Path,
    registry_root: Path,
    version: str,
    report: dict[str, Any],
    decision: PromotionDecision,
    model_file: str,
) -> Path:
    destination = registry_root / version
    if destination.exists():
        raise FileExistsError(f"La version ya existe: {destination}")
    temporary = registry_root / f".{version}.new"
    if temporary.exists():
        shutil.rmtree(temporary)
    registry_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(candidate, temporary)
    manifest = {
        "schema_version": 1,
        "model_version": version,
        "registered_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "champion": decision.champion,
        "mode": report["mode"],
        "probability_policy": decision.policy,
        "model_file": model_file,
        "metadata_file": "model_metadata.json",
        "training_report_file": "training_report.json",
        "candidate_calibration_logloss": decision.candidate_calibration_logloss,
        "elo_surface_calibration_logloss": decision.elo_surface_calibration_logloss,
        "promotion_warnings": decision.warnings,
        "direct_market_model_file": DIRECT_MARKET_MODEL_FILE if (temporary / DIRECT_MARKET_MODEL_FILE).exists() else None,
        "direct_market_metadata_file": DIRECT_MARKET_METADATA_FILE if (temporary / DIRECT_MARKET_METADATA_FILE).exists() else None,
        "dataset_fingerprint": report.get("dataset_fingerprint", {}),
        "dataset_schema_version": report.get("dataset_schema_version"),
        "feature_profile": report.get("feature_profile"),
        "feature_families": report.get("feature_families", []),
        "feature_builder_manifest": report.get("feature_builder_manifest"),
        "hashes_sha256": directory_hashes(temporary),
    }
    atomic_json(manifest, temporary / "registry_manifest.json")
    os.replace(temporary, destination)
    return destination


def verify_registry(registry_dir: Path) -> dict[str, Any]:
    manifest = read_json(registry_dir / "registry_manifest.json")
    expected = manifest.get("hashes_sha256", {})
    for relative, expected_hash in expected.items():
        path = registry_dir / relative
        if not path.exists():
            raise RuntimeError(f"Registro incompleto: falta {relative}")
        actual = sha256(path)
        if actual != expected_hash:
            raise RuntimeError(
                f"Hash invalido para {relative}: {actual} != {expected_hash}"
            )
    return manifest


def promote_registry_atomically(
    registry_dir: Path,
    production_root: Path,
    mode: str,
) -> tuple[Path, Path | None]:
    production_root.mkdir(parents=True, exist_ok=True)
    production_dir = production_root / mode
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup = None
    temporary = production_root / f".{mode}.new.{timestamp}"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(registry_dir, temporary)
    verify_registry(temporary)

    if production_dir.exists():
        backup = production_root / "backups" / f"{mode}_{timestamp}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        os.replace(production_dir, backup)
    os.replace(temporary, production_dir)
    return production_dir, backup


def write_active_pointer(
    production_root: Path,
    production_dir: Path,
    registry_manifest: dict[str, Any],
) -> Path:
    active_path = production_root / "active_model.json"
    relative_directory = os.path.relpath(production_dir, production_root)
    pointer = {
        "schema_version": 1,
        "active_model_version": registry_manifest["model_version"],
        "mode": registry_manifest["mode"],
        "champion": registry_manifest["champion"],
        "probability_policy": registry_manifest["probability_policy"],
        "artifact_directory": relative_directory.replace("\\", "/"),
        "model_file": registry_manifest["model_file"],
        "metadata_file": registry_manifest["metadata_file"],
        "registry_manifest_file": "registry_manifest.json",
        "dataset_schema_version": registry_manifest.get("dataset_schema_version"),
        "feature_profile": registry_manifest.get("feature_profile"),
        "feature_families": registry_manifest.get("feature_families", []),
        "feature_builder_manifest": registry_manifest.get("feature_builder_manifest"),
        "market_model_file": MARKET_MODEL_FILE if (production_dir / MARKET_MODEL_FILE).exists() else None,
        "market_metadata_file": MARKET_METADATA_FILE if (production_dir / MARKET_METADATA_FILE).exists() else None,
        "direct_market_model_file": DIRECT_MARKET_MODEL_FILE if (production_dir / DIRECT_MARKET_MODEL_FILE).exists() else None,
        "direct_market_metadata_file": DIRECT_MARKET_METADATA_FILE if (production_dir / DIRECT_MARKET_METADATA_FILE).exists() else None,
        "activated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    atomic_json(pointer, active_path)
    return active_path


def run_trainer(
    trainer: Path,
    root: Path,
    trainer_arguments: list[str],
) -> None:
    command = [sys.executable, str(trainer), *trainer_arguments]
    print("\nEjecutando entrenador candidate:\n" + subprocess.list2cmdline(command))
    subprocess.run(command, cwd=root, check=True)



def parse_profiles(value: str) -> list[str]:
    profiles = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(profiles) - set(PROFILE_DATASET_NAMES))
    if unknown:
        raise argparse.ArgumentTypeError(f"Perfiles desconocidos: {unknown}")
    if not profiles:
        raise argparse.ArgumentTypeError("Debe indicarse al menos un perfil")
    return list(dict.fromkeys(profiles))


def set_cli_value(arguments: list[str], option: str, value: str) -> list[str]:
    result = list(arguments)
    if option in result:
        index = result.index(option)
        if index + 1 >= len(result):
            raise ValueError(f"Falta valor para {option}")
        result[index + 1] = value
    else:
        result.extend([option, value])
    return result


def trainer_help(trainer: Path, root: Path) -> str:
    completed = subprocess.run(
        [sys.executable, str(trainer), "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout + completed.stderr


def supported_option(help_text: str, candidates: tuple[str, ...], label: str) -> str:
    for option in candidates:
        if option in help_text:
            return option
    raise RuntimeError(
        f"El entrenador no expone una opcion compatible para {label}. "
        f"Candidatas={candidates}"
    )


def nested_metrics(report: dict[str, Any], key: str, champion: str) -> dict[str, Any]:
    block = report.get(key, {})
    if not isinstance(block, dict):
        return {}
    value = block.get(champion, {})
    return value if isinstance(value, dict) else {}


def finite_metric(metrics: dict[str, Any], name: str) -> float:
    try:
        value = float(metrics.get(name, math.nan))
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def tune_metrics_for_selected_champion(report: dict[str, Any]) -> dict[str, Any]:
    champion = str(report.get("champion_selected_on_tune", ""))
    for key in ("tune_metrics", "validation_metrics", "model_selection_metrics"):
        metrics = nested_metrics(report, key, champion)
        if "log_loss" in metrics:
            return metrics
    selection = report.get("champion_selection", {})
    if isinstance(selection, dict):
        metrics = selection.get("selected_metrics", {})
        if isinstance(metrics, dict) and "log_loss" in metrics:
            return metrics
    raise RuntimeError(
        "training_report.json no contiene Log Loss de tune para el champion. "
        "El ablation study debe seleccionar perfiles exclusivamente con tune."
    )


def profile_artifacts(
    processed: Path,
    profile: str,
    mode: str = "no_market",
) -> tuple[Path, Path]:
    dataset_name = PROFILE_DATASET_NAMES[profile]
    if mode == "with_market":
        dataset_name = dataset_name.replace("tennis_ml_no_market", "tennis_ml_with_market")
    dataset = processed / dataset_name
    metadata = processed / PROFILE_METADATA_NAMES[profile]
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    if not metadata.exists():
        raise FileNotFoundError(metadata)
    return dataset, metadata


def verify_identical_profile_rows(
    processed: Path,
    profiles: list[str],
    mode: str,
) -> dict[str, Any]:
    keys = ["tourney_id", "competition_type", "match_num", "split", TARGET]
    reference: pd.DataFrame | None = None
    reference_profile = profiles[0]
    row_counts: dict[str, int] = {}
    for profile in profiles:
        dataset, _ = profile_artifacts(processed, profile, mode)
        frame = pd.read_parquet(dataset, columns=keys).reset_index(drop=True)
        row_counts[profile] = len(frame)
        if reference is None:
            reference = frame
        elif not reference.equals(frame):
            raise RuntimeError(
                f"El perfil {profile} no contiene exactamente las mismas filas, "
                f"splits y targets que {reference_profile}"
            )
    return {"reference_profile": reference_profile, "rows": row_counts}


def run_ablation(
    args: argparse.Namespace,
    trainer_args: list[str],
    trainer: Path,
    root: Path,
) -> None:
    processed = root / "data" / "processed"
    profiles = args.ablation_profiles
    row_audit = verify_identical_profile_rows(processed, profiles, args.ablation_mode)
    help_text = trainer_help(trainer, root)
    input_option = supported_option(
        help_text,
        ("--input", "--dataset", "--data", "--input-no-market"),
        "dataset de entrada",
    )
    metadata_option = supported_option(
        help_text,
        ("--metadata",),
        "metadata del dataset",
    )
    output_option = supported_option(help_text, ("--output-root",), "output root")
    mode_option = "--mode" if "--mode" in help_text else None

    ablation_root = Path(args.ablation_root).resolve()
    ablation_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}

    for profile in profiles:
        dataset, metadata = profile_artifacts(processed, profile, args.ablation_mode)
        profile_root = ablation_root / profile
        arguments = list(trainer_args)
        arguments = set_cli_value(arguments, input_option, str(dataset))
        arguments = set_cli_value(arguments, metadata_option, str(metadata))
        arguments = set_cli_value(arguments, output_option, str(profile_root))
        if mode_option:
            arguments = set_cli_value(arguments, mode_option, args.ablation_mode)
        print(f"\n{'=' * 78}\nABLATION PROFILE: {profile}\n{'=' * 78}")
        run_trainer(trainer, root, arguments)
        candidate = profile_root / args.ablation_mode
        report = read_json(candidate / "training_report.json")
        metadata = read_json(candidate / "model_metadata.json")
        if report.get("test_used_for_selection") is not False:
            raise RuntimeError(f"{profile}: test_used_for_selection debe ser false")
        champion = str(report.get("champion_selected_on_tune", ""))
        tune = tune_metrics_for_selected_champion(report)
        selected_name, calibration = selected_metric_from_report(report, "calibration")
        policy = str(report["probability_policy_selected_on_calibration"]["selected"])
        if metadata.get("champion") != champion:
            raise RuntimeError(f"{profile}: champion inconsistente entre report y metadata")
        rows.append({
            "profile": profile,
            "dataset": str(dataset),
            "metadata": str(metadata),
            "candidate_dir": str(candidate),
            "champion_selected_on_tune": champion,
            "probability_policy_selected_on_calibration": policy,
            "tune_log_loss": finite_metric(tune, "log_loss"),
            "tune_brier": finite_metric(tune, "brier_score"),
            "tune_ece": finite_metric(tune, "ece_15_bins"),
            "tune_accuracy": finite_metric(tune, "accuracy"),
            "calibration_metric_name": selected_name,
            "calibration_log_loss": finite_metric(calibration, "log_loss"),
            "calibration_brier": finite_metric(calibration, "brier_score"),
            "calibration_ece": finite_metric(calibration, "ece_15_bins"),
            "feature_count": len(metadata.get("features", [])),
        })
        reports[profile] = report

    summary = pd.DataFrame(rows)
    if summary["tune_log_loss"].isna().any():
        missing = summary.loc[summary["tune_log_loss"].isna(), "profile"].tolist()
        raise RuntimeError(f"Tune Log Loss ausente para perfiles: {missing}")
    summary = summary.sort_values(
        ["tune_log_loss", "calibration_log_loss", "feature_count", "profile"],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.insert(0, "selection_rank", np.arange(1, len(summary) + 1))
    selected_profile = str(summary.iloc[0]["profile"])
    summary["selected_on_tune"] = summary["profile"].eq(selected_profile)

    summary_csv = ablation_root / "ablation_summary.csv"
    summary_json = ablation_root / "ablation_summary.json"
    summary.to_csv(summary_csv, index=False)
    payload = {
        "schema_version": 1,
        "selection_rule": (
            "Minimum tune Log Loss; calibration only chooses raw/calibrated "
            "inside each profile; test is never used for profile selection."
        ),
        "test_used_for_profile_selection": False,
        "selected_profile_on_tune": selected_profile,
        "selected_candidate_dir": str(
            ablation_root / selected_profile / args.ablation_mode
        ),
        "profiles": profiles,
        "row_identity_audit": row_audit,
        "results": summary.to_dict(orient="records"),
    }
    atomic_json(payload, summary_json)
    print("\nABLATION SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nSelected on tune: {selected_profile}")
    print(f"CSV: {summary_csv}")
    print(f"JSON: {summary_json}")
    print(
        "\nTest no se ha usado para seleccionar el perfil. "
        "Promociona posteriormente el candidate elegido con --skip-train."
    )


# ---------------------------------------------------------------------------
# Secondary tennis markets: point model plus hierarchical score simulation
# ---------------------------------------------------------------------------
def _score_totals(
    score: Any,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """Obtiene sets y juegos desde un marcador orientado al ganador.

    Devuelve:
        winner_sets
        loser_sets
        winner_games
        loser_games

    Los marcadores ausentes, walkovers y textos sin sets analizables
    devuelven cuatro valores NaN.
    """

    if score is None or pd.isna(score):
        return (
            math.nan,
            math.nan,
            math.nan,
            math.nan,
        )

    text = str(score).upper().strip()

    missing_scores = {
        "",
        "NAN",
        "NONE",
        "<NA>",
        "W/O",
        "WO",
        "WALKOVER",
    }

    if text in missing_scores:
        return (
            math.nan,
            math.nan,
            math.nan,
            math.nan,
        )

    winner_sets = 0
    loser_sets = 0
    winner_games = 0
    loser_games = 0
    parsed_sets = 0

    for token in text.split():
        cleaned = token.strip()

        score_match = re.match(
            r"^(\d+)-(\d+)"
            r"(?:\([^)]*\))?"
            r"[*]?$",
            cleaned,
        )

        if score_match is None:
            continue

        left = int(
            score_match.group(1)
        )

        right = int(
            score_match.group(2)
        )

        parsed_sets += 1

        winner_games += left
        loser_games += right

        if left > right:
            winner_sets += 1
        elif right > left:
            loser_sets += 1

    if parsed_sets == 0:
        return (
            math.nan,
            math.nan,
            math.nan,
            math.nan,
        )

    return (
        float(winner_sets),
        float(loser_sets),
        float(winner_games),
        float(loser_games),
    )


def _market_targets(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Construye targets secundarios sin incorporarlos al contrato de features."""

    result = frame.copy()

    target = pd.to_numeric(
        result[TARGET],
        errors="coerce",
    )

    winner_service_points_won = (
        pd.to_numeric(
            result["w_1stWon"],
            errors="coerce",
        )
        + pd.to_numeric(
            result["w_2ndWon"],
            errors="coerce",
        )
    )

    winner_service_points = (
        pd.to_numeric(
            result["w_svpt"],
            errors="coerce",
        )
        .replace(
            0,
            np.nan,
        )
    )

    loser_service_points_won = (
        pd.to_numeric(
            result["l_1stWon"],
            errors="coerce",
        )
        + pd.to_numeric(
            result["l_2ndWon"],
            errors="coerce",
        )
    )

    loser_service_points = (
        pd.to_numeric(
            result["l_svpt"],
            errors="coerce",
        )
        .replace(
            0,
            np.nan,
        )
    )

    winner_spw = (
        winner_service_points_won
        / winner_service_points
    )

    loser_spw = (
        loser_service_points_won
        / loser_service_points
    )

    result[
        "target_player_1_serve_point_win"
    ] = winner_spw.where(
        target.eq(1),
        loser_spw,
    )

    result[
        "target_player_2_serve_point_win"
    ] = loser_spw.where(
        target.eq(1),
        winner_spw,
    )

    score_text = result["score"].astype("string").fillna("").str.upper()
    retired_or_incomplete = score_text.str.contains(
        r"\b(?:RET|ABD|DEF|ABN|INT)\b", regex=True, na=False
    )

    parsed_scores = (
        result["score"]
        .astype("object")
        .map(_score_totals)
    )

    totals = pd.DataFrame(
        parsed_scores.tolist(),
        index=result.index,
        columns=[
            "winner_sets",
            "loser_sets",
            "winner_games",
            "loser_games",
        ],
        dtype="float64",
    )

    result = pd.concat(
        [
            result,
            totals,
        ],
        axis=1,
    )

    valid_score = (
        totals.notna()
        .all(axis=1)
    )

    valid_target = target.isin(
        [
            0,
            1,
        ]
    )

    valid = (
        valid_score
        & valid_target
        & ~retired_or_incomplete
    )

    canonical_p1_set = pd.to_numeric(
        result.get("target_player_1_wins_any_set", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    canonical_p2_set = pd.to_numeric(
        result.get("target_player_2_wins_any_set", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    canonical_total_games = pd.to_numeric(
        result.get("target_total_games", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    result["target_player_1_wins_set"] = canonical_p1_set
    result["target_player_2_wins_set"] = canonical_p2_set
    result["target_total_games"] = canonical_total_games

    player_1_won_match = (
        target.eq(1)
    )

    fallback_p1 = valid & result["target_player_1_wins_set"].isna()
    result.loc[
        fallback_p1,
        "target_player_1_wins_set",
    ] = np.where(
        player_1_won_match.loc[
            fallback_p1
        ],
        totals.loc[
            fallback_p1,
            "winner_sets",
        ].ge(1),
        totals.loc[
            fallback_p1,
            "loser_sets",
        ].ge(1),
    ).astype(float)

    fallback_p2 = valid & result["target_player_2_wins_set"].isna()
    result.loc[
        fallback_p2,
        "target_player_2_wins_set",
    ] = np.where(
        player_1_won_match.loc[
            fallback_p2
        ],
        totals.loc[
            fallback_p2,
            "loser_sets",
        ].ge(1),
        totals.loc[
            fallback_p2,
            "winner_sets",
        ].ge(1),
    ).astype(float)

    fallback_total = valid & result["target_total_games"].isna()
    result.loc[
        fallback_total,
        "target_total_games",
    ] = (
        totals.loc[
            fallback_total,
            "winner_games",
        ]
        + totals.loc[
            fallback_total,
            "loser_games",
        ]
    )

    return result


def _numeric_market_features(frame: pd.DataFrame, metadata: dict[str, Any]) -> list[str]:
    """Select numeric prematch variables using the clean-dataset metadata contract."""
    metadata_keys = (
        "no_market_feature_columns", "market_feature_columns",
        "features", "baseline_no_market_feature_columns",
    )
    candidates: list[str] = []
    selected_key: str | None = None
    for key in metadata_keys:
        values = metadata.get(key, [])
        if isinstance(values, list):
            available_values = [str(value) for value in values if str(value) in frame.columns]
            if available_values:
                candidates = available_values
                selected_key = key
                break
    if not candidates:
        raise RuntimeError(
            "Metadata sin lista de features compatible. "
            f"Comprobadas={metadata_keys}; disponibles={sorted(metadata)}"
        )
    excluded = {
        TARGET, "split", "score", "match_date",
        "winner_sets", "loser_sets", "winner_games", "loser_games",
        "target_player_1_serve_point_win", "target_player_2_serve_point_win",
        "target_player_1_wins_set", "target_player_2_wins_set", "target_total_games",
        *MARKET_REQUIRED_STAT_COLUMNS,
    }
    excluded.update({
        "player_1_id", "player_2_id", "player_1_name", "player_2_name",
        "tourney_id", "tourney_name", "match_num", "processing_sequence",
        "source_origin", "source_priority", "eligible_for_model",
    })
    minimum = max(100, int(0.01 * len(frame)))
    selected: list[str] = []
    categorical: list[str] = []
    low_coverage: list[str] = []
    for column in candidates:
        if column in excluded:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        count = int(values.notna().sum())
        if count < minimum:
            if frame[column].notna().any() and count == 0:
                categorical.append(column)
            else:
                low_coverage.append(column)
            continue
        finite = values.dropna().to_numpy(float)
        if len(finite) and not np.isfinite(finite).all():
            continue
        selected.append(column)
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise RuntimeError(
            "No hay features numericas suficientes para mercados. "
            f"key={selected_key}; candidates={len(candidates)}; "
            f"categorical={len(categorical)}; low_coverage={len(low_coverage)}"
        )
    print()
    print(
        "MARKET FEATURE SELECTION"
    )
    print(f"Metadata key: {selected_key}")
    print(f"Candidate features: {len(candidates):,}")
    print(f"Selected numeric features: {len(selected):,}")
    print(f"Categorical excluded: {len(categorical):,}")
    print(f"Low coverage excluded: {len(low_coverage):,}")
    return selected


def _numeric_feature_frame(
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Construye una matriz numérica compatible con scikit-learn.

    Convierte pandas.NA y extensiones nullable a np.nan/float64.
    No realiza imputación, porque esa responsabilidad corresponde
    al SimpleImputer incluido dentro del pipeline.
    """

    missing_columns = sorted(
        set(features)
        - set(frame.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Faltan features requeridas para inferencia "
            f"de mercados: {missing_columns}"
        )

    numeric_frame = pd.DataFrame(
        {
            column: pd.to_numeric(
                frame[column],
                errors="coerce",
            ).astype("float32")
            for column in features
        },
        index=frame.index,
    )

    numeric_frame = numeric_frame.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return numeric_frame

def _fit_spw_model(
    frame: pd.DataFrame,
    features: list[str],
    target_name: str,
) -> Pipeline:
    """Entrena un regresor de probabilidad de punto al servicio."""

    target = pd.to_numeric(
        frame[target_name],
        errors="coerce",
    )

    valid = target.between(
        0.20,
        0.95,
    )

    valid_rows = int(
        valid.sum()
    )

    if valid_rows < 1_000:
        raise RuntimeError(
            f"Muestra insuficiente para {target_name}: "
            f"{valid_rows:,}"
        )

    training_features = _numeric_feature_frame(
        frame.loc[valid],
        features,
    )

    training_target = (
        target.loc[valid]
        .astype("float64")
    )

    completely_missing = [
        column
        for column in features
        if training_features[column]
        .notna()
        .sum()
        == 0
    ]

    if completely_missing:
        raise RuntimeError(
            "El modelo SPW recibió features sin ninguna "
            "observación después de la selección: "
            f"{completely_missing}"
        )

    model = Pipeline(
        [
            (
                "impute",
                SimpleImputer(
                    strategy="constant",
                    fill_value=0.0,
                    add_indicator=False,
                    keep_empty_features=True,
                ),
            ),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.04,
                    max_iter=300,
                    max_leaf_nodes=31,
                    min_samples_leaf=40,
                    l2_regularization=2.0,
                    random_state=20260908,
                ),
            ),
        ]
    )

    model.fit(
        training_features,
        training_target,
    )

    return model


def _simulate_tennis_markets(
    p1_serve_point: float,
    p2_serve_point: float,
    best_of: int,
    over_lines: tuple[float, ...],
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    """Point -> game -> set -> match Monte Carlo under standard tiebreak rules."""
    rng = np.random.default_rng(seed)
    sets_needed = best_of // 2 + 1
    set_scores: Counter[str] = Counter()
    total_games = np.empty(simulations, dtype=np.int16)
    p1_match_wins = p1_set_any = p2_set_any = 0

    def point(server: int) -> int:
        probability = p1_serve_point if server == 1 else p2_serve_point
        return server if rng.random() < probability else (2 if server == 1 else 1)

    def game(server: int) -> int:
        a = b = 0
        while True:
            winner = point(server)
            a += winner == 1; b += winner == 2
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
            a += winner == 1; b += winner == 2; points += 1
            if max(a, b) >= 7 and abs(a - b) >= 2:
                return 1 if a > b else 2

    for simulation in range(simulations):
        sets1 = sets2 = games_total = 0
        first_server = 1 if rng.random() < 0.5 else 2
        while sets1 < sets_needed and sets2 < sets_needed:
            games1 = games2 = 0
            server = first_server
            while True:
                winner = tiebreak(server) if games1 == games2 == 6 else game(server)
                games1 += winner == 1; games2 += winner == 2
                if games1 == games2 == 6:
                    pass
                elif max(games1, games2) >= 6 and abs(games1 - games2) >= 2:
                    break
                elif max(games1, games2) == 7:
                    break
                server = 2 if server == 1 else 1
            games_total += games1 + games2
            sets1 += games1 > games2; sets2 += games2 > games1
            first_server = 2 if server == 1 else 1
        total_games[simulation] = games_total
        p1_match_wins += sets1 > sets2
        p1_set_any += sets1 > 0
        p2_set_any += sets2 > 0
        set_scores[f"{sets1}-{sets2}"] += 1
    return {
        "match_win_player_1": p1_match_wins / simulations,
        "player_1_wins_set": p1_set_any / simulations,
        "player_2_wins_set": p2_set_any / simulations,
        "expected_total_games": float(total_games.mean()),
        "set_score_probabilities": {key: value / simulations for key, value in sorted(set_scores.items())},
        "over_probabilities": {str(line): float(np.mean(total_games > line)) for line in over_lines},
    }


def _market_line_suffix(line: float) -> str:
    return f"{float(line):.1f}".replace(".", "_")


def _expected_calibration_error(
    target: pd.Series,
    probability: pd.Series,
    bins: int = 10,
) -> float:
    frame = pd.DataFrame({
        "target": pd.to_numeric(target, errors="coerce"),
        "probability": pd.to_numeric(probability, errors="coerce"),
    }).dropna()
    frame = frame.loc[frame["target"].isin([0, 1])].copy()
    if frame.empty:
        return math.nan
    frame["probability"] = frame["probability"].clip(0.0, 1.0)
    frame["band"] = pd.cut(
        frame["probability"],
        bins=np.linspace(0.0, 1.0, bins + 1),
        include_lowest=True,
        duplicates="drop",
    )
    grouped = frame.groupby("band", observed=True).agg(
        rows=("target", "size"),
        mean_probability=("probability", "mean"),
        observed_rate=("target", "mean"),
    )
    return float((
        grouped["rows"] / len(frame)
        * (grouped["mean_probability"] - grouped["observed_rate"]).abs()
    ).sum())


def _binary_market_metrics(
    frame: pd.DataFrame,
    market: str,
    target_column: str,
    probability_column: str,
    best_of: int | None = None,
    line: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    selected = frame[[target_column, probability_column]].copy()
    selected.columns = ["target", "probability"]
    selected["target"] = pd.to_numeric(selected["target"], errors="coerce")
    selected["probability"] = pd.to_numeric(selected["probability"], errors="coerce")
    selected = selected.dropna()
    selected = selected.loc[selected["target"].isin([0, 1])].copy()
    if selected.empty:
        raise RuntimeError(f"No hay observaciones validas para {market}")
    selected["probability"] = selected["probability"].clip(1e-6, 1 - 1e-6)
    target = selected["target"]
    probability = selected["probability"]
    positive_rate = float(target.mean())
    baseline_probability = np.full(len(selected), positive_rate, dtype=float)
    auc = (
        float(roc_auc_score(target, probability))
        if target.nunique() == 2 else math.nan
    )
    metrics = {
        "market": market,
        "best_of": best_of,
        "line": line,
        "rows": int(len(selected)),
        "positive_rate": positive_rate,
        "mean_probability": float(probability.mean()),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "baseline_log_loss": float(log_loss(target, baseline_probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probability)),
        "baseline_brier_score": float(brier_score_loss(target, baseline_probability)),
        "roc_auc": auc,
        "accuracy_at_0_50": float(accuracy_score(target, probability.ge(0.5))),
        "baseline_accuracy": float(max(positive_rate, 1.0 - positive_rate)),
        "ece_10_bins": _expected_calibration_error(target, probability, 10),
    }
    metrics["log_loss_gain_vs_baseline"] = (
        metrics["baseline_log_loss"] - metrics["log_loss"]
    )
    metrics["brier_gain_vs_baseline"] = (
        metrics["baseline_brier_score"] - metrics["brier_score"]
    )

    bands = selected.copy()
    bands["probability_band"] = pd.cut(
        bands["probability"],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
        duplicates="drop",
    )
    calibration = bands.groupby("probability_band", observed=True).agg(
        rows=("target", "size"),
        mean_probability=("probability", "mean"),
        observed_rate=("target", "mean"),
    ).reset_index()
    calibration.insert(0, "market", market)
    calibration.insert(1, "best_of", best_of)
    calibration.insert(2, "line", line)
    calibration["calibration_gap"] = (
        calibration["mean_probability"] - calibration["observed_rate"]
    )

    segment_rows: list[dict[str, Any]] = []
    for threshold in (0.80, 0.90, 0.95, 0.99):
        mask = probability.ge(threshold)
        count = int(mask.sum())
        segment_rows.append({
            "market": market,
            "best_of": best_of,
            "line": line,
            "minimum_probability": threshold,
            "rows": count,
            "mean_probability": float(probability.loc[mask].mean()) if count else math.nan,
            "observed_rate": float(target.loc[mask].mean()) if count else math.nan,
        })
    segments = pd.DataFrame(segment_rows)
    segments["calibration_gap"] = (
        segments["mean_probability"] - segments["observed_rate"]
    )
    return metrics, calibration, segments


def _swap_market_feature_values(
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Invierte P1/P2 construyendo el DataFrame una sola vez."""
    source = frame.copy()
    transformed: dict[str, Any] = {}
    missing_series = pd.Series(np.nan, index=source.index, dtype="float64")

    for feature in features:
        if feature.startswith("player_1_"):
            counterpart = "player_2_" + feature[len("player_1_"):]
            transformed[feature] = source.get(counterpart, missing_series)
        elif feature.startswith("player_2_"):
            counterpart = "player_1_" + feature[len("player_2_"):]
            transformed[feature] = source.get(counterpart, missing_series)
        elif feature.startswith("diff_"):
            transformed[feature] = -pd.to_numeric(source[feature], errors="coerce")
        elif feature.endswith("_p1_p2"):
            value = pd.to_numeric(source[feature], errors="coerce")
            transformed[feature] = pd.Series(
                np.where(value.ne(0), 1.0 / value, np.nan),
                index=source.index,
                dtype="float64",
            )
        elif feature in {"elo_expected_p1", "surface_elo_expected_p1"}:
            transformed[feature] = 1.0 - pd.to_numeric(
                source[feature], errors="coerce"
            )
        elif "_player_1_before" in feature:
            value = pd.to_numeric(source[feature], errors="coerce")
            if any(token in feature for token in ("rate", "share", "probability")):
                transformed[feature] = 1.0 - value
            elif any(token in feature for token in ("score", "intransitivity")):
                transformed[feature] = -value
            else:
                transformed[feature] = value
        else:
            transformed[feature] = source[feature]

    mirrored = pd.DataFrame(
        transformed,
        index=source.index,
        columns=features,
    ).copy()
    return _numeric_feature_frame(mirrored, features)

def _fit_mirrored_spw_models(frame: pd.DataFrame, features: list[str]) -> tuple[Pipeline, Pipeline, dict[str, Any]]:
    target1 = pd.to_numeric(frame["target_player_1_serve_point_win"], errors="coerce")
    target2 = pd.to_numeric(frame["target_player_2_serve_point_win"], errors="coerce")
    valid = target1.between(0.20, 0.95) & target2.between(0.20, 0.95)
    base = frame.loc[valid].copy()
    original = _numeric_feature_frame(base, features)
    mirrored = _swap_market_feature_values(base, features)

    def fit(x: pd.DataFrame, y: pd.Series) -> Pipeline:
        model = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=False, keep_empty_features=True)),
            ("regressor", HistGradientBoostingRegressor(
                loss="squared_error", learning_rate=0.04, max_iter=300,
                max_leaf_nodes=31, min_samples_leaf=40,
                l2_regularization=2.0, random_state=20260908,
            )),
        ])
        model.fit(x, y.astype("float64"))
        return model

    x = pd.concat([original, mirrored], ignore_index=True)
    p1_y = pd.concat([target1.loc[valid], target2.loc[valid]], ignore_index=True)
    p2_y = pd.concat([target2.loc[valid], target1.loc[valid]], ignore_index=True)
    audit = {"architecture": "mirrored_side_augmentation", "base_rows": int(len(base)),
             "rows_per_side_model": int(len(x)), "shared_calibration": True}
    return fit(x, p1_y), fit(x, p2_y), audit


def _swapped_market_feature_frame(row: pd.Series, features: list[str]) -> pd.DataFrame:
    return _swap_market_feature_values(pd.DataFrame([row]), features)


def _sample_strength(rows: int) -> str:
    if rows >= 300:
        return "validated_sample"
    if rows >= 100:
        return "provisional_sample"
    if rows >= 50:
        return "exploratory_sample"
    return "insufficient_sample"


def _market_reliability_label(metrics: dict[str, Any]) -> str:
    rows = int(metrics.get("rows", 0))
    auc = float(metrics.get("roc_auc", math.nan))
    ece = float(metrics.get("ece_10_bins", math.nan))
    log_gain = float(metrics.get("log_loss_gain_vs_baseline", math.nan))
    brier_gain = float(metrics.get("brier_gain_vs_baseline", math.nan))
    market = str(metrics.get("market", ""))

    if rows < 50:
        return "insufficient_sample"
    if not all(math.isfinite(value) for value in (auc, ece, log_gain, brier_gain)):
        return "not_validated"
    if log_gain <= 0 or brier_gain <= 0:
        return "not_validated"

    if market == "over_21.5_bo3":
        if rows >= 300 and auc >= 0.60 and ece <= 0.08 and log_gain >= 0.01 and brier_gain >= 0.005:
            return "validated"
        if rows >= 100 and auc >= 0.58 and ece <= 0.10:
            return "provisional"
        return "experimental"

    if rows >= 300 and auc >= 0.70 and ece <= 0.07:
        return "validated"
    if rows >= 100 and auc >= 0.65 and ece <= 0.09:
        return "provisional"
    return "experimental"


def _segment_definitions(validation: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    definitions: list[tuple[str, str, pd.Series]] = [
        ("global", "ALL", pd.Series(True, index=validation.index)),
    ]
    for column in ("competition_type", "surface"):
        if column not in validation:
            continue
        normalized = validation[column].astype("string").fillna("UNKNOWN").str.upper()
        for value in sorted(normalized.dropna().unique().tolist()):
            definitions.append((column, str(value), normalized.eq(value)))

    if {"competition_type", "surface"}.issubset(validation.columns):
        competition = validation["competition_type"].astype("string").fillna("UNKNOWN").str.upper()
        surface = validation["surface"].astype("string").fillna("UNKNOWN").str.upper()
        combination = competition + "|" + surface
        for value in sorted(combination.dropna().unique().tolist()):
            definitions.append(("competition_surface", str(value), combination.eq(value)))

    for column in (
        "cold_start_any_player",
        "synthetic_player_any",
        "stats_reliable_both",
        "surface_stats_reliable_both",
    ):
        if column not in validation:
            continue
        numeric = pd.to_numeric(validation[column], errors="coerce")
        for value in (0, 1):
            if numeric.eq(value).any():
                definitions.append((column, str(value), numeric.eq(value)))
    return definitions


def _write_segment_reports(
    validation: pd.DataFrame,
    report_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    markets = [
        ("player_1_wins_set", "target_player_1_wins_set", "probability_player_1_wins_set"),
        ("player_2_wins_set", "target_player_2_wins_set", "probability_player_2_wins_set"),
        ("over_21.5_bo3", "target_over_21_5", "probability_over_21_5"),
    ]
    metric_rows: list[dict[str, Any]] = []
    calibration_frames: list[pd.DataFrame] = []

    for segment_name, segment_value, segment_mask in _segment_definitions(validation):
        for market, target_column, probability_column in markets:
            if target_column not in validation or probability_column not in validation:
                continue
            source = validation.loc[segment_mask].copy()
            if market == "over_21.5_bo3":
                source = source.loc[pd.to_numeric(source["best_of"], errors="coerce").eq(3)]
            if source[[target_column, probability_column]].dropna().empty:
                continue
            metrics, calibration, _ = _binary_market_metrics(
                source,
                market,
                target_column,
                probability_column,
                3 if market == "over_21.5_bo3" else None,
                MARKET_PRIMARY_OVER_LINE if market == "over_21.5_bo3" else None,
            )
            metrics.update({
                "segment": segment_name,
                "segment_value": segment_value,
                "sample_strength": _sample_strength(int(metrics["rows"])),
            })
            metrics["reliability"] = _market_reliability_label(metrics)
            metric_rows.append(metrics)
            calibration.insert(0, "segment", segment_name)
            calibration.insert(1, "segment_value", segment_value)
            calibration_frames.append(calibration)

    metrics_frame = pd.DataFrame(metric_rows)
    calibration_frame = (
        pd.concat(calibration_frames, ignore_index=True)
        if calibration_frames else pd.DataFrame()
    )
    reliability_columns = [
        "market", "segment", "segment_value", "rows", "sample_strength",
        "reliability", "positive_rate", "mean_probability", "roc_auc",
        "ece_10_bins", "log_loss_gain_vs_baseline", "brier_gain_vs_baseline",
    ]
    reliability_frame = (
        metrics_frame.loc[:, [column for column in reliability_columns if column in metrics_frame]].copy()
        if not metrics_frame.empty else pd.DataFrame(columns=reliability_columns)
    )
    if not reliability_frame.empty:
        order = {"validated": 0, "provisional": 1, "experimental": 2,
                 "not_validated": 3, "insufficient_sample": 4}
        reliability_frame["reliability_order"] = reliability_frame["reliability"].map(order).fillna(9)
        reliability_frame = reliability_frame.sort_values(
            ["market", "reliability_order", "rows"],
            ascending=[True, True, False],
            kind="mergesort",
        ).drop(columns="reliability_order")

    metrics_frame.to_csv(report_directory / MARKET_SEGMENT_METRICS_FILE, index=False)
    calibration_frame.to_csv(report_directory / MARKET_SEGMENT_CALIBRATION_FILE, index=False)
    reliability_frame.to_csv(report_directory / MARKET_RELIABILITY_FILE, index=False)
    return metrics_frame, calibration_frame, reliability_frame


def _write_market_validation_reports(
    validation: pd.DataFrame,
    candidate: Path,
) -> dict[str, Any]:
    report_directory = candidate / MARKET_REPORT_DIRECTORY
    report_directory.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    calibration_frames: list[pd.DataFrame] = []
    segment_frames: list[pd.DataFrame] = []

    definitions = [
        ("player_1_wins_set", "target_player_1_wins_set", "probability_player_1_wins_set", None, None),
        ("player_2_wins_set", "target_player_2_wins_set", "probability_player_2_wins_set", None, None),
    ]
    for best_of, lines in ((3, MARKET_BO3_OVER_LINES), (5, MARKET_BO5_OVER_LINES)):
        for line in lines:
            suffix = _market_line_suffix(line)
            definitions.append((
                f"over_{line:.1f}_bo{best_of}",
                f"target_over_{suffix}",
                f"probability_over_{suffix}",
                best_of,
                line,
            ))

    for market, target_column, probability_column, best_of, line in definitions:
        if target_column not in validation or probability_column not in validation:
            continue
        source = validation
        if best_of is not None:
            source = source.loc[pd.to_numeric(source["best_of"], errors="coerce").eq(best_of)]
        if source[[target_column, probability_column]].dropna().empty:
            continue
        metrics, calibration, segments = _binary_market_metrics(
            source, market, target_column, probability_column, best_of, line
        )
        metric_rows.append(metrics)
        calibration_frames.append(calibration)
        segment_frames.append(segments)

    metrics_frame = pd.DataFrame(metric_rows)
    calibration_frame = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()
    segments_frame = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()
    metrics_frame.to_csv(report_directory / "binary_market_metrics.csv", index=False)
    calibration_frame.to_csv(report_directory / "market_calibration_bands.csv", index=False)
    segments_frame.to_csv(report_directory / "high_probability_segments.csv", index=False)
    segment_metrics, segment_calibration, reliability_summary = _write_segment_reports(
        validation,
        report_directory,
    )

    actual_games = pd.to_numeric(validation["target_total_games"], errors="coerce")
    expected_games = pd.to_numeric(validation["expected_total_games"], errors="coerce")
    valid_games = actual_games.notna() & expected_games.notna()
    total_games_rows: list[dict[str, Any]] = []
    for label, mask in (
        ("all", valid_games),
        ("bo3", valid_games & pd.to_numeric(validation["best_of"], errors="coerce").eq(3)),
        ("bo5", valid_games & pd.to_numeric(validation["best_of"], errors="coerce").eq(5)),
    ):
        if not mask.any():
            continue
        truth = actual_games.loc[mask]
        prediction = expected_games.loc[mask]
        total_games_rows.append({
            "segment": label,
            "rows": int(mask.sum()),
            "actual_mean_games": float(truth.mean()),
            "predicted_mean_games": float(prediction.mean()),
            "mean_error": float((prediction - truth).mean()),
            "mae": float(mean_absolute_error(truth, prediction)),
            "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        })
    total_games_frame = pd.DataFrame(total_games_rows)
    total_games_frame.to_csv(report_directory / "total_games_metrics.csv", index=False)

    symmetry_columns = [
        "symmetry_spw_p1_vs_swapped_p2_abs_error",
        "symmetry_spw_p2_vs_swapped_p1_abs_error",
        "symmetry_set_p1_vs_swapped_p2_abs_error",
        "symmetry_set_p2_vs_swapped_p1_abs_error",
    ]
    symmetry_frame = validation[[
        column for column in ["best_of", *symmetry_columns]
        if column in validation.columns
    ]].copy()
    symmetry_frame.to_csv(report_directory / "spw_symmetry_validation.csv", index=False)
    symmetry_summary = {
        column: {
            "mean_absolute_difference": float(pd.to_numeric(symmetry_frame[column], errors="coerce").mean()),
            "p95_absolute_difference": float(pd.to_numeric(symmetry_frame[column], errors="coerce").quantile(0.95)),
            "maximum_absolute_difference": float(pd.to_numeric(symmetry_frame[column], errors="coerce").max()),
        }
        for column in symmetry_columns
        if column in symmetry_frame
    }

    summary = {
        "schema_version": 2,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "test_rows": int(len(validation)),
        "test_used_for_fitting_or_calibration": False,
        "binary_markets": metrics_frame.to_dict(orient="records"),
        "total_games": total_games_frame.to_dict(orient="records"),
        "symmetry": symmetry_summary,
        "segment_validation": {
            "rows": int(len(segment_metrics)),
            "reliability_counts": (
                reliability_summary["reliability"].value_counts().to_dict()
                if not reliability_summary.empty else {}
            ),
            "validated_or_provisional": (
                reliability_summary.loc[
                    reliability_summary["reliability"].isin(["validated", "provisional"])
                ].to_dict(orient="records")
                if not reliability_summary.empty else []
            ),
        },
        "files": {
            "binary_metrics": "binary_market_metrics.csv",
            "calibration_bands": "market_calibration_bands.csv",
            "high_probability_segments": "high_probability_segments.csv",
            "total_games_metrics": "total_games_metrics.csv",
            "symmetry_validation": "spw_symmetry_validation.csv",
            "segment_metrics": MARKET_SEGMENT_METRICS_FILE,
            "segment_calibration": MARKET_SEGMENT_CALIBRATION_FILE,
            "reliability_summary": MARKET_RELIABILITY_FILE,
        },
    }
    atomic_json(summary, report_directory / MARKET_SUMMARY_FILE)
    print("\nMARKET TEST VALIDATION")
    print(metrics_frame.to_string(index=False))
    print("\nTOTAL GAMES")
    print(total_games_frame.to_string(index=False))
    print("\nSYMMETRY")
    print(json.dumps(symmetry_summary, indent=2, ensure_ascii=False))
    print(f"\nReports: {report_directory}")
    return summary


def train_market_models(
    dataset_path: Path,
    dataset_metadata_path: Path,
    candidate: Path,
    simulations: int,
    validation_rows_limit: int,
    validation_simulations: int,
    symmetry_rows_limit: int,
) -> None:
    """Fit serve-point models and validate derived set/total market probabilities."""
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    frame = pd.read_parquet(dataset_path, engine="pyarrow")
    missing = sorted({TARGET, "split", *MARKET_REQUIRED_STAT_COLUMNS} - set(frame.columns))
    if missing:
        raise RuntimeError(f"Dataset no permite entrenar mercados. Faltan: {missing}")
    dataset_metadata = read_json(dataset_metadata_path) if dataset_metadata_path.exists() else {}
    frame = _market_targets(frame)
    features = _numeric_market_features(frame, dataset_metadata)
    split_values = (
        frame["split"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    train = frame.loc[
        split_values.eq("train")
    ].copy()

    tune = frame.loc[
        split_values.eq("tune")
    ].copy()

    calibration = frame.loc[
        split_values.eq("calibration")
    ].copy()

    validation = frame.loc[
        split_values.eq("validation")
    ].copy()

    test = frame.loc[
        split_values.eq("test")
    ].copy()


    if tune.empty and calibration.empty:
        if validation.empty:
            raise RuntimeError(
                "El dataset no contiene tune/calibration "
                "ni un split validation divisible."
            )

        validation = (
            validation.sort_values(
                [
                    column
                    for column in (
                        "match_date",
                        "tourney_id",
                        "competition_type",
                        "match_num",
                    )
                    if column in validation.columns
                ],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        if "match_date" not in validation.columns:
            raise RuntimeError(
                "Validation no contiene match_date. "
                "No se puede crear una división temporal "
                "segura para tune y calibration."
            )

        validation_dates = pd.to_datetime(
            validation["match_date"],
            errors="coerce",
        ).dt.normalize()

        if validation_dates.isna().any():
            raise RuntimeError(
                "Validation contiene fechas nulas o inválidas: "
                f"{int(validation_dates.isna().sum()):,}"
            )

        unique_validation_dates = (
            pd.Index(
                validation_dates.unique()
            )
            .sort_values()
        )

        if len(unique_validation_dates) < 2:
            raise RuntimeError(
                "Validation necesita al menos dos fechas "
                "diferentes para crear tune y calibration. "
                f"Fechas disponibles: {len(unique_validation_dates)}"
            )

        date_split_position = (
            len(unique_validation_dates)
            // 2
        )

        if (
            date_split_position <= 0
            or date_split_position
            >= len(unique_validation_dates)
        ):
            raise RuntimeError(
                "No se pudo obtener una frontera temporal "
                "válida dentro de validation."
            )

        calibration_start_date = (
            unique_validation_dates[
                date_split_position
            ]
        )

        tune_mask = (
            validation_dates
            < calibration_start_date
        )

        calibration_mask = (
            validation_dates
            >= calibration_start_date
        )

        tune = validation.loc[
            tune_mask
        ].copy()

        calibration = validation.loc[
            calibration_mask
        ].copy()

        if tune.empty:
            raise RuntimeError(
                "La división temporal dejó tune vacío. "
                f"Calibration start: {calibration_start_date}"
            )

        if calibration.empty:
            raise RuntimeError(
                "La división temporal dejó calibration vacío. "
                f"Calibration start: {calibration_start_date}"
            )

        tune_max_date = pd.to_datetime(
            tune["match_date"],
            errors="coerce",
        ).max()

        calibration_min_date = pd.to_datetime(
            calibration["match_date"],
            errors="coerce",
        ).min()

        if (
            pd.isna(tune_max_date)
            or pd.isna(calibration_min_date)
        ):
            raise RuntimeError(
                "No se pudieron calcular las fronteras "
                "temporales de tune y calibration."
            )

        if tune_max_date >= calibration_min_date:
            raise RuntimeError(
                "La división temporal de validation "
                "continúa solapada: "
                f"tune_max={tune_max_date}; "
                f"calibration_min={calibration_min_date}"
            )

        print()
        print("MARKET TEMPORAL SPLIT")
        print(
            "Validation rows:",
            f"{len(validation):,}",
        )
        print(
            "Validation unique dates:",
            f"{len(unique_validation_dates):,}",
        )
        print(
            "Tune rows:",
            f"{len(tune):,}",
        )
        print(
            "Tune period:",
            pd.to_datetime(
                tune["match_date"],
                errors="coerce",
            ).min(),
            "->",
            tune_max_date,
        )
        print(
            "Calibration rows:",
            f"{len(calibration):,}",
        )
        print(
            "Calibration period:",
            calibration_min_date,
            "->",
            pd.to_datetime(
                calibration["match_date"],
                errors="coerce",
            ).max(),
        )


    elif tune.empty and not validation.empty:
        tune = validation.copy()


    elif calibration.empty and not validation.empty:
        calibration = validation.copy()


    model_training = pd.concat(
        [
            train,
            tune,
        ],
        ignore_index=True,
    )


    if model_training.empty:
        raise RuntimeError(
            "No hay filas disponibles para entrenar "
            "los modelos de mercados."
        )

    if calibration.empty:
        raise RuntimeError(
            "No hay filas disponibles para calibrar "
            "los modelos de mercados."
        )

    if test.empty:
        raise RuntimeError(
            "No hay filas test para evaluar "
            "los modelos de mercados."
        )


    training_max_date = pd.to_datetime(
        model_training.get(
            "match_date"
        ),
        errors="coerce",
    ).max()

    calibration_min_date = pd.to_datetime(
        calibration.get(
            "match_date"
        ),
        errors="coerce",
    ).min()

    test_min_date = pd.to_datetime(
        test.get(
            "match_date"
        ),
        errors="coerce",
    ).min()


    if (
        pd.notna(training_max_date)
        and pd.notna(calibration_min_date)
        and training_max_date
        >= calibration_min_date
    ):
        raise RuntimeError(
            "Solapamiento temporal entre entrenamiento "
            "y calibration de mercados: "
            f"training_max={training_max_date}; "
            f"calibration_min={calibration_min_date}"
        )

    calibration_max_date = pd.to_datetime(
        calibration["match_date"],
        errors="coerce",
    ).max()

    if (
        pd.notna(calibration_max_date)
        and pd.notna(test_min_date)
        and calibration_max_date
        >= test_min_date
    ):
        raise RuntimeError(
            "Calibration de mercados no es anterior "
            "al test temporal: "
            f"calibration_max={calibration_max_date}; "
            f"test_min={test_min_date}"
        )
        
    training_numeric_coverage = {
        column: int(
            pd.to_numeric(
                model_training[column],
                errors="coerce",
            )
            .notna()
            .sum()
        )
        for column in features
    }

    empty_training_features = [
        column
        for column, count
        in training_numeric_coverage.items()
        if count == 0
    ]

    if empty_training_features:
        print()
        print(
            "FEATURES REMOVED FROM MARKET MODELS"
        )

        print(
            "No observations in model_training:",
            empty_training_features,
        )

        features = [
            column
            for column in features
            if column
            not in empty_training_features
        ]

    if not features:
        raise RuntimeError(
            "Todas las features de mercados están vacías "
            "en model_training."
        )

    print(
        "Final market features:",
        f"{len(features):,}",
    )
    
    p1_model, p2_model, mirrored_training_audit = _fit_mirrored_spw_models(
        model_training, features
    )

    # Linear bias correction fitted only on calibration.
    calibration_valid = calibration[["target_player_1_serve_point_win", "target_player_2_serve_point_win"]].notna().all(axis=1)
    calibration_features = (
        _numeric_feature_frame(
            calibration.loc[
                calibration_valid
            ],
            features,
        )
    )

    raw1 = np.clip(
        p1_model.predict(
            calibration_features
        ),
        0.35,
        0.85,
    )

    raw2 = np.clip(
        p2_model.predict(
            calibration_features
        ),
        0.35,
        0.85,
    )
    truth1 = calibration.loc[calibration_valid, "target_player_1_serve_point_win"].to_numpy(float)
    truth2 = calibration.loc[calibration_valid, "target_player_2_serve_point_win"].to_numpy(float)
    shared_correction = np.polyfit(
        np.concatenate([raw1, raw2]), np.concatenate([truth1, truth2]), 1
    ).tolist()
    correction1 = list(shared_correction)
    correction2 = list(shared_correction)

    validation_rows = []
    valid_test = test[["target_player_1_serve_point_win", "target_player_2_serve_point_win", "target_total_games"]].notna().all(axis=1)
    sample = test.loc[valid_test].copy()
    sample = sample.sort_values(
        [
            column
            for column in (
                "match_date",
                "tourney_id",
                "match_num",
            )
            if column in sample.columns
        ],
        kind="mergesort",
    ).copy()

    if validation_rows_limit > 0 and len(sample) > validation_rows_limit:
        sample = sample.sample(
            n=validation_rows_limit,
            random_state=20260908,
        ).sort_values(
            [
                column
                for column in (
                    "match_date",
                    "tourney_id",
                    "match_num",
                )
                if column in sample.columns
            ],
            kind="mergesort",
        )
    for row_number, (
        _,
        row,
    ) in enumerate(
        sample.iterrows()
    ):
        x = _numeric_feature_frame(
            pd.DataFrame(
                [
                    {
                        feature: row.get(
                            feature,
                            np.nan,
                        )
                        for feature in features
                    }
                ]
            ),
            features,
        )

        raw_spw1 = float(
            p1_model.predict(
                x
            )[0]
        )

        raw_spw2 = float(
            p2_model.predict(
                x
            )[0]
        )

        spw1 = float(
            np.clip(
                np.polyval(
                    correction1,
                    raw_spw1,
                ),
                0.40,
                0.82,
            )
        )

        spw2 = float(
            np.clip(
                np.polyval(
                    correction2,
                    raw_spw2,
                ),
                0.40,
                0.82,
            )
        )
        run_symmetry = symmetry_rows_limit <= 0 or row_number < symmetry_rows_limit
        swapped_spw1 = swapped_spw2 = np.nan
        if run_symmetry:
            swapped_x = _swapped_market_feature_frame(row, features)
            swapped_raw_spw1 = float(p1_model.predict(swapped_x)[0])
            swapped_raw_spw2 = float(p2_model.predict(swapped_x)[0])
            swapped_spw1 = float(np.clip(np.polyval(shared_correction, swapped_raw_spw1), 0.40, 0.82))
            swapped_spw2 = float(np.clip(np.polyval(shared_correction, swapped_raw_spw2), 0.40, 0.82))
        best_of_value = pd.to_numeric(
            row.get(
                "best_of",
                3,
            ),
            errors="coerce",
        )

        best_of = (
            int(best_of_value)
            if pd.notna(best_of_value)
            and int(best_of_value) in {3, 5}
            else 3
        )

        validation_seed = 20_260_908 + row_number
        prediction = _simulate_tennis_markets(
            p1_serve_point=spw1,
            p2_serve_point=spw2,
            best_of=best_of,
            over_lines=MARKET_DEFAULT_OVER_LINES,
            simulations=validation_simulations,
            seed=validation_seed,
        )
        swapped_prediction = None
        if run_symmetry:
            swapped_prediction = _simulate_tennis_markets(
                p1_serve_point=swapped_spw1,
                p2_serve_point=swapped_spw2,
                best_of=best_of,
                over_lines=MARKET_DEFAULT_OVER_LINES,
                simulations=validation_simulations,
                seed=validation_seed,
            )
        validation_row = {
            "competition_type": str(row.get("competition_type", "UNKNOWN")).strip().upper(),
            "surface": str(row.get("surface", "UNKNOWN")).strip().upper(),
            "tourney_level": str(row.get("tourney_level", "UNKNOWN")).strip().upper(),
            "cold_start_any_player": pd.to_numeric(row.get("cold_start_any_player"), errors="coerce"),
            "synthetic_player_any": pd.to_numeric(row.get("synthetic_player_any"), errors="coerce"),
            "stats_reliable_both": pd.to_numeric(row.get("stats_reliable_both"), errors="coerce"),
            "surface_stats_reliable_both": pd.to_numeric(
                row.get("surface_stats_reliable_both"), errors="coerce"
            ),
            "target_player_1_wins_set": row["target_player_1_wins_set"],
            "probability_player_1_wins_set": prediction["player_1_wins_set"],
            "target_player_2_wins_set": row["target_player_2_wins_set"],
            "probability_player_2_wins_set": prediction["player_2_wins_set"],
            "target_total_games": row["target_total_games"],
            "expected_total_games": prediction["expected_total_games"],
            "predicted_player_1_serve_point_win": spw1,
            "predicted_player_2_serve_point_win": spw2,
            "swapped_predicted_player_1_serve_point_win": swapped_spw1,
            "swapped_predicted_player_2_serve_point_win": swapped_spw2,
            "symmetry_spw_p1_vs_swapped_p2_abs_error": abs(spw1 - swapped_spw2) if run_symmetry else np.nan,
            "symmetry_spw_p2_vs_swapped_p1_abs_error": abs(spw2 - swapped_spw1) if run_symmetry else np.nan,
            "symmetry_set_p1_vs_swapped_p2_abs_error": abs(prediction["player_1_wins_set"] - swapped_prediction["player_2_wins_set"])
            if run_symmetry else np.nan,
            "symmetry_set_p2_vs_swapped_p1_abs_error": abs(prediction["player_2_wins_set"] - swapped_prediction["player_1_wins_set"])
            if run_symmetry else np.nan,
            "best_of": best_of,
            "validation_simulations": validation_simulations,
        }
        target_total_games = pd.to_numeric(row["target_total_games"], errors="coerce")
        applicable_lines = MARKET_BO5_OVER_LINES if best_of == 5 else MARKET_BO3_OVER_LINES
        for line in MARKET_DEFAULT_OVER_LINES:
            suffix = _market_line_suffix(line)
            if line in applicable_lines:
                validation_row[f"probability_over_{suffix}"] = prediction[
                    "over_probabilities"
                ][str(line)]
                validation_row[f"target_over_{suffix}"] = float(
                    target_total_games > line
                ) if pd.notna(target_total_games) else np.nan
            else:
                validation_row[f"probability_over_{suffix}"] = np.nan
                validation_row[f"target_over_{suffix}"] = np.nan
        validation_rows.append(validation_row)
    validation = pd.DataFrame(validation_rows)
    if validation.empty:
        raise RuntimeError(
            "La validación de mercados quedó vacía."
        )

    probability_columns = [
        "probability_player_1_wins_set",
        "probability_player_2_wins_set",
    ]

    for column in probability_columns:
        values = pd.to_numeric(
            validation[column],
            errors="coerce",
        )

        if values.isna().any():
            raise RuntimeError(
                f"{column} contiene valores nulos."
            )

        if not values.between(
            0.0,
            1.0,
        ).all():
            raise RuntimeError(
                f"{column} contiene probabilidades "
                "fuera de [0, 1]."
            )

    expected_games = pd.to_numeric(
        validation[
            "expected_total_games"
        ],
        errors="coerce",
    )

    if expected_games.isna().any():
        raise RuntimeError(
            "expected_total_games contiene "
            "valores nulos."
        )

    if expected_games.le(0).any():
        raise RuntimeError(
            "expected_total_games contiene "
            "valores no positivos."
        )
    over_probability_columns = [
        column for column in validation.columns
        if column.startswith("probability_over_")
    ]
    for column in over_probability_columns:
        values = pd.to_numeric(validation[column], errors="coerce").dropna()
        if not values.between(0.0, 1.0).all():
            raise RuntimeError(f"{column} contiene probabilidades fuera de [0, 1]")
    validation.to_csv(candidate / MARKET_VALIDATION_FILE, index=False)
    market_validation_summary = _write_market_validation_reports(validation, candidate)
    artifact = {
        "schema_version": 1,
        "method": "mirrored_side_augmented_serve_point_regression_plus_hierarchical_monte_carlo",
        "features": features,
        "spw_architecture": mirrored_training_audit,
        "player_1_serve_point_model": p1_model,
        "player_2_serve_point_model": p2_model,
        "player_1_calibration_linear": correction1,
        "player_2_calibration_linear": correction2,
        "serve_point_probability_bounds": [0.40, 0.82],
        "default_over_lines": list(MARKET_DEFAULT_OVER_LINES),
        "recommended_simulations": simulations,
    }
    joblib.dump(artifact, candidate / MARKET_MODEL_FILE, compress=3)
    metadata = {
        "schema_version": 1,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_file": MARKET_MODEL_FILE,
        "validation_file": MARKET_VALIDATION_FILE,
        "method": artifact["method"],
        "features": features,
        "training_rows": int(
            len(model_training)
        ),
        "train_rows": int(
            len(train)
        ),
        "tune_rows": int(
            len(tune)
        ),

        "calibration_rows": len(calibration),
        "test_validation_rows": len(validation),
        "segment_dimensions": [
            "competition_type", "surface", "competition_surface",
            "cold_start_any_player", "synthetic_player_any",
            "stats_reliable_both", "surface_stats_reliable_both",
        ],
        "primary_segmented_markets": [
            "player_1_wins_set", "player_2_wins_set", "over_21.5_bo3"
        ],
        "validation_reports_directory": MARKET_REPORT_DIRECTORY,
        "market_validation_summary": market_validation_summary,
        "over_lines_by_format": {
            "best_of_3": list(MARKET_BO3_OVER_LINES),
            "best_of_5": list(MARKET_BO5_OVER_LINES),
        },
        "targets": ["player_1_wins_at_least_one_set", "player_2_wins_at_least_one_set", "total_games_over_line", "exact_set_score"],
        "selection_policy": "No market target uses test for fitting or calibration; test is reporting only.",
        "references": [
            "Ingram, A point-based Bayesian hierarchical model to predict tennis matches, JQAS, 2019.",
            "Gollub, Producing Win Probabilities for Professional Tennis Matches from any Score, Harvard, 2017.",
            "Klaassen and Magnus, Are Points in Tennis Independent and Identically Distributed?, 2001.",
        ],
    }
    atomic_json(metadata, candidate / MARKET_METADATA_FILE)
    # Reload check.
    reloaded = joblib.load(candidate / MARKET_MODEL_FILE)
    if reloaded.get("features") != features:
        raise RuntimeError("Market model reload inconsistente")
    print(f"[OK] Modelos de mercados: {candidate / MARKET_MODEL_FILE}")

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    root = find_project_root(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(
        description="Entrena candidate y opcionalmente promociona el champion",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--trainer",
        default=str(
            root / "modeling" / "train_professional_tennis_model_professional_v5.py"
        ),
    )
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--candidate-dir")
    parser.add_argument(
        "--direct-market-dir",
        help=(
            "Directorio generado por train_tennis_direct_market_models_v3.py. "
            "Se valida y copia dentro del candidate antes de registrar/promocionar."
        ),
    )
    parser.add_argument(
        "--require-direct-market-models",
        action="store_true",
        help="Bloquea validacion/promocion si falta el bundle supervisado direct-market",
    )
    parser.add_argument(
        "--registry-root",
        default=str(root / "modeling" / "artifacts" / "registry"),
    )
    parser.add_argument(
        "--production-root",
        default=str(root / "modeling" / "artifacts" / "production"),
    )
    parser.add_argument(
        "--min-calibration-gain-vs-elo",
        type=float,
        default=DEFAULT_MIN_CALIBRATION_GAIN_VS_ELO,
    )
    parser.add_argument(
        "--max-calibration-logloss",
        type=float,
        default=DEFAULT_MAX_CALIBRATION_LOGLOSS,
    )
    parser.add_argument(
        "--max-segment-logloss",
        type=float,
        default=DEFAULT_MAX_SEGMENT_LOGLOSS,
    )
    parser.add_argument(
        "--max-segment-ece",
        type=float,
        default=DEFAULT_MAX_SEGMENT_ECE,
    )
    parser.add_argument(
        "--fail-on-segment-warning",
        action="store_true",
        help="Convierte las alertas de segmentos en bloqueo de promocion",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Valida/promociona un candidate ya existente",
    )
    parser.add_argument(
        "--train-market-models",
        action="store_true",
        help="Entrena modelos secundarios de set, marcador y total de juegos",
    )
    parser.add_argument(
        "--market-simulations",
        type=int,
        default=20000,
        help="Simulaciones recomendadas por partido para los mercados",
    )
    parser.add_argument("--market-validation-rows", type=int, default=500)
    parser.add_argument("--market-validation-simulations", type=int, default=1000)
    parser.add_argument("--market-symmetry-rows", type=int, default=100)
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Entrena todos los perfiles v5 y los ordena usando solo tune Log Loss",
    )
    parser.add_argument(
        "--ablation-profiles",
        type=parse_profiles,
        default=list(ABLATION_PROFILES),
        help="Perfiles separados por coma",
    )
    parser.add_argument(
        "--ablation-root",
        default=str(root / "modeling" / "artifacts" / "ablation_v5"),
    )
    parser.add_argument(
        "--ablation-mode",
        choices=["no_market", "with_market"],
        default="no_market",
    )
    args, trainer_args = parser.parse_known_args()
    return args, trainer_args


def trainer_value(arguments: list[str], option: str, default: str) -> str:
    if option in arguments:
        index = arguments.index(option)
        if index + 1 >= len(arguments):
            raise ValueError(f"Falta valor para {option}")
        return arguments[index + 1]
    return default


def resolve_market_training_inputs(arguments: list[str]) -> tuple[Path, Path]:
    dataset_value = ""
    for option in ("--dataset", "--input", "--data", "--input-no-market"):
        if option in arguments:
            index = arguments.index(option)
            if index + 1 >= len(arguments):
                raise ValueError(f"Falta valor para {option}")
            dataset_value = arguments[index + 1]
            break
    if not dataset_value:
        raise RuntimeError("--train-market-models requiere un dataset explicito")
    if "--metadata" not in arguments:
        raise RuntimeError("--train-market-models requiere --metadata")
    index = arguments.index("--metadata")
    if index + 1 >= len(arguments):
        raise ValueError("Falta valor para --metadata")
    dataset = Path(dataset_value).resolve()
    metadata = Path(arguments[index + 1]).resolve()
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    if not metadata.exists():
        raise FileNotFoundError(metadata)
    return dataset, metadata


def install_direct_market_artifacts(source: Path, candidate: Path) -> None:
    """Validate and copy an already trained direct-market bundle into candidate."""
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    missing = sorted(name for name in DIRECT_MARKET_REQUIRED_FILES if not (source / name).exists())
    if missing:
        raise RuntimeError(f"Direct-market candidate incompleto. Faltan: {missing}")
    verify_direct_market_artifacts(source, required=True)
    candidate.mkdir(parents=True, exist_ok=True)
    for name in sorted(DIRECT_MARKET_REQUIRED_FILES):
        shutil.copy2(source / name, candidate / name)
    print(f"[OK] Artefactos direct-market instalados en candidate: {candidate}")


def verify_direct_market_artifacts(candidate: Path, required: bool = False) -> None:
    paths = [candidate / name for name in DIRECT_MARKET_REQUIRED_FILES]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        if required:
            raise RuntimeError(f"Faltan artefactos direct-market obligatorios: {missing}")
        return
    bundle = joblib.load(candidate / DIRECT_MARKET_MODEL_FILE)
    metadata = read_json(candidate / DIRECT_MARKET_METADATA_FILE)
    features = [str(value) for value in bundle.get("features", [])]
    if not features:
        raise RuntimeError("El bundle direct-market no contiene features")
    expected_markets = {"first_set", "win_any_set", *{f"over_{line:.1f}" for line in MARKET_BO3_OVER_LINES}}
    models = bundle.get("models", {})
    calibrators = bundle.get("calibrators", {})
    missing_models = sorted(expected_markets - set(models))
    missing_calibrators = sorted(expected_markets - set(calibrators))
    if missing_models or missing_calibrators:
        raise RuntimeError(
            f"Bundle direct-market incompleto. models={missing_models}; calibrators={missing_calibrators}"
        )
    if metadata.get("test_used_for_selection") is not False:
        raise RuntimeError("Direct-market metadata no garantiza test_used_for_selection=false")
    if metadata.get("features") != features:
        raise RuntimeError("Features inconsistentes entre bundle y metadata direct-market")
    validation = pd.read_parquet(candidate / DIRECT_MARKET_VALIDATION_FILE)
    probability_columns = [column for column in validation if column.startswith("probability_")]
    if not probability_columns:
        raise RuntimeError("Validation direct-market no contiene probabilidades")
    for column in probability_columns:
        values = pd.to_numeric(validation[column], errors="coerce").dropna()
        if not values.between(0.0, 1.0).all():
            raise RuntimeError(f"{column}: probabilidades fuera de [0,1]")
    over_columns = [f"probability_over_{line:.1f}" for line in MARKET_BO3_OVER_LINES]
    available = [column for column in over_columns if column in validation]
    if len(available) >= 2:
        matrix = validation[available].to_numpy(dtype=float)
        valid_rows = np.isfinite(matrix).all(axis=1)
        if valid_rows.any() and (np.diff(matrix[valid_rows], axis=1) > 1e-7).any():
            raise RuntimeError("Las probabilidades direct-market Over no son monotonas")


def verify_market_artifacts(candidate: Path, required: bool = False) -> None:
    paths = [candidate / name for name in (
        MARKET_MODEL_FILE, MARKET_METADATA_FILE, MARKET_VALIDATION_FILE,
    )]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        if required:
            raise RuntimeError(f"Faltan artefactos obligatorios de mercados: {missing}")
        return
    bundle = joblib.load(candidate / MARKET_MODEL_FILE)
    if not bundle.get("features"):
        raise RuntimeError("El artefacto de mercados no contiene features")
    for key in ("player_1_serve_point_model", "player_2_serve_point_model"):
        if bundle.get(key) is None:
            raise RuntimeError(f"El artefacto de mercados no contiene {key}")


def main() -> None:
    args, trainer_args = parse_args()
    root = find_project_root(Path(__file__).resolve().parent)
    trainer = Path(args.trainer).resolve()
    if not args.skip_train and not trainer.exists():
        raise FileNotFoundError(trainer)
    if args.ablation:
        if args.promote:
            raise ValueError("--ablation y --promote no pueden usarse juntos")
        if args.skip_train:
            raise ValueError("--ablation requiere ejecutar el entrenador")
        run_ablation(args, trainer_args, trainer, root)
        return

    mode = trainer_value(trainer_args, "--mode", "no_market")
    output_root = Path(
        trainer_value(
            trainer_args,
            "--output-root",
            str(root / "modeling" / "artifacts" / "candidate_professional_v5"),
        )
    ).resolve()
    candidate = (
        Path(args.candidate_dir).resolve()
        if args.candidate_dir
        else output_root / mode
    )

    if not args.skip_train:
        run_trainer(trainer, root, trainer_args)
    if not candidate.exists():
        raise FileNotFoundError(f"No existe candidate: {candidate}")
    if args.direct_market_dir:
        install_direct_market_artifacts(Path(args.direct_market_dir), candidate)

    print("\nWRAPPER ARGUMENTS")
    print(json.dumps({
        "skip_train": bool(args.skip_train),
        "train_market_models": bool(args.train_market_models),
        "direct_market_dir": args.direct_market_dir,
        "require_direct_market_models": bool(args.require_direct_market_models),
        "market_simulations": int(args.market_simulations),
        "candidate_dir": str(candidate),
        "promote": bool(args.promote),
        "trainer_args": trainer_args,
    }, indent=2, ensure_ascii=False))

    if args.train_market_models:
        market_dataset, market_metadata = resolve_market_training_inputs(trainer_args)
        print("\n" + "=" * 78)
        print("ENTRENAMIENTO DE MODELOS DE MERCADOS")
        print("=" * 78)
        print(f"Dataset: {market_dataset}")
        print(f"Metadata: {market_metadata}")
        print(f"Candidate: {candidate}")
        print(f"Simulations: {args.market_simulations:,}")
        train_market_models(
            dataset_path=market_dataset,
            dataset_metadata_path=market_metadata,
            candidate=candidate,
            simulations=args.market_simulations,
            validation_rows_limit=max(0, args.market_validation_rows),
            validation_simulations=max(250, args.market_validation_simulations),
            symmetry_rows_limit=max(0, args.market_symmetry_rows),
        )
        verify_market_artifacts(candidate, required=True)

    verify_market_artifacts(candidate, required=bool(args.promote))
    verify_direct_market_artifacts(
        candidate, required=bool(args.require_direct_market_models or args.direct_market_dir)
    )
    report = read_json(candidate / "training_report.json")
    metadata = read_json(candidate / "model_metadata.json")
    champion = str(report["champion_selected_on_tune"])
    policy = str(report["probability_policy_selected_on_calibration"]["selected"])
    if metadata.get("champion") != champion:
        raise RuntimeError("Champion inconsistente entre report y metadata")
    if metadata.get("probability_policy") != policy:
        raise RuntimeError("Politica inconsistente entre report y metadata")
    if metadata.get("mode") != report.get("mode"):
        raise RuntimeError("Modo inconsistente entre report y metadata")
    for contract_key in ("dataset_schema_version", "feature_profile"):
        report_value = report.get(contract_key)
        metadata_value = metadata.get(contract_key)
        if not report_value or not metadata_value:
            raise RuntimeError(
                f"Falta {contract_key} en training_report.json o model_metadata.json"
            )
        if report_value != metadata_value:
            raise RuntimeError(
                f"{contract_key} inconsistente: report={report_value!r}; "
                f"metadata={metadata_value!r}"
            )

    model_file = verify_candidate_files(candidate, champion)
    expected_features = [str(value) for value in metadata.get("features", [])]
    if not expected_features:
        raise RuntimeError("model_metadata.json no contiene features")
    verify_model_reload(candidate, champion, model_file, expected_features)
    verify_prediction_artifacts(candidate, champion, policy)
    verify_market_artifacts(candidate, required=bool(args.promote))

    decision = evaluate_promotion(
        candidate,
        report,
        args.min_calibration_gain_vs_elo,
        args.max_calibration_logloss,
        args.max_segment_logloss,
        args.max_segment_ece,
    )
    if args.fail_on_segment_warning and decision.warnings:
        decision = PromotionDecision(
            approved=False,
            reasons=[*decision.reasons, *decision.warnings],
            warnings=decision.warnings,
            champion=decision.champion,
            policy=decision.policy,
            candidate_calibration_logloss=decision.candidate_calibration_logloss,
            elo_surface_calibration_logloss=decision.elo_surface_calibration_logloss,
        )

    print("\nVALIDACION DEL CANDIDATE")
    print(json.dumps(decision.__dict__, indent=2, ensure_ascii=False))
    if not decision.approved:
        raise RuntimeError(
            "Candidate no aprobado para promocion: " + "; ".join(decision.reasons)
        )
    if not args.promote:
        print("\n[OK] Candidate aprobado. No se promociona porque falta --promote.")
        return

    registry_root = Path(args.registry_root).resolve()
    production_root = Path(args.production_root).resolve()
    version = model_version(champion, mode, report)
    registry_dir = copy_candidate_to_registry(
        candidate,
        registry_root,
        version,
        report,
        decision,
        model_file,
    )
    registry_manifest = verify_registry(registry_dir)
    production_dir, backup = promote_registry_atomically(
        registry_dir, production_root, mode
    )
    active_pointer = write_active_pointer(
        production_root, production_dir, registry_manifest
    )

    result = {
        "status": "promoted",
        "model_version": version,
        "champion": champion,
        "mode": mode,
        "probability_policy": policy,
        "registry_directory": str(registry_dir),
        "production_directory": str(production_dir),
        "production_backup": str(backup) if backup else None,
        "active_model_pointer": str(active_pointer),
        "warnings": decision.warnings,
    }
    atomic_json(result, candidate / "promotion_result.json")
    print("\n" + json.dumps(result, indent=2, ensure_ascii=False))
    print("\n[OK] Champion promocionado y active_model.json actualizado.")


if __name__ == "__main__":
    main()
