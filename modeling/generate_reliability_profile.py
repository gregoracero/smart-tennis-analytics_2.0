#!/usr/bin/env python3
"""Genera el perfil de fiabilidad probabilistica del modelo de tenis.

Lee test_predictions.parquet, orienta cada observacion hacia el favorito y crea:

- reliability_profile.parquet
- reliability_report.json
- reliability_global_summary.csv
- reliability_context_summary.csv

El script NO entrena, NO recalibra y NO promociona modelos. Solo analiza las
predicciones out-of-sample ya producidas por el test temporal final.

Ejemplos
--------
Modelo activo, resolviendo la ruta desde active_model.json:

    python modeling/generate_reliability_profile.py

Ruta de artefactos explicita:

    python modeling/generate_reliability_profile.py \
        --artifact-dir modeling/artifacts/production/no_market

Candidate antes de promocionar:

    python modeling/generate_reliability_profile.py \
        --artifact-dir modeling/artifacts/candidate_professional_v5/no_market \
        --model-version candidate_professional_v5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

TARGET_COLUMN = "target_player_1_win"
PROBABILITY_COLUMN = "probability_selected"
DEFAULT_EDGES = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.000001,
)

# Contextos disponibles en el test actual y utiles para Streamlit.
CONTEXT_SCOPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("GLOBAL", ()),
    ("SURFACE", ("surface",)),
    ("COMPETITION", ("competition_type",)),
    ("TOURNEY_LEVEL", ("tourney_level",)),
    ("ROUND", ("round",)),
    ("BEST_OF", ("best_of",)),
    ("SURFACE_COMPETITION", ("surface", "competition_type")),
    (
        "SURFACE_COMPETITION_LEVEL",
        ("surface", "competition_type", "tourney_level"),
    ),
    ("COLD_START", ("cold_start_any_player",)),
    ("SURFACE_COLD_START", ("surface_cold_start_any_player",)),
    ("STATS_RELIABLE", ("stats_reliable_both",)),
    ("SYNTHETIC_PLAYER", ("synthetic_player_any",)),
)

PROFILE_CONTEXT_COLUMNS = (
    "surface",
    "competition_type",
    "tourney_level",
    "round",
    "best_of",
    "cold_start_any_player",
    "surface_cold_start_any_player",
    "stats_reliable_both",
    "synthetic_player_any",
)


@dataclass(frozen=True)
class InputSummary:
    source_path: str
    row_count: int
    valid_row_count: int
    invalid_row_count: int
    test_start_date: str | None
    test_end_date: str | None
    probability_column: str
    target_column: str
    selected_models: list[str]


def find_project_root(start: Path) -> Path:
    """Localiza la raiz mediante las carpetas modeling y data/processed."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (
            (candidate / "modeling").exists()
            and (candidate / "data" / "processed").exists()
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


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)


def resolve_active_artifact_dir(project_root: Path) -> tuple[Path, dict[str, Any]]:
    production_root = project_root / "modeling" / "artifacts" / "production"
    active_path = production_root / "active_model.json"
    active = read_json(active_path)
    relative = active.get("artifact_directory")
    if not relative:
        raise RuntimeError("active_model.json no contiene artifact_directory")
    artifact_dir = (production_root / str(relative)).resolve()
    production_resolved = production_root.resolve()
    if (
        artifact_dir != production_resolved
        and production_resolved not in artifact_dir.parents
    ):
        raise RuntimeError("active_model.json apunta fuera de production")
    return artifact_dir, active


def finite_probability(values: pd.Series, label: str) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce")
    if result.isna().any():
        raise RuntimeError(f"{label}: contiene valores nulos o no numericos")
    if not np.isfinite(result.to_numpy(float)).all():
        raise RuntimeError(f"{label}: contiene valores no finitos")
    if not result.between(0.0, 1.0).all():
        raise RuntimeError(f"{label}: contiene valores fuera de [0, 1]")
    return result.astype(float)


def binary_target(values: pd.Series, label: str) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce")
    if result.isna().any() or not result.isin([0, 1]).all():
        raise RuntimeError(f"{label}: debe contener exclusivamente 0 y 1")
    return result.astype("int8")


def wilson_interval(
    successes: int,
    sample_size: int,
    z_score: float = 1.959963984540054,
) -> tuple[float, float]:
    """Intervalo de Wilson bilateral, por defecto al 95 %."""
    if sample_size <= 0:
        return np.nan, np.nan
    proportion = successes / sample_size
    z2 = z_score**2
    denominator = 1.0 + z2 / sample_size
    centre = proportion + z2 / (2.0 * sample_size)
    adjustment = z_score * math.sqrt(
        proportion * (1.0 - proportion) / sample_size
        + z2 / (4.0 * sample_size**2)
    )
    lower = (centre - adjustment) / denominator
    upper = (centre + adjustment) / denominator
    return max(0.0, float(lower)), min(1.0, float(upper))


def reliability_label(sample_size: int, calibration_gap: float) -> str:
    """Etiqueta inicial de producto basada en muestra y calibracion."""
    absolute_gap = abs(float(calibration_gap))
    if sample_size >= 300 and absolute_gap <= 0.03:
        return "HIGH"
    if sample_size >= 100 and absolute_gap <= 0.05:
        return "MEDIUM"
    if sample_size >= 30:
        return "LOW"
    return "VERY_LOW"


def probability_band_labels(edges: Iterable[float]) -> list[str]:
    values = list(edges)
    labels: list[str] = []
    for left, right in zip(values[:-1], values[1:]):
        right_display = min(right, 1.0)
        labels.append(f"{left:.0%}-{right_display:.0%}")
    return labels


def prepare_test_frame(
    source: Path,
    probability_column: str,
    target_column: str,
    edges: tuple[float, ...],
) -> tuple[pd.DataFrame, InputSummary]:
    frame = pd.read_parquet(source, engine="pyarrow")
    required = {probability_column, target_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"test_predictions incompleto. Faltan: {missing}")

    probability = finite_probability(frame[probability_column], probability_column)
    target = binary_target(frame[target_column], target_column)

    valid = probability.between(0.0, 1.0) & target.isin([0, 1])
    prepared = frame.loc[valid].copy()
    probability = probability.loc[valid]
    target = target.loc[valid]

    prepared["favorite_probability"] = np.maximum(
        probability,
        1.0 - probability,
    )
    prepared["favorite_won"] = np.where(
        probability >= 0.5,
        target,
        1 - target,
    ).astype("int8")
    prepared["favorite_side"] = np.where(
        probability >= 0.5,
        "PLAYER_1",
        "PLAYER_2",
    )

    labels = probability_band_labels(edges)
    prepared["probability_band"] = pd.cut(
        prepared["favorite_probability"],
        bins=list(edges),
        labels=labels,
        right=False,
        include_lowest=True,
    )
    if prepared["probability_band"].isna().any():
        count = int(prepared["probability_band"].isna().sum())
        raise RuntimeError(f"{count} filas no entran en ninguna banda")

    match_dates = (
        pd.to_datetime(prepared["match_date"], errors="coerce")
        if "match_date" in prepared
        else pd.Series(dtype="datetime64[ns]")
    )
    selected_models = (
        sorted(prepared["selected_model"].dropna().astype(str).unique().tolist())
        if "selected_model" in prepared
        else []
    )
    summary = InputSummary(
        source_path=str(source),
        row_count=int(len(frame)),
        valid_row_count=int(len(prepared)),
        invalid_row_count=int(len(frame) - len(prepared)),
        test_start_date=(
            match_dates.min().date().isoformat()
            if not match_dates.dropna().empty
            else None
        ),
        test_end_date=(
            match_dates.max().date().isoformat()
            if not match_dates.dropna().empty
            else None
        ),
        probability_column=probability_column,
        target_column=target_column,
        selected_models=selected_models,
    )
    return prepared, summary


def calculate_group_profile(
    frame: pd.DataFrame,
    profile_scope: str,
    context_columns: tuple[str, ...],
    model_version: str,
    champion: str,
    probability_policy: str,
) -> pd.DataFrame:
    available_context = tuple(c for c in context_columns if c in frame.columns)
    if len(available_context) != len(context_columns):
        return pd.DataFrame()

    group_columns = ["probability_band", *available_context]
    grouped = (
        frame.groupby(group_columns, observed=True, dropna=False)
        .agg(
            sample_size=("favorite_won", "size"),
            favorite_wins=("favorite_won", "sum"),
            mean_predicted_probability=("favorite_probability", "mean"),
            observed_win_rate=("favorite_won", "mean"),
            min_predicted_probability=("favorite_probability", "min"),
            max_predicted_probability=("favorite_probability", "max"),
        )
        .reset_index()
    )
    if grouped.empty:
        return grouped

    grouped["profile_scope"] = profile_scope
    for column in PROFILE_CONTEXT_COLUMNS:
        if column not in grouped.columns:
            grouped[column] = "__ALL__"
        else:
            grouped[column] = grouped[column].astype("string").fillna("__MISSING__")

    grouped["calibration_gap"] = (
        grouped["observed_win_rate"]
        - grouped["mean_predicted_probability"]
    )
    grouped["absolute_calibration_gap"] = grouped["calibration_gap"].abs()

    intervals = [
        wilson_interval(int(wins), int(size))
        for wins, size in zip(grouped["favorite_wins"], grouped["sample_size"])
    ]
    grouped["confidence_lower"] = [value[0] for value in intervals]
    grouped["confidence_upper"] = [value[1] for value in intervals]
    grouped["confidence_width"] = (
        grouped["confidence_upper"] - grouped["confidence_lower"]
    )
    grouped["reliability_label"] = [
        reliability_label(int(size), float(gap))
        for size, gap in zip(grouped["sample_size"], grouped["calibration_gap"])
    ]

    band_text = grouped["probability_band"].astype(str)
    grouped["probability_bin_left"] = (
        band_text.str.split("-").str[0].str.rstrip("%").astype(float) / 100.0
    )
    grouped["probability_bin_right"] = (
        band_text.str.split("-").str[1].str.rstrip("%").astype(float) / 100.0
    )
    grouped["probability_band"] = band_text
    grouped["model_version"] = model_version
    grouped["champion"] = champion
    grouped["probability_policy"] = probability_policy

    return grouped


def build_reliability_profile(
    frame: pd.DataFrame,
    model_version: str,
    champion: str,
    probability_policy: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for scope, context in CONTEXT_SCOPES:
        part = calculate_group_profile(
            frame=frame,
            profile_scope=scope,
            context_columns=context,
            model_version=model_version,
            champion=champion,
            probability_policy=probability_policy,
        )
        if not part.empty:
            parts.append(part)
    if not parts:
        raise RuntimeError("No se pudo generar ningun perfil")

    profile = pd.concat(parts, ignore_index=True)
    ordered_columns = [
        "model_version",
        "champion",
        "probability_policy",
        "profile_scope",
        *PROFILE_CONTEXT_COLUMNS,
        "probability_band",
        "probability_bin_left",
        "probability_bin_right",
        "sample_size",
        "favorite_wins",
        "mean_predicted_probability",
        "observed_win_rate",
        "calibration_gap",
        "absolute_calibration_gap",
        "confidence_lower",
        "confidence_upper",
        "confidence_width",
        "min_predicted_probability",
        "max_predicted_probability",
        "reliability_label",
    ]
    profile = profile[ordered_columns].sort_values(
        [
            "profile_scope",
            "surface",
            "competition_type",
            "tourney_level",
            "round",
            "best_of",
            "probability_bin_left",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    return profile


def expected_calibration_error(global_profile: pd.DataFrame) -> float:
    if global_profile.empty:
        return np.nan
    weights = global_profile["sample_size"] / global_profile["sample_size"].sum()
    return float((weights * global_profile["absolute_calibration_gap"]).sum())


def brier_score(frame: pd.DataFrame) -> float:
    return float(
        np.mean(
            (
                frame["favorite_probability"].to_numpy(float)
                - frame["favorite_won"].to_numpy(float)
            )
            ** 2
        )
    )


def log_loss(frame: pd.DataFrame) -> float:
    probability = np.clip(
        frame["favorite_probability"].to_numpy(float),
        1e-15,
        1.0 - 1e-15,
    )
    target = frame["favorite_won"].to_numpy(float)
    return float(
        -np.mean(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))
    )


def build_report(
    frame: pd.DataFrame,
    profile: pd.DataFrame,
    input_summary: InputSummary,
    model_version: str,
    champion: str,
    probability_policy: str,
) -> dict[str, Any]:
    global_profile = profile.loc[profile["profile_scope"].eq("GLOBAL")].copy()
    global_rows = []
    for row in global_profile.itertuples(index=False):
        global_rows.append(
            {
                "probability_band": row.probability_band,
                "sample_size": int(row.sample_size),
                "favorite_wins": int(row.favorite_wins),
                "mean_predicted_probability": float(row.mean_predicted_probability),
                "observed_win_rate": float(row.observed_win_rate),
                "calibration_gap": float(row.calibration_gap),
                "absolute_calibration_gap": float(row.absolute_calibration_gap),
                "confidence_lower": float(row.confidence_lower),
                "confidence_upper": float(row.confidence_upper),
                "reliability_label": str(row.reliability_label),
            }
        )

    scope_counts = {
        str(scope): int(count)
        for scope, count in profile["profile_scope"].value_counts().items()
    }
    return {
        "schema_version": 1,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "model_version": model_version,
        "champion": champion,
        "probability_policy": probability_policy,
        "input": asdict(input_summary),
        "orientation": {
            "description": "Every row is oriented to the model favorite.",
            "favorite_probability": "max(p_player_1, 1 - p_player_1)",
            "favorite_won": "target if p_player_1 >= 0.5 else 1 - target",
        },
        "global_metrics": {
            "sample_size": int(len(frame)),
            "favorite_accuracy": float(frame["favorite_won"].mean()),
            "mean_favorite_probability": float(frame["favorite_probability"].mean()),
            "ece_weighted": expected_calibration_error(global_profile),
            "brier_score_favorite": brier_score(frame),
            "log_loss_favorite": log_loss(frame),
        },
        "reliability_rules": {
            "HIGH": "sample_size >= 300 and absolute_gap <= 0.03",
            "MEDIUM": "sample_size >= 100 and absolute_gap <= 0.05",
            "LOW": "sample_size >= 30",
            "VERY_LOW": "sample_size < 30",
        },
        "profile_rows_by_scope": scope_counts,
        "global_probability_bands": global_rows,
    }


def validate_outputs(
    profile: pd.DataFrame,
    report: dict[str, Any],
    model_version: str,
    champion: str,
    probability_policy: str,
) -> None:
    required = {
        "model_version",
        "champion",
        "probability_policy",
        "profile_scope",
        "probability_band",
        "sample_size",
        "mean_predicted_probability",
        "observed_win_rate",
        "calibration_gap",
        "confidence_lower",
        "confidence_upper",
        "reliability_label",
    }
    missing = sorted(required - set(profile.columns))
    if missing:
        raise RuntimeError(f"Perfil incompleto. Faltan: {missing}")
    if profile.empty:
        raise RuntimeError("El perfil esta vacio")
    if not profile["model_version"].astype(str).eq(model_version).all():
        raise RuntimeError("model_version inconsistente en el perfil")
    if not profile["champion"].astype(str).eq(champion).all():
        raise RuntimeError("champion inconsistente en el perfil")
    if not profile["probability_policy"].astype(str).eq(probability_policy).all():
        raise RuntimeError("probability_policy inconsistente en el perfil")

    for column in (
        "mean_predicted_probability",
        "observed_win_rate",
        "confidence_lower",
        "confidence_upper",
    ):
        values = pd.to_numeric(profile[column], errors="coerce")
        if values.isna().any() or not values.between(0.0, 1.0).all():
            raise RuntimeError(f"Valores invalidos en {column}")
    if pd.to_numeric(profile["sample_size"], errors="coerce").le(0).any():
        raise RuntimeError("El perfil contiene grupos sin muestra")
    if report.get("model_version") != model_version:
        raise RuntimeError("model_version inconsistente en el informe")
    if report.get("champion") != champion:
        raise RuntimeError("champion inconsistente en el informe")
    if report.get("probability_policy") != probability_policy:
        raise RuntimeError("probability_policy inconsistente en el informe")


def print_global_summary(profile: pd.DataFrame) -> None:
    global_profile = profile.loc[profile["profile_scope"].eq("GLOBAL")].copy()
    display = global_profile[
        [
            "probability_band",
            "sample_size",
            "mean_predicted_probability",
            "observed_win_rate",
            "calibration_gap",
            "confidence_lower",
            "confidence_upper",
            "reliability_label",
        ]
    ].copy()
    for column in (
        "mean_predicted_probability",
        "observed_win_rate",
        "calibration_gap",
        "confidence_lower",
        "confidence_upper",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    print("\nPERFIL GLOBAL DE FIABILIDAD\n")
    print(display.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera fiabilidad a partir del test temporal out-of-sample",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--artifact-dir",
        help="Carpeta que contiene test_predictions.parquet. Por defecto usa active_model.json.",
    )
    parser.add_argument("--input", help="Ruta explicita a test_predictions.parquet")
    parser.add_argument("--output-dir", help="Directorio de salida. Por defecto, artifact-dir")
    parser.add_argument("--model-version")
    parser.add_argument("--champion")
    parser.add_argument("--probability-policy")
    parser.add_argument("--probability-column", default=PROBABILITY_COLUMN)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root(Path(__file__).resolve().parent)

    active: dict[str, Any] = {}
    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir).resolve()
    elif args.input:
        artifact_dir = Path(args.input).resolve().parent
    else:
        artifact_dir, active = resolve_active_artifact_dir(project_root)

    source = Path(args.input).resolve() if args.input else artifact_dir / "test_predictions.parquet"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else artifact_dir
    if not source.exists():
        raise FileNotFoundError(source)

    metadata_path = artifact_dir / "model_metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    model_version = str(
        args.model_version
        or active.get("active_model_version")
        or metadata.get("model_version")
        or artifact_dir.name
    )
    champion = str(
        args.champion
        or active.get("champion")
        or metadata.get("champion")
        or "unknown"
    )
    probability_policy = str(
        args.probability_policy
        or active.get("probability_policy")
        or metadata.get("probability_policy")
        or "unknown"
    )

    frame, input_summary = prepare_test_frame(
        source=source,
        probability_column=args.probability_column,
        target_column=args.target_column,
        edges=DEFAULT_EDGES,
    )
    profile = build_reliability_profile(
        frame=frame,
        model_version=model_version,
        champion=champion,
        probability_policy=probability_policy,
    )
    report = build_report(
        frame=frame,
        profile=profile,
        input_summary=input_summary,
        model_version=model_version,
        champion=champion,
        probability_policy=probability_policy,
    )
    validate_outputs(profile, report, model_version, champion, probability_policy)

    profile_path = output_dir / "reliability_profile.parquet"
    report_path = output_dir / "reliability_report.json"
    global_csv = output_dir / "reliability_global_summary.csv"
    context_csv = output_dir / "reliability_context_summary.csv"

    atomic_parquet(profile, profile_path)
    atomic_json(report, report_path)
    atomic_csv(
        profile.loc[profile["profile_scope"].eq("GLOBAL")],
        global_csv,
    )
    atomic_csv(
        profile.loc[
            profile["profile_scope"].isin(
                ["SURFACE", "COMPETITION", "SURFACE_COMPETITION"]
            )
        ],
        context_csv,
    )

    # Secondary consistency check: reload both primary outputs.
    reloaded_profile = pd.read_parquet(profile_path, engine="pyarrow")
    reloaded_report = read_json(report_path)
    validate_outputs(
        reloaded_profile,
        reloaded_report,
        model_version,
        champion,
        probability_policy,
    )

    print_global_summary(reloaded_profile)
    print("\n[OK] Reliability artifacts generated and reloaded successfully")
    print(f"Input:   {source}")
    print(f"Rows:    {len(frame):,}")
    print(f"Period:  {input_summary.test_start_date} -> {input_summary.test_end_date}")
    print(f"Model:   {model_version}")
    print(f"Champion:{champion}")
    print(f"Policy:  {probability_policy}")
    print(f"Profile: {profile_path}")
    print(f"Report:  {report_path}")
    print(f"Global:  {global_csv}")
    print(f"Context: {context_csv}")


if __name__ == "__main__":
    main()
