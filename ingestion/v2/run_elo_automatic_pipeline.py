from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class CandidateConfig:
    name: str
    initial_elo: float = 1500.0
    elo_scale: float = 400.0
    k_min: float = 20.0
    k_max: float = 80.0
    k_decay_matches: float = 30.0
    overall_update_weight: float = 0.70
    surface_update_weight: float = 0.30
    surface_max_weight: float = 0.50
    surface_prior_matches: float = 20.0
    inactivity_grace_days: int = 90
    inactivity_decay_rate: float = 0.001
    inactivity_min_confidence: float = 0.70


DEFAULT_CONFIG = CandidateConfig(name="default")

logger = logging.getLogger("elo_auto_pipeline")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# COMMAND EXECUTION
# =============================================================================


def safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    )


def output_paths(output_root: Path, run_name: str) -> dict[str, Path]:
    run_directory = output_root / safe_name(run_name)
    return {
        "directory": run_directory,
        "history": run_directory / "elo_history.parquet",
        "matches": run_directory / "elo_match_predictions.parquet",
        "metrics": run_directory / "elo_metrics.parquet",
        "calibration": run_directory / "elo_calibration.parquet",
        "summary": run_directory / "elo_summary.json",
        "stdout": run_directory / "execution.log",
    }


def config_to_cli(config: CandidateConfig) -> list[str]:
    return [
        "--initial-elo", str(config.initial_elo),
        "--elo-scale", str(config.elo_scale),
        "--k-min", str(config.k_min),
        "--k-max", str(config.k_max),
        "--k-decay-matches", str(config.k_decay_matches),
        "--overall-update-weight", str(config.overall_update_weight),
        "--surface-update-weight", str(config.surface_update_weight),
        "--surface-max-weight", str(config.surface_max_weight),
        "--surface-prior-matches", str(config.surface_prior_matches),
        "--inactivity-grace-days", str(config.inactivity_grace_days),
        "--inactivity-decay-rate", str(config.inactivity_decay_rate),
        "--inactivity-min-confidence", str(config.inactivity_min_confidence),
    ]


def is_completed(paths: dict[str, Path]) -> bool:
    required = [
        paths["history"],
        paths["matches"],
        paths["metrics"],
        paths["calibration"],
        paths["summary"],
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def execute_candidate(
    python_executable: str,
    elo_script: Path,
    input_path: Path,
    output_root: Path,
    config: CandidateConfig,
    metrics_min_year: int,
    progress_every: int,
    force: bool,
) -> dict[str, Any]:
    paths = output_paths(output_root, config.name)
    paths["directory"].mkdir(parents=True, exist_ok=True)

    if not force and is_completed(paths):
        logger.info("Reusing completed run: %s", config.name)
        return load_run_result(config, paths, reused=True)

    command = [
        python_executable,
        str(elo_script),
        "--input", str(input_path),
        "--output-history", str(paths["history"]),
        "--output-matches", str(paths["matches"]),
        "--output-metrics", str(paths["metrics"]),
        "--output-calibration", str(paths["calibration"]),
        "--output-summary", str(paths["summary"]),
        "--metrics-min-year", str(metrics_min_year),
        "--progress-every", str(progress_every),
        *config_to_cli(config),
    ]

    logger.info("=" * 80)
    logger.info("Running candidate: %s", config.name)
    logger.info("Command: %s", subprocess.list2cmdline(command))

    started = time.perf_counter()
    output_lines: list[str] = []

    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    ) as process:
        if process.stdout is None:
            raise RuntimeError("Could not capture child process output")

        for line in process.stdout:
            clean_line = line.rstrip()
            output_lines.append(clean_line)
            print(f"[{config.name}] {clean_line}", flush=True)

        return_code = process.wait()

    elapsed_seconds = time.perf_counter() - started
    paths["stdout"].write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    if return_code != 0:
        raise RuntimeError(
            f"Candidate {config.name} failed with return code {return_code}. "
            f"See {paths['stdout']}"
        )

    result = load_run_result(config, paths, reused=False)
    result["elapsed_seconds"] = elapsed_seconds
    return result


def load_run_result(
    config: CandidateConfig,
    paths: dict[str, Path],
    reused: bool,
) -> dict[str, Any]:
    with paths["summary"].open("r", encoding="utf-8") as file:
        summary = json.load(file)

    metrics = pd.read_parquet(paths["metrics"])
    overall = metrics.loc[metrics["segment"].eq("all")]
    if overall.empty:
        raise ValueError(f"No overall metrics found for {config.name}")

    row = overall.iloc[0]
    return {
        "name": config.name,
        "config": config,
        "paths": paths,
        "matches": int(row["matches"]),
        "accuracy": float(row["accuracy"]),
        "log_loss": float(row["log_loss"]),
        "brier_score": float(row["brier_score"]),
        "mean_favorite_probability": float(row["mean_favorite_probability"]),
        "processed_matches": int(summary.get("processed_matches", 0)),
        "skipped_matches": int(summary.get("skipped_matches", 0)),
        "elapsed_seconds": 0.0,
        "reused": reused,
    }


# =============================================================================
# SELECTION AND REPORTING
# =============================================================================


def ranking_key(result: dict[str, Any]) -> tuple[float, float, float]:
    return (
        result["log_loss"],
        result["brier_score"],
        -result["accuracy"],
    )


def select_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("No candidate results to select from")
    return min(results, key=ranking_key)


def result_row(result: dict[str, Any], stage: str) -> dict[str, Any]:
    config: CandidateConfig = result["config"]
    return {
        "stage": stage,
        "name": result["name"],
        "matches": result["matches"],
        "accuracy": result["accuracy"],
        "log_loss": result["log_loss"],
        "brier_score": result["brier_score"],
        "mean_favorite_probability": result["mean_favorite_probability"],
        "processed_matches": result["processed_matches"],
        "skipped_matches": result["skipped_matches"],
        "elapsed_seconds": result["elapsed_seconds"],
        "reused": result["reused"],
        **asdict(config),
    }


def weighted_calibration_error(calibration: pd.DataFrame) -> float:
    calibration = calibration.loc[calibration["matches"].gt(0)].copy()
    if calibration.empty:
        return float("nan")
    weights = calibration["matches"].to_numpy(dtype=float)
    errors = calibration["absolute_calibration_error"].to_numpy(dtype=float)
    return float(np.average(errors, weights=weights))


def build_segment_report(best_result: dict[str, Any]) -> pd.DataFrame:
    metrics = pd.read_parquet(best_result["paths"]["metrics"])
    metrics = metrics.copy()
    metrics["model"] = best_result["name"]
    columns = [
        "model", "segment", "segment_value", "matches", "accuracy",
        "log_loss", "brier_score", "mean_favorite_probability",
    ]
    return metrics[columns].sort_values(
        ["segment", "matches"], ascending=[True, False]
    )


def build_season_report(best_result: dict[str, Any]) -> pd.DataFrame:
    metrics = pd.read_parquet(best_result["paths"]["metrics"])
    seasons = metrics.loc[metrics["segment"].eq("season")].copy()
    if seasons.empty:
        return seasons
    seasons["season"] = pd.to_numeric(seasons["segment_value"], errors="coerce")
    return seasons.sort_values("season")


def compare_with_legacy(
    comparison: pd.DataFrame,
    legacy_metrics_path: Path | None,
) -> pd.DataFrame:
    if legacy_metrics_path is None:
        return comparison
    if not legacy_metrics_path.exists():
        logger.warning("Legacy metrics file does not exist: %s", legacy_metrics_path)
        return comparison

    legacy = pd.read_parquet(legacy_metrics_path)
    legacy_overall = legacy.loc[legacy["segment"].eq("all")]
    if legacy_overall.empty:
        logger.warning("Legacy metrics do not contain segment=all")
        return comparison

    row = legacy_overall.iloc[0]
    legacy_row = {
        "stage": "legacy",
        "name": "legacy_model",
        "matches": int(row["matches"]),
        "accuracy": float(row["accuracy"]),
        "log_loss": float(row["log_loss"]),
        "brier_score": float(row["brier_score"]),
        "mean_favorite_probability": float(row.get("mean_favorite_probability", np.nan)),
    }
    return pd.concat([comparison, pd.DataFrame([legacy_row])], ignore_index=True)


def write_markdown_report(
    output_path: Path,
    comparison: pd.DataFrame,
    best_result: dict[str, Any],
    calibration_error: float,
    metrics_min_year: int,
    elapsed_seconds: float,
) -> None:
    best = best_result
    config: CandidateConfig = best["config"]

    ordered = comparison.sort_values(
        ["log_loss", "brier_score", "accuracy"],
        ascending=[True, True, False],
    ).copy()

    display_columns = [
        "stage", "name", "matches", "accuracy", "log_loss", "brier_score"
    ]
    available = [column for column in display_columns if column in ordered.columns]

    lines = [
        "# Automatic Elo Optimization Report",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Evaluation starts in season: {metrics_min_year}",
        f"Total pipeline runtime: {elapsed_seconds / 60.0:.1f} minutes",
        "",
        "## Selected model",
        "",
        f"- Name: `{best['name']}`",
        f"- Accuracy: `{best['accuracy']:.6f}`",
        f"- Log Loss: `{best['log_loss']:.6f}`",
        f"- Brier Score: `{best['brier_score']:.6f}`",
        f"- Weighted calibration error: `{calibration_error:.6f}`",
        "",
        "## Selected parameters",
        "",
        "```json",
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Candidate comparison",
        "",
        ordered[available].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Selection rule",
        "",
        "Candidates are ranked first by minimum Log Loss, then by minimum "
        "Brier Score, and finally by maximum accuracy.",
        "",
        "## Generated outputs",
        "",
        f"- Final history: `{best['paths']['history']}`",
        f"- Final match predictions: `{best['paths']['matches']}`",
        f"- Final metrics: `{best['paths']['metrics']}`",
        f"- Final calibration: `{best['paths']['calibration']}`",
        f"- Final summary: `{best['paths']['summary']}`",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# CANDIDATE GENERATION
# =============================================================================


def scale_candidates(base: CandidateConfig, mode: str) -> list[CandidateConfig]:
    values = [350.0, 400.0, 450.0, 500.0]
    if mode == "quick":
        values = [400.0, 450.0, 500.0]
    return [
        replace(base, name=f"stage1_scale_{int(value)}", elo_scale=value)
        for value in values
    ]


def k_candidates(base: CandidateConfig, mode: str) -> list[CandidateConfig]:
    combinations = [
        (15.0, 60.0, 30.0),
        (15.0, 80.0, 30.0),
        (20.0, 80.0, 30.0),
        (20.0, 100.0, 30.0),
        (20.0, 80.0, 50.0),
        (25.0, 80.0, 30.0),
    ]
    if mode == "quick":
        combinations = [
            (15.0, 80.0, 30.0),
            (20.0, 80.0, 30.0),
            (20.0, 80.0, 50.0),
        ]
    return [
        replace(
            base,
            name=f"stage2_k_{int(k_min)}_{int(k_max)}_{int(decay)}",
            k_min=k_min,
            k_max=k_max,
            k_decay_matches=decay,
        )
        for k_min, k_max, decay in combinations
    ]


def surface_candidates(base: CandidateConfig, mode: str) -> list[CandidateConfig]:
    combinations = [
        (0.60, 0.40, 0.50, 20.0),
        (0.70, 0.30, 0.35, 20.0),
        (0.70, 0.30, 0.50, 10.0),
        (0.70, 0.30, 0.50, 20.0),
        (0.70, 0.30, 0.50, 30.0),
        (0.70, 0.30, 0.65, 20.0),
        (0.80, 0.20, 0.50, 20.0),
    ]
    if mode == "quick":
        combinations = [
            (0.60, 0.40, 0.50, 20.0),
            (0.70, 0.30, 0.50, 20.0),
            (0.80, 0.20, 0.50, 20.0),
        ]
    return [
        replace(
            base,
            name=(
                f"stage3_surface_{int(overall * 100)}_"
                f"{int(surface * 100)}_{int(max_weight * 100)}_"
                f"{int(prior)}"
            ),
            overall_update_weight=overall,
            surface_update_weight=surface,
            surface_max_weight=max_weight,
            surface_prior_matches=prior,
        )
        for overall, surface, max_weight, prior in combinations
    ]


def inactivity_candidates(base: CandidateConfig, mode: str) -> list[CandidateConfig]:
    combinations = [
        (90, 0.0007, 0.70),
        (90, 0.0010, 0.70),
        (120, 0.0010, 0.70),
        (90, 0.0010, 0.80),
    ]
    if mode == "quick":
        combinations = [
            (90, 0.0010, 0.70),
            (120, 0.0010, 0.70),
        ]
    return [
        replace(
            base,
            name=f"stage4_inactivity_{grace}_{rate}_{int(min_conf * 100)}",
            inactivity_grace_days=grace,
            inactivity_decay_rate=rate,
            inactivity_min_confidence=min_conf,
        )
        for grace, rate, min_conf in combinations
    ]


# =============================================================================
# PIPELINE
# =============================================================================


def run_stage(
    stage_name: str,
    candidates: list[CandidateConfig],
    common_arguments: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    logger.info("#" * 80)
    logger.info("STARTING %s WITH %s CANDIDATES", stage_name, len(candidates))
    logger.info("#" * 80)

    results = []
    for candidate in candidates:
        result = execute_candidate(config=candidate, **common_arguments)
        results.append(result)

    best = select_best(results)
    logger.info(
        "%s winner: %s | accuracy=%.4f | log_loss=%.4f | brier=%.4f",
        stage_name,
        best["name"],
        best["accuracy"],
        best["log_loss"],
        best["brier_score"],
    )
    return results, best


def copy_best_outputs(best_result: dict[str, Any], final_directory: Path) -> None:
    final_directory.mkdir(parents=True, exist_ok=True)
    mapping = {
        "history": "elo_history_best.parquet",
        "matches": "elo_match_predictions_best.parquet",
        "metrics": "elo_metrics_best.parquet",
        "calibration": "elo_calibration_best.parquet",
        "summary": "elo_summary_best.json",
    }
    for key, filename in mapping.items():
        source = best_result["paths"][key]
        destination = final_directory / filename
        destination.write_bytes(source.read_bytes())


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic staged optimization and evaluation for tennis Elo"
    )
    parser.add_argument(
        "--elo-script",
        type=Path,
        default=Path("ingestion/v2/build_elo_history_improved.py"),
        help="Path to the complete improved Elo builder",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/parquet/v2/player_per_match_statistics_atp_qualy_challenger.parquet"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/parquet/v2/elo_automatic_optimization"),
    )
    parser.add_argument("--metrics-min-year", type=int, default=2022)
    parser.add_argument("--progress-every", type=int, default=25000)
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
        help="quick runs fewer candidates; full performs every optimization stage",
    )
    parser.add_argument(
        "--legacy-metrics",
        type=Path,
        default=None,
        help="Optional parquet containing metrics from the old model",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    configure_logging(args.log_level)

    if not args.elo_script.exists():
        raise FileNotFoundError(f"Elo script not found: {args.elo_script}")
    if not args.input.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.input}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    pipeline_started = time.perf_counter()

    common_arguments = {
        "python_executable": args.python_executable,
        "elo_script": args.elo_script,
        "input_path": args.input,
        "output_root": args.output_root / "runs",
        "metrics_min_year": args.metrics_min_year,
        "progress_every": args.progress_every,
        "force": args.force,
    }

    all_rows: list[dict[str, Any]] = []

    stage1_results, stage1_best = run_stage(
        "STAGE 1: ELO SCALE",
        scale_candidates(DEFAULT_CONFIG, args.mode),
        common_arguments,
    )
    all_rows.extend(result_row(result, "scale") for result in stage1_results)

    stage2_base = replace(stage1_best["config"], name="stage2_base")
    stage2_results, stage2_best = run_stage(
        "STAGE 2: K FACTOR",
        k_candidates(stage2_base, args.mode),
        common_arguments,
    )
    all_rows.extend(result_row(result, "k_factor") for result in stage2_results)

    stage3_base = replace(stage2_best["config"], name="stage3_base")
    stage3_results, stage3_best = run_stage(
        "STAGE 3: SURFACE",
        surface_candidates(stage3_base, args.mode),
        common_arguments,
    )
    all_rows.extend(result_row(result, "surface") for result in stage3_results)

    stage4_base = replace(stage3_best["config"], name="stage4_base")
    stage4_results, stage4_best = run_stage(
        "STAGE 4: INACTIVITY",
        inactivity_candidates(stage4_base, args.mode),
        common_arguments,
    )
    all_rows.extend(result_row(result, "inactivity") for result in stage4_results)

    # Select globally among every candidate, not only the last-stage winner.
    every_result = stage1_results + stage2_results + stage3_results + stage4_results
    best_result = select_best(every_result)

    comparison = pd.DataFrame(all_rows)
    comparison = compare_with_legacy(comparison, args.legacy_metrics)
    comparison = comparison.sort_values(
        ["log_loss", "brier_score", "accuracy"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    comparison.insert(0, "rank", np.arange(1, len(comparison) + 1))

    final_directory = args.output_root / "final"
    final_directory.mkdir(parents=True, exist_ok=True)
    copy_best_outputs(best_result, final_directory)

    comparison.to_csv(final_directory / "model_comparison.csv", index=False)
    comparison.to_parquet(final_directory / "model_comparison.parquet", index=False)

    segment_report = build_segment_report(best_result)
    segment_report.to_csv(final_directory / "best_model_segments.csv", index=False)
    segment_report.to_parquet(final_directory / "best_model_segments.parquet", index=False)

    season_report = build_season_report(best_result)
    season_report.to_csv(final_directory / "best_model_seasons.csv", index=False)
    season_report.to_parquet(final_directory / "best_model_seasons.parquet", index=False)

    calibration = pd.read_parquet(best_result["paths"]["calibration"])
    calibration.to_csv(final_directory / "best_model_calibration.csv", index=False)
    calibration_error = weighted_calibration_error(calibration)

    best_config = asdict(best_result["config"])
    best_config["selected_by"] = "minimum log_loss, then brier_score, then maximum accuracy"
    best_config["metrics_min_year"] = args.metrics_min_year
    (final_directory / "best_config.json").write_text(
        json.dumps(best_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    elapsed_seconds = time.perf_counter() - pipeline_started
    write_markdown_report(
        final_directory / "AUTOMATIC_ELO_REPORT.md",
        comparison,
        best_result,
        calibration_error,
        args.metrics_min_year,
        elapsed_seconds,
    )

    logger.info("=" * 80)
    logger.info("AUTOMATIC ELO OPTIMIZATION COMPLETED")
    logger.info("Best model: %s", best_result["name"])
    logger.info("Accuracy: %.6f", best_result["accuracy"])
    logger.info("Log Loss: %.6f", best_result["log_loss"])
    logger.info("Brier Score: %.6f", best_result["brier_score"])
    logger.info("Weighted calibration error: %.6f", calibration_error)
    logger.info("Final outputs: %s", final_directory)
    logger.info("Runtime: %.1f minutes", elapsed_seconds / 60.0)
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logger.error("Pipeline interrupted by user")
        raise SystemExit(130)
    except Exception:
        logger.exception("Automatic Elo pipeline failed")
        raise SystemExit(1)
