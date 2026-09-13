from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloConfig:
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
    probability_epsilon: float = 1e-8
    calibration_bin_width: float = 0.05
    metrics_min_year: int | None = None


VALID_SURFACES = {"hard", "clay", "grass", "carpet"}

SURFACE_ALIASES = {
    "hard court": "hard",
    "hardcourt": "hard",
    "outdoor hard": "hard",
    "indoor hard": "hard",
    "hard indoor": "hard",
    "hard outdoor": "hard",
    "clay court": "clay",
    "claycourt": "clay",
    "red clay": "clay",
    "green clay": "clay",
    "grass court": "grass",
    "grasscourt": "grass",
    "synthetic": "carpet",
}

ROUND_ORDER = {
    "PQ": 0,
    "PRE-Q": 0,
    "PREQ": 0,
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "QUAL": 3,
    "R256": 4,
    "R128": 5,
    "R64": 6,
    "R32": 7,
    "R16": 8,
    "QF": 9,
    "SF": 10,
    "F": 11,
    "RR": 7,
    "BR": 12,
}

logger = logging.getLogger("tennis_elo")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def normalize_result(value: Any) -> str | None:
    value = normalize_text(value)
    if value is None:
        return None
    value = value.upper()
    if value in {"W", "WIN", "WINNER", "1"}:
        return "W"
    if value in {"L", "LOSS", "LOSER", "0"}:
        return "L"
    return None


def normalize_surface(value: Any) -> str | None:
    value = normalize_text(value)
    if value is None:
        return None
    value = value.lower()
    value = SURFACE_ALIASES.get(value, value)
    return value if value in VALID_SURFACES else None


def normalize_environment(row: pd.Series) -> str | None:
    for column in ("environment", "indoor_outdoor", "court_environment", "indoor"):
        if column not in row.index or pd.isna(row[column]):
            continue
        value = str(row[column]).strip().lower()
        if value in {"indoor", "i", "yes", "true", "1"}:
            return "indoor"
        if value in {"outdoor", "o", "no", "false", "0"}:
            return "outdoor"

    raw_surface = normalize_text(row.get("surface"))
    if raw_surface:
        raw_surface = raw_surface.lower()
        if "indoor" in raw_surface:
            return "indoor"
        if "outdoor" in raw_surface:
            return "outdoor"
    return None


def safe_int(value: Any, default: int) -> int:
    if pd.isna(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def get_value(row: pd.Series, column: str, default: Any = None) -> Any:
    if column not in row.index or pd.isna(row[column]):
        return default
    return row[column]


def dynamic_k(matches_played: int, config: EloConfig) -> float:
    return config.k_min + (config.k_max - config.k_min) * exp(
        -max(0, matches_played) / config.k_decay_matches
    )


def shared_k(k_a: float, k_b: float) -> float:
    if k_a <= 0 or k_b <= 0:
        raise ValueError("Los factores K deben ser positivos")
    return 2.0 * k_a * k_b / (k_a + k_b)


def expected_probability(rating_a: float, rating_b: float, config: EloConfig) -> float:
    probability = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / config.elo_scale))
    return float(np.clip(
        probability,
        config.probability_epsilon,
        1.0 - config.probability_epsilon,
    ))


def surface_weight(matches_on_surface: int, config: EloConfig) -> float:
    matches_on_surface = max(0, matches_on_surface)
    return config.surface_max_weight * matches_on_surface / (
        matches_on_surface + config.surface_prior_matches
    )


def inactivity_confidence(days: int, config: EloConfig) -> float:
    if days <= config.inactivity_grace_days:
        return 1.0
    value = exp(
        -config.inactivity_decay_rate * (days - config.inactivity_grace_days)
    )
    return max(config.inactivity_min_confidence, value)


def effective_rating(
    overall: float,
    surface_adjustment: float,
    surface_match_count: int,
    inactivity_days: int,
    config: EloConfig,
) -> tuple[float, float, float]:
    weight = surface_weight(surface_match_count, config)
    rating_before_inactivity = overall + weight * surface_adjustment
    confidence = inactivity_confidence(inactivity_days, config)
    rating = config.initial_elo + (
        rating_before_inactivity - config.initial_elo
    ) * confidence
    return rating, weight, confidence


def days_since_last_match(
    player_id: Any,
    match_date: pd.Timestamp,
    last_match_date: dict[Any, pd.Timestamp],
) -> int:
    previous_date = last_match_date.get(player_id)
    if previous_date is None:
        return 0
    return max(0, int((match_date - previous_date).total_seconds() / 86400.0))


def excluded_match(row: pd.Series) -> bool:
    excluded = {
        "W/O", "WO", "WALKOVER", "CANCELLED", "CANCELED",
        "ABANDONED", "NOT PLAYED",
    }
    for column in ("match_status", "status", "result_status"):
        value = normalize_text(row.get(column))
        if value and value.upper() in excluded:
            return True
    return False


def validate_config(config: EloConfig) -> None:
    if config.elo_scale <= 0:
        raise ValueError("elo_scale debe ser positivo")
    if config.k_min <= 0 or config.k_max < config.k_min:
        raise ValueError("La configuración de K no es válida")
    if config.k_decay_matches <= 0:
        raise ValueError("k_decay_matches debe ser positivo")
    if not np.isclose(
        config.overall_update_weight + config.surface_update_weight,
        1.0,
    ):
        raise ValueError("Los pesos overall y surface deben sumar 1")
    if not 0 <= config.surface_max_weight <= 1:
        raise ValueError("surface_max_weight debe estar entre 0 y 1")
    if config.surface_prior_matches <= 0:
        raise ValueError("surface_prior_matches debe ser positivo")
    if not 0 < config.inactivity_min_confidence <= 1:
        raise ValueError("inactivity_min_confidence debe estar entre 0 y 1")


def require_columns(df: pd.DataFrame) -> None:
    required = {
        "master_match_id", "match_date", "player_id", "result", "surface"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")


def atomic_write_parquet(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    df.to_parquet(temporary, index=False)
    os.replace(temporary, output)


def atomic_write_json(data: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=str)
    os.replace(temporary, output)


def binary_log_loss(y: np.ndarray, p: np.ndarray, epsilon: float) -> float:
    p = np.clip(p, epsilon, 1.0 - epsilon)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def metric_record(
    df: pd.DataFrame,
    segment: str,
    value: Any,
    config: EloConfig,
) -> dict[str, Any]:
    y = df["player_a_won"].to_numpy(dtype=float)
    p = df["player_a_win_probability"].to_numpy(dtype=float)
    prediction = (p >= 0.5).astype(int)
    return {
        "segment": segment,
        "segment_value": str(value),
        "matches": len(df),
        "log_loss": binary_log_loss(y, p, config.probability_epsilon),
        "brier_score": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean(prediction == y)),
        "mean_favorite_probability": float(np.mean(np.maximum(p, 1.0 - p))),
    }


def build_metrics(predictions: pd.DataFrame, config: EloConfig) -> pd.DataFrame:
    df = predictions.copy()
    if config.metrics_min_year is not None:
        df = df[df["season"] >= config.metrics_min_year].copy()
    if df.empty:
        return pd.DataFrame()

    records = [metric_record(df, "all", "all", config)]
    for column in (
        "season", "surface", "environment", "competition_type",
        "tournament_level", "best_of",
    ):
        if column not in df.columns:
            continue
        for value, group in df.groupby(column, dropna=False, sort=True):
            records.append(metric_record(group, column, value, config))
    return pd.DataFrame(records)


def build_calibration(predictions: pd.DataFrame, config: EloConfig) -> pd.DataFrame:
    df = predictions.copy()
    if config.metrics_min_year is not None:
        df = df[df["season"] >= config.metrics_min_year].copy()
    if df.empty:
        return pd.DataFrame()

    edges = np.arange(0.0, 1.0 + config.calibration_bin_width, config.calibration_bin_width)
    edges[-1] = 1.0
    df["probability_bin"] = pd.cut(
        df["player_a_win_probability"],
        bins=edges,
        include_lowest=True,
    )
    result = (
        df.groupby("probability_bin", observed=False)
        .agg(
            matches=("master_match_id", "size"),
            mean_predicted_probability=("player_a_win_probability", "mean"),
            actual_win_rate=("player_a_won", "mean"),
        )
        .reset_index()
    )
    result["calibration_error"] = (
        result["actual_win_rate"] - result["mean_predicted_probability"]
    )
    result["absolute_calibration_error"] = result["calibration_error"].abs()
    result["probability_bin"] = result["probability_bin"].astype(str)
    return result


def build_elo_history(
    input_path: Path,
    config: EloConfig,
    progress_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    logger.info("Loading %s", input_path)
    df = pd.read_parquet(input_path)
    require_columns(df)

    input_rows = len(df)
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce", utc=True)
    df["result_normalized"] = df["result"].map(normalize_result)

    invalid_date_rows = int(df["match_date"].isna().sum())
    df = df[df["match_date"].notna()].copy()

    duplicate_rows = int(df.duplicated(
        subset=["master_match_id", "player_id"], keep="first"
    ).sum())
    if duplicate_rows:
        logger.warning("Removing %s duplicate match-player rows", duplicate_rows)
        df = df.drop_duplicates(
            subset=["master_match_id", "player_id"], keep="first"
        )

    if "round" in df.columns:
        df["round_normalized"] = df["round"].map(
            lambda x: normalize_text(x).upper() if normalize_text(x) else None
        )
        df["round_order"] = df["round_normalized"].map(ROUND_ORDER).fillna(999)
    else:
        df["round_order"] = 999

    order = (
        df.groupby("master_match_id", as_index=False, sort=False)
        .agg(match_date=("match_date", "min"), round_order=("round_order", "min"))
        .sort_values(
            ["match_date", "round_order", "master_match_id"],
            kind="mergesort",
        )
    )
    groups = {
        match_id: group
        for match_id, group in df.groupby("master_match_id", sort=False)
    }
    match_ids = order["master_match_id"].tolist()

    overall_elo: defaultdict[Any, float] = defaultdict(lambda: config.initial_elo)
    surface_adjustment: defaultdict[tuple[Any, str], float] = defaultdict(float)
    surface_matches: defaultdict[tuple[Any, str], int] = defaultdict(int)
    career_matches: defaultdict[Any, int] = defaultdict(int)
    last_match_date: dict[Any, pd.Timestamp] = {}

    history_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    skipped = Counter()

    total = len(match_ids)
    logger.info("Matches found: %s", f"{total:,}")

    for index, match_id in enumerate(match_ids, start=1):
        if progress_every > 0 and (index % progress_every == 0 or index == total):
            logger.info(
                "Progress: %s/%s (%.1f%%)",
                f"{index:,}", f"{total:,}", 100.0 * index / total,
            )

        group = groups[match_id]
        if len(group) != 2 or group["player_id"].nunique() != 2:
            skipped["not_two_unique_players"] += 1
            continue

        winners = group[group["result_normalized"] == "W"]
        losers = group[group["result_normalized"] == "L"]
        if len(winners) != 1 or len(losers) != 1:
            skipped["invalid_winner_loser"] += 1
            continue

        winner = winners.iloc[0]
        loser = losers.iloc[0]
        if excluded_match(winner):
            skipped["excluded_match_status"] += 1
            continue

        winner_id = winner["player_id"]
        loser_id = loser["player_id"]
        if pd.isna(winner_id) or pd.isna(loser_id) or winner_id == loser_id:
            skipped["invalid_player_id"] += 1
            continue

        match_date = winner["match_date"]
        surface = normalize_surface(winner["surface"])
        environment = normalize_environment(winner)
        best_of = safe_int(get_value(winner, "best_of", 3), 3)

        common = {
            "master_match_id": match_id,
            "match_date": match_date,
            "season": match_date.year,
            "surface": surface,
            "environment": environment,
            "competition_type": get_value(winner, "competition_type"),
            "tournament_level": get_value(winner, "tournament_level"),
            "tournament_id": get_value(
                winner, "tournament_id", get_value(winner, "competition_id")
            ),
            "round": get_value(winner, "round"),
            "best_of": best_of,
        }

        player_data = {}
        for label, player_id in (("winner", winner_id), ("loser", loser_id)):
            days = days_since_last_match(player_id, match_date, last_match_date)
            overall_before = overall_elo[player_id]
            surface_key = (player_id, surface) if surface else None
            adjustment_before = surface_adjustment[surface_key] if surface_key else 0.0
            surface_count = surface_matches[surface_key] if surface_key else 0
            effective, weight, confidence = effective_rating(
                overall_before,
                adjustment_before,
                surface_count,
                days,
                config,
            )
            player_data[label] = {
                "player_id": player_id,
                "days": days,
                "overall_before": overall_before,
                "surface_key": surface_key,
                "adjustment_before": adjustment_before,
                "surface_count": surface_count,
                "effective": effective,
                "surface_weight": weight,
                "inactivity_confidence": confidence,
                "career_matches": career_matches[player_id],
            }

        win = player_data["winner"]
        lose = player_data["loser"]
        probability_winner = expected_probability(win["effective"], lose["effective"], config)

        winner_k = dynamic_k(win["career_matches"], config)
        loser_k = dynamic_k(lose["career_matches"], config)
        match_k = shared_k(winner_k, loser_k)
        total_delta = match_k * (1.0 - probability_winner)
        overall_delta = config.overall_update_weight * total_delta
        surface_delta = config.surface_update_weight * total_delta

        overall_elo[winner_id] += overall_delta
        overall_elo[loser_id] -= overall_delta

        if surface:
            surface_adjustment[win["surface_key"]] += surface_delta
            surface_adjustment[lose["surface_key"]] -= surface_delta
            surface_matches[win["surface_key"]] += 1
            surface_matches[lose["surface_key"]] += 1

        for label, opponent_label, result, target, sign, individual_k in (
            ("winner", "loser", "W", 1, 1.0, winner_k),
            ("loser", "winner", "L", 0, -1.0, loser_k),
        ):
            player = player_data[label]
            opponent = player_data[opponent_label]
            probability = probability_winner if target == 1 else 1.0 - probability_winner
            adjustment_after = (
                surface_adjustment[player["surface_key"]]
                if player["surface_key"] else 0.0
            )
            history_rows.append({
                **common,
                "player_id": player["player_id"],
                "opponent_player_id": opponent["player_id"],
                "result": result,
                "target": target,
                "career_matches_before": player["career_matches"],
                "surface_matches_before": player["surface_count"],
                "days_since_last_match": player["days"],
                "inactivity_confidence": player["inactivity_confidence"],
                "inactivity_adjusted": player["inactivity_confidence"] < 1.0,
                "individual_k": individual_k,
                "match_k": match_k,
                "elo_overall_before": player["overall_before"],
                "elo_overall_after": overall_elo[player["player_id"]],
                "elo_surface_adjustment_before": player["adjustment_before"],
                "elo_surface_adjustment_after": adjustment_after,
                "surface_weight": player["surface_weight"],
                "elo_effective_before": player["effective"],
                "opponent_elo_effective_before": opponent["effective"],
                "elo_difference_before": player["effective"] - opponent["effective"],
                "expected_win_probability": probability,
                "elo_total_delta": sign * total_delta,
                "elo_overall_delta": sign * overall_delta,
                "elo_surface_delta": sign * surface_delta if surface else 0.0,
            })

        # Orientación A/B alternada de forma determinista por índice cronológico.
        if index % 2 == 0:
            a, b, a_won, probability_a = win, lose, 1, probability_winner
        else:
            a, b, a_won, probability_a = lose, win, 0, 1.0 - probability_winner

        prediction_rows.append({
            **common,
            "player_a_id": a["player_id"],
            "player_b_id": b["player_id"],
            "player_a_won": a_won,
            "player_a_win_probability": probability_a,
            "player_b_win_probability": 1.0 - probability_a,
            "favorite_win_probability": max(probability_a, 1.0 - probability_a),
            "player_a_career_matches_before": a["career_matches"],
            "player_b_career_matches_before": b["career_matches"],
            "career_matches_diff": a["career_matches"] - b["career_matches"],
            "player_a_surface_matches_before": a["surface_count"],
            "player_b_surface_matches_before": b["surface_count"],
            "surface_matches_diff": a["surface_count"] - b["surface_count"],
            "player_a_days_since_last_match": a["days"],
            "player_b_days_since_last_match": b["days"],
            "days_since_last_match_diff": a["days"] - b["days"],
            "player_a_elo_overall_before": a["overall_before"],
            "player_b_elo_overall_before": b["overall_before"],
            "elo_overall_diff": a["overall_before"] - b["overall_before"],
            "player_a_surface_adjustment_before": a["adjustment_before"],
            "player_b_surface_adjustment_before": b["adjustment_before"],
            "surface_adjustment_diff": a["adjustment_before"] - b["adjustment_before"],
            "player_a_surface_weight": a["surface_weight"],
            "player_b_surface_weight": b["surface_weight"],
            "player_a_elo_effective_before": a["effective"],
            "player_b_elo_effective_before": b["effective"],
            "elo_effective_diff": a["effective"] - b["effective"],
            "player_a_inactivity_confidence": a["inactivity_confidence"],
            "player_b_inactivity_confidence": b["inactivity_confidence"],
            "match_k": match_k,
        })

        career_matches[winner_id] += 1
        career_matches[loser_id] += 1
        last_match_date[winner_id] = match_date
        last_match_date[loser_id] = match_date

    history = pd.DataFrame(history_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "input_rows": input_rows,
        "invalid_date_rows": invalid_date_rows,
        "duplicate_player_rows_removed": duplicate_rows,
        "input_distinct_matches": total,
        "processed_matches": len(predictions),
        "output_player_rows": len(history),
        "players": int(history["player_id"].nunique()) if not history.empty else 0,
        "skipped_matches": int(sum(skipped.values())),
        "skipped_reasons": dict(skipped),
        "config": asdict(config),
    }
    return history, predictions, summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tennis Elo history")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/parquet/v2/player_per_match_statistics_atp_qualy_challenger.parquet"
        ),
    )
    parser.add_argument(
        "--output-history",
        type=Path,
        default=Path("data/parquet/v2/elo_history_atp_qualy_challenger.parquet"),
    )
    parser.add_argument(
        "--output-matches",
        type=Path,
        default=Path("data/parquet/v2/elo_match_predictions_atp_qualy_challenger.parquet"),
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path("data/parquet/v2/elo_metrics_atp_qualy_challenger.parquet"),
    )
    parser.add_argument(
        "--output-calibration",
        type=Path,
        default=Path("data/parquet/v2/elo_calibration_atp_qualy_challenger.parquet"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("data/parquet/v2/elo_summary_atp_qualy_challenger.json"),
    )
    parser.add_argument("--initial-elo", type=float, default=1500.0)
    parser.add_argument("--elo-scale", type=float, default=400.0)
    parser.add_argument("--k-min", type=float, default=20.0)
    parser.add_argument("--k-max", type=float, default=80.0)
    parser.add_argument("--k-decay-matches", type=float, default=30.0)
    parser.add_argument("--overall-update-weight", type=float, default=0.70)
    parser.add_argument("--surface-update-weight", type=float, default=0.30)
    parser.add_argument("--surface-max-weight", type=float, default=0.50)
    parser.add_argument("--surface-prior-matches", type=float, default=20.0)
    parser.add_argument("--inactivity-grace-days", type=int, default=90)
    parser.add_argument("--inactivity-decay-rate", type=float, default=0.001)
    parser.add_argument("--inactivity-min-confidence", type=float, default=0.70)
    parser.add_argument("--metrics-min-year", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    configure_logging(args.log_level)

    config = EloConfig(
        initial_elo=args.initial_elo,
        elo_scale=args.elo_scale,
        k_min=args.k_min,
        k_max=args.k_max,
        k_decay_matches=args.k_decay_matches,
        overall_update_weight=args.overall_update_weight,
        surface_update_weight=args.surface_update_weight,
        surface_max_weight=args.surface_max_weight,
        surface_prior_matches=args.surface_prior_matches,
        inactivity_grace_days=args.inactivity_grace_days,
        inactivity_decay_rate=args.inactivity_decay_rate,
        inactivity_min_confidence=args.inactivity_min_confidence,
        metrics_min_year=args.metrics_min_year,
    )
    validate_config(config)

    if not args.input.exists():
        raise FileNotFoundError(f"No existe el input: {args.input}")

    history, predictions, summary = build_elo_history(
        args.input, config, args.progress_every
    )
    metrics = build_metrics(predictions, config)
    calibration = build_calibration(predictions, config)

    if not metrics.empty:
        overall = metrics[metrics["segment"] == "all"].iloc[0]
        summary["overall_metrics"] = {
            "matches": int(overall["matches"]),
            "log_loss": float(overall["log_loss"]),
            "brier_score": float(overall["brier_score"]),
            "accuracy": float(overall["accuracy"]),
        }

    atomic_write_parquet(history, args.output_history)
    atomic_write_parquet(predictions, args.output_matches)
    atomic_write_parquet(metrics, args.output_metrics)
    atomic_write_parquet(calibration, args.output_calibration)
    atomic_write_json(summary, args.output_summary)

    logger.info("=" * 80)
    logger.info("ELO BATCH COMPLETED")
    logger.info("Processed matches: %s", f"{len(predictions):,}")
    logger.info("History rows: %s", f"{len(history):,}")
    logger.info("Players: %s", f"{summary['players']:,}")
    logger.info("Skipped matches: %s", f"{summary['skipped_matches']:,}")
    if "overall_metrics" in summary:
        logger.info("Accuracy: %.4f", summary["overall_metrics"]["accuracy"])
        logger.info("Log Loss: %.4f", summary["overall_metrics"]["log_loss"])
        logger.info("Brier Score: %.4f", summary["overall_metrics"]["brier_score"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Elo batch failed")
        sys.exit(1)
