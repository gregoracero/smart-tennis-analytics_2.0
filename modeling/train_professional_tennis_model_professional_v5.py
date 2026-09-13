#!/usr/bin/env python3
"""Benchmark profesional temporal para tenis Sackmann v4/v5.

Incluye:
- Baselines: constante de train, Elo global, Elo superficie y ranking calibrado.
- Modelos: CatBoost, XGBoost y regresion logistica.
- Optuna independiente para CatBoost y XGBoost.
- Especialistas CatBoost opcionales por superficie con mezcla parcial.
- Seleccion de champion solo en tune.
- Refit de candidatos ordinarios con train+tune.
- Platt scaling ajustado solo en calibration y politica conservadora.
- Test congelado, bootstrap raw-vs-raw y metricas por segmentos.
- Artefactos atomicos y reproducibles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    import optuna
except ImportError:
    optuna = None

SEED = 42
TARGET = "target_player_1_win"
DATE = "match_date"
SPLIT = "split"
MODEL_NAMES = {"catboost", "xgboost", "logistic"}
MIN_CALIBRATION_LOGLOSS_GAIN = 0.002
MIN_CALIBRATION_BRIER_GAIN = 0.0005
MAX_CALIBRATION_ECE_DEGRADATION = 0.0
PREFERRED_CATEGORICALS = {
    "surface", "tourney_level", "competition_type", "indoor", "best_of",
    "round", "player_1_hand", "player_2_hand", "player_1_ioc", "player_2_ioc",
}
AUDIT_COLUMNS = [
    "processing_sequence", "tourney_id", "tourney_name", "competition_type",
    "match_num", DATE, "surface", "tourney_level", "round", "best_of",
    "source_origin", "player_1_id", "player_2_id", "player_1_name",
    "player_2_name", "cold_start_any_player", "surface_cold_start_any_player",
    "synthetic_player_any", "stats_reliable_both", SPLIT, TARGET,
]


@dataclass
class DatasetParts:
    train: pd.DataFrame
    tune: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    features: list[str]
    categoricals: list[str]
    numericals: list[str]
    metadata: dict[str, Any]
    audit: pd.DataFrame


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "data" / "processed").exists():
            return candidate
    raise FileNotFoundError("No se pudo localizar la raiz del proyecto")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".new")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temp.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".new")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".new.parquet")
    frame.to_parquet(temp, index=False, engine="pyarrow", compression="zstd")
    if len(pd.read_parquet(temp, columns=[TARGET])) != len(frame):
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"Escritura incompleta: {path}")
    temp.replace(path)


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            handle.seek(max(stat.st_size - 1024 * 1024, 0))
            digest.update(handle.read(1024 * 1024))
    return {
        "path": str(path), "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "edge_sha256": digest.hexdigest(),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def expected_calibration_error(y_true: Any, probability: Any, bins: int = 15) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(p, edges, right=True) - 1, 0, bins - 1)
    result = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            result += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(result)


def metric_dict(y_true: Any, probability: Any) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    output = {
        "rows": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(np.int8))),
        "ece_15_bins": expected_calibration_error(y, p),
        "mean_probability": float(p.mean()),
        "target_rate": float(y.mean()),
    }
    output["roc_auc"] = (
        float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    )
    return output


def bootstrap_metric_difference(
    y: np.ndarray,
    champion: np.ndarray,
    challenger: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    if repetitions <= 0:
        return {}
    rng = np.random.default_rng(seed)
    n = len(y)
    values = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sample = rng.integers(0, n, n)
        values[index] = (
            log_loss(y[sample], challenger[sample], labels=[0, 1])
            - log_loss(y[sample], champion[sample], labels=[0, 1])
        )
    return {
        "challenger_minus_champion_logloss_mean": float(values.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "prob_champion_better": float((values > 0).mean()),
    }


def load_features(metadata: dict[str, Any], columns: set[str], mode: str) -> list[str]:
    key = "no_market_feature_columns" if mode == "no_market" else "market_feature_columns"
    features = [str(column) for column in metadata.get(key, [])]
    if not features:
        raise ValueError(f"Metadata sin {key}")
    missing = sorted(set(features) - columns)
    if missing:
        raise ValueError(f"Dataset sin features declaradas: {missing}")
    forbidden = {TARGET, DATE, SPLIT, "processing_sequence", "player_1_id", "player_2_id"}
    leaking = sorted(set(features) & forbidden)
    if leaking:
        raise ValueError(f"Metadata incluye columnas no predictoras: {leaking}")
    return features


def infer_categoricals(frame: pd.DataFrame, features: list[str]) -> list[str]:
    return [
        column for column in features
        if column in PREFERRED_CATEGORICALS
        or pd.api.types.is_string_dtype(frame[column].dtype)
        or isinstance(frame[column].dtype, pd.CategoricalDtype)
    ]


def sanitize_categoricals(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = (
            result[column].astype("string").fillna("__MISSING__").astype(str)
        )
    return result


def audit_training_features(
    train: pd.DataFrame,
    features: list[str],
    categoricals: list[str],
    maximum_missing_fraction: float,
    keep_constant_features: bool,
) -> tuple[list[str], list[str], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    kept: list[str] = []
    for column in features:
        missing = float(train[column].isna().mean())
        unique = int(train[column].nunique(dropna=True))
        reason = ""
        if missing > maximum_missing_fraction:
            reason = "missing_fraction"
        elif unique <= 1 and not keep_constant_features:
            reason = "constant"
        elif column not in categoricals and not pd.api.types.is_numeric_dtype(train[column]):
            reason = "unexpected_non_numeric"
        if not reason:
            kept.append(column)
        rows.append({
            "feature": column, "dtype": str(train[column].dtype),
            "missing_fraction_train": missing, "unique_train": unique,
            "categorical": column in categoricals, "kept": not bool(reason),
            "drop_reason": reason,
        })
    if not kept:
        raise ValueError("No quedan features entrenables")
    return kept, [c for c in categoricals if c in kept], pd.DataFrame(rows)


def prepare_dataset(
    path: Path,
    metadata_path: Path,
    mode: str,
    calibration_fraction: float,
    maximum_missing_fraction: float,
    keep_constant_features: bool,
) -> DatasetParts:
    metadata = load_json(metadata_path)
    columns = set(pq.ParquetFile(path).schema_arrow.names)
    required = {TARGET, DATE, SPLIT}
    if required - columns:
        raise ValueError(f"Faltan columnas esenciales: {sorted(required - columns)}")
    features = load_features(metadata, columns, mode)
    audit_columns = [column for column in AUDIT_COLUMNS if column in columns]
    selected = list(dict.fromkeys(audit_columns + features + [TARGET, DATE, SPLIT]))
    frame = pd.read_parquet(path, columns=selected, engine="pyarrow")
    frame[DATE] = pd.to_datetime(frame[DATE], errors="coerce")
    if frame[DATE].isna().any() or frame[TARGET].isna().any():
        raise ValueError("Fechas o target nulos")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target no binario")
    categoricals = infer_categoricals(frame, features)
    frame = sanitize_categoricals(frame, categoricals)
    train = frame.loc[frame[SPLIT].eq("train")].sort_values(DATE).reset_index(drop=True)
    validation = frame.loc[frame[SPLIT].eq("validation")].sort_values(DATE).reset_index(drop=True)
    test = frame.loc[frame[SPLIT].eq("test")].sort_values(DATE).reset_index(drop=True)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Split vacio")
    cut = max(1, min(len(validation) - 1, int(round(len(validation) * (1 - calibration_fraction)))))
    tune = validation.iloc[:cut].copy()
    calibration = validation.iloc[cut:].copy()
    if tune[DATE].max() > calibration[DATE].min():
        raise RuntimeError("Tune y calibration solapan temporalmente")
    features, categoricals, audit = audit_training_features(
        train, features, categoricals, maximum_missing_fraction,
        keep_constant_features,
    )
    numericals = [column for column in features if column not in categoricals]
    return DatasetParts(
        train, tune, calibration, test, features, categoricals,
        numericals, metadata, audit,
    )


def recency_weights(dates: pd.Series, half_life_days: float) -> np.ndarray | None:
    if half_life_days <= 0:
        return None
    parsed = pd.to_datetime(dates)
    age = (parsed.max() - parsed).dt.days.clip(lower=0).to_numpy(float)
    weights = np.power(0.5, age / half_life_days)
    weights /= weights.mean()
    return weights.astype("float32")


def build_sklearn_preprocessor(parts: DatasetParts, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
    ])
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), parts.numericals),
        ("cat", categorical_pipe, parts.categoricals),
    ], remainder="drop")


def catboost_parameters(seed: int, threads: int) -> dict[str, Any]:
    return {
        "loss_function": "Logloss", "eval_metric": "Logloss",
        "iterations": 2200, "learning_rate": 0.03, "depth": 6,
        "l2_leaf_reg": 10.0, "random_strength": 0.5,
        "bagging_temperature": 0.35, "border_count": 96, "rsm": 0.8,
        "bootstrap_type": "Bayesian", "random_seed": seed,
        "thread_count": threads, "allow_writing_files": False, "verbose": 200,
    }


def tune_catboost(
    parts: DatasetParts,
    trials: int,
    seed: int,
    threads: int,
    max_rows: int,
    half_life: float,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    base = catboost_parameters(seed, threads)
    train = (
        parts.train.iloc[-max_rows:].copy()
        if max_rows > 0 and len(parts.train) > max_rows else parts.train
    )
    train_pool = Pool(
        train[parts.features], train[TARGET].astype("int8"),
        weight=recency_weights(train[DATE], half_life),
        cat_features=parts.categoricals, feature_names=parts.features,
    )
    tune_pool = Pool(
        parts.tune[parts.features], parts.tune[TARGET].astype("int8"),
        cat_features=parts.categoricals, feature_names=parts.features,
    )
    if trials <= 0 or optuna is None:
        model = CatBoostClassifier(**base)
        model.fit(train_pool, eval_set=tune_pool, early_stopping_rounds=150)
        return base, int(model.get_best_iteration() + 1), {
            "engine": "default_early_stopping",
            "best_value": float(model.get_best_score()["validation"]["Logloss"]),
        }

    def objective(trial: Any) -> float:
        params = {
            **base, "iterations": 2600,
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2, 30, log=True),
            "random_strength": trial.suggest_float("random_strength", 0, 1.5),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1.5),
            "rsm": trial.suggest_float("rsm", 0.55, 1.0),
        }
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=tune_pool, early_stopping_rounds=150, verbose=False)
        trial.set_user_attr("best_iteration", int(model.get_best_iteration() + 1))
        return float(model.get_best_score()["validation"]["Logloss"])

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=trials)
    params = {**base, **study.best_params}
    return params, int(study.best_trial.user_attrs["best_iteration"]), {
        "engine": "optuna", "trials": trials,
        "best_value": float(study.best_value),
    }


def fit_catboost(
    parts: DatasetParts,
    params: dict[str, Any],
    iterations: int,
    half_life: float,
    include_tune: bool,
) -> CatBoostClassifier:
    fit = pd.concat([parts.train, parts.tune], ignore_index=True) if include_tune else parts.train.copy()
    print(f"Entrenando CatBoost con {len(fit):,} filas...", flush=True)
    model = CatBoostClassifier(**{**params, "iterations": max(100, iterations), "verbose": 200})
    pool = Pool(
        fit[parts.features], fit[TARGET].astype("int8"),
        weight=recency_weights(fit[DATE], half_life),
        cat_features=parts.categoricals, feature_names=parts.features,
    )
    model.fit(pool)
    return model


def xgboost_default_parameters(seed: int, threads: int) -> dict[str, Any]:
    return {
        "n_estimators": 1400, "learning_rate": 0.025, "max_depth": 6,
        "min_child_weight": 5.0, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.05, "reg_lambda": 8.0, "gamma": 0.0,
        "objective": "binary:logistic", "eval_metric": "logloss",
        "tree_method": "hist", "n_jobs": threads, "random_state": seed,
    }


def transformed_frames_for_xgboost(
    parts: DatasetParts,
    train: pd.DataFrame,
    tune: pd.DataFrame,
) -> tuple[ColumnTransformer, Any, Any]:
    preprocessor = build_sklearn_preprocessor(parts, False)
    x_train = preprocessor.fit_transform(train[parts.features])
    x_tune = preprocessor.transform(tune[parts.features])
    return preprocessor, x_train, x_tune


def tune_xgboost(
    parts: DatasetParts,
    trials: int,
    seed: int,
    threads: int,
    max_rows: int,
    half_life: float,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    if XGBClassifier is None:
        raise ImportError("xgboost no esta instalado")
    base = xgboost_default_parameters(seed, threads)
    train = (
        parts.train.iloc[-max_rows:].copy()
        if max_rows > 0 and len(parts.train) > max_rows else parts.train
    )
    preprocessor, x_train, x_tune = transformed_frames_for_xgboost(parts, train, parts.tune)
    y_train = train[TARGET].astype("int8").to_numpy()
    y_tune = parts.tune[TARGET].astype("int8").to_numpy()
    weights = recency_weights(train[DATE], half_life)

    def fit_and_score(params: dict[str, Any]) -> tuple[float, int]:
        model = XGBClassifier(**params)
        model.fit(
            x_train, y_train, sample_weight=weights,
            eval_set=[(x_tune, y_tune)], verbose=False,
        )
        probability = model.predict_proba(x_tune)[:, 1]
        return float(log_loss(y_tune, probability, labels=[0, 1])), int(params["n_estimators"])

    if trials <= 0 or optuna is None:
        value, iterations = fit_and_score(base)
        return base, iterations, {"engine": "fixed_parameters", "best_value": value}

    def objective(trial: Any) -> float:
        params = {
            **base,
            "n_estimators": trial.suggest_int("n_estimators", 500, 2200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 30.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        }
        value, _ = fit_and_score(params)
        return value

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed + 101)
    )
    study.optimize(objective, n_trials=trials)
    params = {**base, **study.best_params}
    return params, int(params["n_estimators"]), {
        "engine": "optuna", "trials": trials,
        "best_value": float(study.best_value),
        "preprocessor_fit_rows": len(train),
    }


def fit_xgboost(
    parts: DatasetParts,
    params: dict[str, Any],
    half_life: float,
    include_tune: bool,
) -> Pipeline:
    fit = pd.concat([parts.train, parts.tune], ignore_index=True) if include_tune else parts.train.copy()
    print(f"Entrenando XGBoost con {len(fit):,} filas...", flush=True)
    model = Pipeline([
        ("preprocess", build_sklearn_preprocessor(parts, False)),
        ("model", XGBClassifier(**params)),
    ])
    model.fit(
        fit[parts.features], fit[TARGET].astype("int8"),
        model__sample_weight=recency_weights(fit[DATE], half_life),
    )
    return model


def fit_logistic(parts: DatasetParts, half_life: float, include_tune: bool) -> Pipeline:
    fit = pd.concat([parts.train, parts.tune], ignore_index=True) if include_tune else parts.train.copy()
    print(f"Entrenando regresion logistica con {len(fit):,} filas...", flush=True)
    model = Pipeline([
        ("preprocess", build_sklearn_preprocessor(parts, True)),
        ("model", LogisticRegression(
            C=0.5, max_iter=500, solver="lbfgs", tol=1e-4,
            random_state=SEED,
        )),
    ])
    model.fit(
        fit[parts.features], fit[TARGET].astype("int8"),
        model__sample_weight=recency_weights(fit[DATE], half_life),
    )
    return model


def predict_model(model: Any, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(frame[features])[:, 1]


def predict_catboost(model: CatBoostClassifier, frame: pd.DataFrame, parts: DatasetParts) -> np.ndarray:
    return model.predict_proba(frame[parts.features])[:, 1]


def fit_ranking_baseline(parts: DatasetParts, include_tune: bool) -> Pipeline | None:
    rank_features = [c for c in ("diff_rank", "diff_rank_points") if c in parts.train]
    if not rank_features:
        return None
    fit = pd.concat([parts.train, parts.tune], ignore_index=True) if include_tune else parts.train.copy()
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=SEED)),
    ])
    pipeline.fit(fit[rank_features], fit[TARGET].astype("int8"))
    pipeline.rank_features_ = rank_features
    return pipeline


def ranking_prediction(model: Pipeline | None, frame: pd.DataFrame) -> np.ndarray | None:
    if model is None:
        return None
    return model.predict_proba(frame[model.rank_features_])[:, 1]


def baseline_probabilities(
    frame: pd.DataFrame,
    train_rate: float,
    ranking_model: Pipeline | None,
) -> dict[str, np.ndarray]:
    result = {"constant_train_rate": np.full(len(frame), train_rate, dtype=float)}
    for name, column in (
        ("elo_global", "diff_elo_before"),
        ("elo_surface", "diff_surface_elo_before"),
    ):
        if column in frame:
            difference = pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy(float)
            result[name] = 1.0 / (1.0 + np.power(10.0, -difference / 400.0))
    ranking = ranking_prediction(ranking_model, frame)
    if ranking is not None:
        result["ranking_fitted_train"] = ranking
    for column in (
        "market_avg_p1_prob_no_vig", "market_ps_p1_prob_no_vig",
        "market_b365_p1_prob_no_vig",
    ):
        if column in frame:
            probability = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            if np.isfinite(probability).all():
                result[column] = probability
                break
    return result


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e6, solver="lbfgs")

    def fit(self, probability: Any, y: Any) -> "PlattCalibrator":
        p = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
        self.model.fit(np.log(p / (1 - p)).reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, probability: Any) -> np.ndarray:
        p = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
        return self.model.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


def choose_calibration(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray) -> dict[str, Any]:
    raw_metrics = metric_dict(y, raw)
    calibrated_metrics = metric_dict(y, calibrated)
    logloss_gain = raw_metrics["log_loss"] - calibrated_metrics["log_loss"]
    brier_gain = raw_metrics["brier_score"] - calibrated_metrics["brier_score"]
    ece_change = calibrated_metrics["ece_15_bins"] - raw_metrics["ece_15_bins"]
    selected = (
        "calibrated"
        if logloss_gain >= MIN_CALIBRATION_LOGLOSS_GAIN
        and brier_gain >= MIN_CALIBRATION_BRIER_GAIN
        and ece_change <= MAX_CALIBRATION_ECE_DEGRADATION
        else "raw"
    )
    return {
        "selected": selected, "raw": raw_metrics,
        "calibrated": calibrated_metrics, "logloss_gain": float(logloss_gain),
        "brier_gain": float(brier_gain), "ece_change": float(ece_change),
        "minimum_logloss_gain": MIN_CALIBRATION_LOGLOSS_GAIN,
        "minimum_brier_gain": MIN_CALIBRATION_BRIER_GAIN,
        "maximum_ece_degradation": MAX_CALIBRATION_ECE_DEGRADATION,
        "test_not_used_for_selection": True,
    }


def fit_surface_specialists(
    parts: DatasetParts,
    global_model: CatBoostClassifier,
    params: dict[str, Any],
    iterations: int,
    minimum_rows: int,
    alphas: list[float],
) -> tuple[dict[str, CatBoostClassifier], dict[str, float], dict[str, Any]]:
    specialists: dict[str, CatBoostClassifier] = {}
    chosen: dict[str, float] = {}
    audit: dict[str, Any] = {}
    global_tune = predict_catboost(global_model, parts.tune, parts)
    for surface in sorted(parts.train["surface"].dropna().astype(str).unique()):
        fit = parts.train.loc[parts.train.surface.astype(str).eq(surface)]
        tune = parts.tune.loc[parts.tune.surface.astype(str).eq(surface)]
        if len(fit) < minimum_rows or len(tune) < 200:
            audit[surface] = {"trained": False, "fit_rows": len(fit), "tune_rows": len(tune)}
            continue
        model = CatBoostClassifier(**{
            **params, "iterations": min(max(iterations, 200), 1400), "verbose": False,
        })
        model.fit(Pool(
            fit[parts.features], fit[TARGET].astype("int8"),
            cat_features=parts.categoricals, feature_names=parts.features,
        ))
        specialist = model.predict_proba(tune[parts.features])[:, 1]
        mask = parts.tune.surface.astype(str).eq(surface).to_numpy()
        scores = {
            alpha: log_loss(
                tune[TARGET], alpha * specialist + (1 - alpha) * global_tune[mask],
                labels=[0, 1],
            )
            for alpha in alphas
        }
        alpha = min(scores, key=scores.get)
        specialists[surface] = model
        chosen[surface] = float(alpha)
        audit[surface] = {
            "trained": True, "fit_rows": len(fit), "tune_rows": len(tune),
            "alpha": alpha, "tune_logloss": scores[alpha],
        }
    return specialists, chosen, audit


def predict_surface_blend(
    frame: pd.DataFrame,
    global_probability: np.ndarray,
    specialists: dict[str, CatBoostClassifier],
    alphas: dict[str, float],
    parts: DatasetParts,
) -> np.ndarray:
    output = global_probability.copy()
    for surface, model in specialists.items():
        mask = frame.surface.astype(str).eq(surface).to_numpy()
        if mask.any():
            probability = model.predict_proba(frame.loc[mask, parts.features])[:, 1]
            alpha = alphas[surface]
            output[mask] = alpha * probability + (1 - alpha) * output[mask]
    return output


def subgroup_metrics(frame: pd.DataFrame, probability: np.ndarray, minimum_rows: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for field in (
        "surface", "competition_type", "source_origin", "cold_start_any_player",
        "surface_cold_start_any_player", "synthetic_player_any", "stats_reliable_both",
    ):
        if field not in frame:
            continue
        for value, indices in frame.groupby(field, dropna=False).groups.items():
            index = np.asarray(list(indices), dtype=int)
            if len(index) >= minimum_rows:
                rows.append({
                    "segment": field, "value": str(value),
                    **metric_dict(frame.iloc[index][TARGET], probability[index]),
                })
    return pd.DataFrame(rows)


def save_predictions(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    selected_name: str,
    path: Path,
) -> None:
    columns = [c for c in AUDIT_COLUMNS if c in frame and c != SPLIT]
    output = frame[columns].copy()
    for name, probability in predictions.items():
        output[f"probability_{name}"] = np.asarray(probability, dtype="float32")
    output["selected_model"] = selected_name
    output["probability_selected"] = np.asarray(predictions[selected_name], dtype="float32")
    output["prediction_selected"] = (output.probability_selected >= 0.5).astype("int8")
    atomic_parquet(output, path)


def parse_args() -> argparse.Namespace:
    root = find_project_root(Path(__file__).resolve().parent)
    processed = root / "data" / "processed"
    parser = argparse.ArgumentParser(description="Benchmark profesional temporal de tenis")
    parser.add_argument("--mode", choices=["no_market", "with_market"], default="no_market")
    parser.add_argument("--dataset")
    parser.add_argument("--metadata", default=str(processed / "tennis_ml_dataset_metadata_sackmann_v4.json"))
    parser.add_argument("--output-root", default=str(root / "modeling" / "artifacts" / "candidate_professional_v5"))
    parser.add_argument("--models", default="catboost,xgboost,logistic")
    parser.add_argument("--trials", type=int, default=0, help="Alias para trials de ambos boosters")
    parser.add_argument("--catboost-trials", type=int)
    parser.add_argument("--xgboost-trials", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tuning-max-rows", type=int, default=120000)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--recency-half-life-days", type=float, default=0.0)
    parser.add_argument("--maximum-missing-fraction", type=float, default=0.995)
    parser.add_argument("--keep-constant-features", action="store_true")
    parser.add_argument("--surface-specialists", action="store_true")
    parser.add_argument("--surface-min-train-rows", type=int, default=5000)
    parser.add_argument("--surface-alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--minimum-subgroup-rows", type=int, default=50)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)
    started = time.time()
    root = find_project_root(Path(__file__).resolve().parent)
    default_dataset = root / "data" / "processed" / (
        "tennis_ml_no_market.sackmann_v4.parquet"
        if args.mode == "no_market"
        else "tennis_ml_with_market.sackmann_v4.parquet"
    )
    dataset = Path(args.dataset).resolve() if args.dataset else default_dataset.resolve()
    metadata_path = Path(args.metadata).resolve()
    output = Path(args.output_root).resolve() / args.mode
    output.mkdir(parents=True, exist_ok=True)
    requested = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(requested) - MODEL_NAMES)
    if unknown:
        raise ValueError(f"Modelos desconocidos: {unknown}")
    cat_trials = args.trials if args.catboost_trials is None else args.catboost_trials
    xgb_trials = args.trials if args.xgboost_trials is None else args.xgboost_trials
    if (cat_trials > 0 or xgb_trials > 0) and optuna is None:
        raise ImportError("Se han solicitado trials pero optuna no esta instalado")

    parts = prepare_dataset(
        dataset, metadata_path, args.mode, args.calibration_fraction,
        args.maximum_missing_fraction, args.keep_constant_features,
    )
    atomic_csv(parts.audit, output / "feature_audit.csv")

    ranking_candidate = fit_ranking_baseline(parts, include_tune=False)
    ranking_final = fit_ranking_baseline(parts, include_tune=True)
    train_rate = float(parts.train[TARGET].mean())
    tune_predictions = baseline_probabilities(parts.tune, train_rate, ranking_candidate)
    calibration_predictions = baseline_probabilities(parts.calibration, train_rate, ranking_final)
    test_predictions = baseline_probabilities(parts.test, train_rate, ranking_final)

    candidate_models: dict[str, Any] = {}
    tuning_info: dict[str, Any] = {}
    cat_spec: tuple[dict[str, Any], int] | None = None
    xgb_spec: tuple[dict[str, Any], int] | None = None

    # Stage A: cada candidato se ajusta solo con train y se selecciona en tune.
    if "catboost" in requested:
        params, iterations, info = tune_catboost(
            parts, cat_trials, args.seed, args.threads,
            args.tuning_max_rows, args.recency_half_life_days,
        )
        model = fit_catboost(
            parts, params, iterations, args.recency_half_life_days,
            include_tune=False,
        )
        candidate_models["catboost"] = model
        cat_spec = (params, iterations)
        tuning_info["catboost"] = {"parameters": params, "iterations": iterations, **info}
        tune_predictions["catboost"] = predict_catboost(model, parts.tune, parts)
        if args.surface_specialists and "surface" in parts.features:
            specialists, alphas, audit = fit_surface_specialists(
                parts, model, params, iterations, args.surface_min_train_rows,
                [float(value) for value in args.surface_alphas.split(",")],
            )
            candidate_models["catboost_surface_blend"] = (model, specialists, alphas)
            tuning_info["surface_specialists"] = audit
            tune_predictions["catboost_surface_blend"] = predict_surface_blend(
                parts.tune, tune_predictions["catboost"], specialists, alphas, parts,
            )

    if "xgboost" in requested:
        if XGBClassifier is None:
            warnings.warn("xgboost no instalado; challenger omitido")
        else:
            params, iterations, info = tune_xgboost(
                parts, xgb_trials, args.seed, args.threads,
                args.tuning_max_rows, args.recency_half_life_days,
            )
            model = fit_xgboost(
                parts, params, args.recency_half_life_days, include_tune=False,
            )
            candidate_models["xgboost"] = model
            xgb_spec = (params, iterations)
            tuning_info["xgboost"] = {"parameters": params, "iterations": iterations, **info}
            tune_predictions["xgboost"] = predict_model(model, parts.tune, parts.features)

    if "logistic" in requested:
        model = fit_logistic(parts, args.recency_half_life_days, include_tune=False)
        candidate_models["logistic"] = model
        tune_predictions["logistic"] = predict_model(model, parts.tune, parts.features)

    if not candidate_models:
        raise RuntimeError("No se entreno ningun modelo candidato")
    tune_metrics = {
        name: metric_dict(parts.tune[TARGET], probability)
        for name, probability in tune_predictions.items()
    }
    champion = min(candidate_models, key=lambda name: tune_metrics[name]["log_loss"])

    # Stage B: candidatos ordinarios se reentrenan con train+tune.
    if "catboost" in candidate_models:
        assert cat_spec is not None
        params, iterations = cat_spec
        model = fit_catboost(
            parts, params, iterations, args.recency_half_life_days,
            include_tune=True,
        )
        model.save_model(str(output / "catboost_model.cbm"))
        calibration_predictions["catboost"] = predict_catboost(model, parts.calibration, parts)
        test_predictions["catboost"] = predict_catboost(model, parts.test, parts)

    if "xgboost" in candidate_models:
        assert xgb_spec is not None
        params, _ = xgb_spec
        model = fit_xgboost(
            parts, params, args.recency_half_life_days, include_tune=True,
        )
        joblib.dump(model, output / "xgboost_pipeline.joblib")
        calibration_predictions["xgboost"] = predict_model(model, parts.calibration, parts.features)
        test_predictions["xgboost"] = predict_model(model, parts.test, parts.features)

    if "logistic" in candidate_models:
        model = fit_logistic(parts, args.recency_half_life_days, include_tune=True)
        joblib.dump(model, output / "logistic_pipeline.joblib")
        calibration_predictions["logistic"] = predict_model(model, parts.calibration, parts.features)
        test_predictions["logistic"] = predict_model(model, parts.test, parts.features)

    # El blend de superficie conserva los modelos train-only para que alpha no vea calibration.
    if champion == "catboost_surface_blend":
        global_model, specialists, alphas = candidate_models[champion]
        calibration_predictions[champion] = predict_surface_blend(
            parts.calibration,
            predict_catboost(global_model, parts.calibration, parts),
            specialists, alphas, parts,
        )
        test_predictions[champion] = predict_surface_blend(
            parts.test, predict_catboost(global_model, parts.test, parts),
            specialists, alphas, parts,
        )
        for surface, model in specialists.items():
            model.save_model(str(output / f"catboost_surface_{surface.lower()}.cbm"))
        joblib.dump(alphas, output / "surface_blend_weights.joblib")

    calibrator = PlattCalibrator().fit(
        calibration_predictions[champion], parts.calibration[TARGET]
    )
    calibrated_calibration = calibrator.predict(calibration_predictions[champion])
    policy = choose_calibration(
        parts.calibration[TARGET].to_numpy(),
        calibration_predictions[champion], calibrated_calibration,
    )
    joblib.dump(calibrator, output / "platt_calibrator.joblib")
    calibrated_test = calibrator.predict(test_predictions[champion])
    selected_test = (
        calibrated_test if policy["selected"] == "calibrated"
        else test_predictions[champion]
    )
    selected_name = f"{champion}_selected"
    test_predictions[selected_name] = selected_test

    calibration_metrics = {
        name: metric_dict(parts.calibration[TARGET], probability)
        for name, probability in calibration_predictions.items()
    }
    test_metrics = {
        name: metric_dict(parts.test[TARGET], probability)
        for name, probability in test_predictions.items()
    }

    segment = subgroup_metrics(
        parts.test, selected_test, args.minimum_subgroup_rows
    )
    atomic_csv(segment, output / "test_segment_metrics.csv")
    save_predictions(parts.test, test_predictions, selected_name, output / "test_predictions.parquet")
    calibration_selected = (
        calibrated_calibration if policy["selected"] == "calibrated"
        else calibration_predictions[champion]
    )
    save_predictions(
        parts.calibration,
        {**calibration_predictions, selected_name: calibration_selected},
        selected_name,
        output / "calibration_predictions.parquet",
    )

    # Comparaciones primarias raw-vs-raw, ademas de selected-vs-all.
    raw_champion = test_predictions[champion]
    bootstrap_raw = {
        name: bootstrap_metric_difference(
            parts.test[TARGET].to_numpy(), raw_champion, probability,
            args.bootstrap_repetitions, args.seed,
        )
        for name, probability in test_predictions.items()
        if name not in {champion, selected_name}
    }
    bootstrap_selected = {
        name: bootstrap_metric_difference(
            parts.test[TARGET].to_numpy(), selected_test, probability,
            args.bootstrap_repetitions, args.seed + 1,
        )
        for name, probability in test_predictions.items()
        if name != selected_name
    }

    report = {
        "mode": args.mode,
        "champion_selected_on_tune": champion,
        "probability_policy_selected_on_calibration": policy,
        "dataset_fingerprint": file_fingerprint(dataset),
        "rows": {
            "train": len(parts.train), "tune": len(parts.tune),
            "calibration": len(parts.calibration), "test": len(parts.test),
        },
        "features": parts.features,
        "categoricals": parts.categoricals,
        "tuning": tuning_info,
        "tune_metrics": tune_metrics,
        "calibration_metrics": calibration_metrics,
        "test_metrics": test_metrics,
        "bootstrap_raw_champion_differences": bootstrap_raw,
        "bootstrap_selected_policy_differences": bootstrap_selected,
        "selection_protocol": (
            "candidate models fit on train only; champion selected on tune; "
            "ordinary candidates refit on train+tune; calibration policy selected "
            "on calibration; test untouched"
        ),
        "test_used_for_selection": False,
        "elapsed_seconds": time.time() - started,
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "pandas": pd.__version__,
        },
    }
    atomic_json(report, output / "training_report.json")
    atomic_json({
        "champion": champion,
        "probability_policy": policy["selected"],
        "features": parts.features,
        "categoricals": parts.categoricals,
        "mode": args.mode,
        "calibration_thresholds": {
            "minimum_logloss_gain": MIN_CALIBRATION_LOGLOSS_GAIN,
            "minimum_brier_gain": MIN_CALIBRATION_BRIER_GAIN,
            "maximum_ece_degradation": MAX_CALIBRATION_ECE_DEGRADATION,
        },
    }, output / "model_metadata.json")

    summary: list[dict[str, Any]] = []
    for period, metrics in (
        ("tune", tune_metrics),
        ("calibration", calibration_metrics),
        ("test", test_metrics),
    ):
        for name, values in metrics.items():
            summary.append({"period": period, "model": name, **values})
    summary_frame = pd.DataFrame(summary)
    atomic_csv(summary_frame, output / "metrics.csv")
    print(
        summary_frame.loc[summary_frame.period.eq("test")]
        .sort_values("log_loss")
        .to_string(index=False)
    )
    print(f"\nChampion: {champion}; policy: {policy['selected']}")
    print(f"Artefactos: {output}")
    print(f"Tiempo: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
