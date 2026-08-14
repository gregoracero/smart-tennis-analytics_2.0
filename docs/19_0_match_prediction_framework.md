# 19.0 Match Prediction Framework

## Objetivo

Predecir la probabilidad de victoria antes de un partido.

## Modelos

Surface specific

- Hard
- Clay
- Grass

Windows

- Last 2 Seasons
- Last 3 Months
- Last 2 Weeks

## Baseline

Surface Elo

## Candidate Models

- Logistic Regression
- Random Forest
- XGBoost

## Features

- Win %
- Service Metrics
- Return Metrics
- Break Point Metrics
- Tiebreak Metrics

## Output

Probability Player A Wins

0.00 - 1.00

## Explainability

SHAP values

Feature Importance

