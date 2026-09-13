# 7. Entrenamiento y evaluación

## Entrenador

```text
modeling/train_professional_tennis_model_tml.py
```

## Flujo

1. Lee la lista de features desde `tennis_ml_dataset_metadata.json`.
2. Carga solo features y columnas de auditoría necesarias.
3. Usa los splits existentes.
4. Divide validation cronológicamente en tune y calibration.
5. Ajusta CatBoost con train y tune.
6. Entrena el modelo final con train + tune.
7. Ajusta Platt solo con calibration.
8. Evalúa test 2026 una única vez.
9. Guarda primero en `candidate`.

## Modelo no market

```text
Train               236.031
Tune                   7.024
Calibration            7.025
Test                   14.864
Features                   82
Categóricas                 4
Iteraciones finales       711
```

### Test 2026

```text
Métrica           Raw         Calibrated    Elo
Log Loss          0,585991    0,588158      0,605975
Brier             0,200625    0,201471      0,208922
Accuracy          0,690258    0,690393      0,673708
ROC AUC           0,757546    0,757546      0,734416
ECE               0,020481    0,032141      0,020482
```

La probabilidad raw supera claramente a Elo y a las baselines constantes. Platt mejora levemente en calibration, pero empeora test.

## Modelo with market

```text
Train                52.441
Tune                   1.067
Calibration            1.068
Test                    1.618
Features                  102
Iteraciones finales       371
```

### Test 2026

```text
Métrica              Raw         Calibrated   Mercado avg no-vig
Log Loss             0,595868    0,595578     0,598485
Brier                0,205638    0,205445     0,206750
Accuracy             0,679234    0,677379     0,677998
ROC AUC              0,741532    0,741532     0,737292
```

La mejora frente al mercado es pequeña y el test es reducido. El modelo con mercado se mantiene como experimental.

## Política raw/calibrated

`probability_policy.json` define qué probabilidad usar. La regla conservadora utiliza solo calibration:

```text
Seleccionar calibrated únicamente si:
- mejora Log Loss al menos 0,001;
- y no empeora Brier.
```

Para los candidatos actuales se selecciona `raw`.

## Calibrador

El calibrador debe serializarse como:

```text
modeling.calibration.PlattCalibrator
```

No se admite `calibration.PlattCalibrator` ni `__main__.PlattCalibrator`.

## Subgrupos no market

### Fuente

```text
TML       Log Loss 0,611309 | Accuracy 0,663126 | AUC 0,726021
SACKMANN  Log Loss 0,559003 | Accuracy 0,724730 | AUC 0,795344
```

### IDs sintéticos

```text
Sin sintéticos  Log Loss 0,585179 | AUC 0,761455 | 14.541 filas
Con sintéticos  Log Loss 0,722252 | AUC 0,617668 |    323 filas
```

El rendimiento inferior con IDs sintéticos justifica una advertencia en producción y continuar la reconciliación.
