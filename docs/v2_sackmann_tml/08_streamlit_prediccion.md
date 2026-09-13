# 8. Página de predicción Streamlit

## Modelo de producción

```text
modeling/artifacts/production/no_market
```

Archivos requeridos:

```text
model.cbm
platt_calibrator.joblib
probability_policy.json
run_metadata.json
```

## Cambios necesarios

### Import estable

El hack antiguo con `__main__` se elimina. La página importa:

```python
from modeling.calibration import PlattCalibrator
```

La raíz del proyecto debe estar añadida a `sys.path` antes del import.

### Política de probabilidad

La página lee `probability_policy.json`:

- `raw`: usa directamente CatBoost.
- `calibrated`: usa Platt.
- Si falta el archivo, usa `raw`.

No basta con que el calibrador exista para aplicarlo.

### Esquema

Antes de predecir se valida:

```text
model.feature_names_ == run_metadata.feature_columns
```

### Orientación de inferencia

El entrenamiento asignó `player_1` al ID menor. La página:

1. Ordena internamente los dos IDs.
2. Construye las features en esa orientación.
3. Predice la probabilidad del `model_player_1`.
4. Invierte la probabilidad si el orden visual es distinto.

Esto garantiza que cambiar el orden visual produzca probabilidades complementarias.

### synthetic_player_any

Durante inferencia:

```text
1 si player_1_id >= 90.000.000 o player_2_id >= 90.000.000
0 en otro caso
```

Se muestra una advertencia si participa una identidad sintética.

### Información del modelo

La página debe mostrar:

```text
Feature Count
Probability Policy
Calibrator Available
Calibration Applied
Synthetic Player Present
Missing Inference Features
```

## Prueba de simetría

Predecir A contra B y después B contra A:

```text
P(A vence a B) + P(B vence a A) ~= 1
```

## Confianza

El score de confianza representa cobertura de datos, no certeza del resultado. Considera el mínimo de:

- Partidos de carrera.
- Partidos por superficie.
- Partidos con estadísticas.
- H2H.
