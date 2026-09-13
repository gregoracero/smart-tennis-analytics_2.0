# 12. Historial de decisiones técnicas

## Mantener trazabilidad en Parquets

Se decidió conservar fuente, archivos originales y banderas de calidad en los Parquets derivados. El manifiesto impide que entren en el modelo.

## Excluir Futures del dataset profesional por defecto

Los Futures siguen contribuyendo al historial de los jugadores, pero no forman parte del dataset limpio de entrenamiento por defecto. ATP, qualifying y Challenger sí se incluyen.

## No imputar globalmente

No se realiza imputación usando todo el dataset. CatBoost maneja ausentes numéricos y las categorías ausentes se representan como `__MISSING__`.

## Walkovers

Se conservan para auditoría, pero no actualizan estado histórico ni se usan para entrenar.

## Ceros estadísticos TML

Los ceros imposibles en `SvGms` y `svpt` se interpretan como ausencia, no como valor deportivo real.

## Reconciliación conservadora

Se aceptan automáticamente solo coincidencias con margen y evidencia suficiente. Los casos dudosos conservan un ID sintético estable.

## Test 2026 aislado

Train termina en 2024, validation cubre 2025 y test cubre 2026. Test no se utiliza para selección ni calibración.

## Probabilidad raw

Se exige una mejora mínima en calibration antes de aplicar Platt. Los modelos actuales seleccionan raw.

## Candidato antes de producción

El entrenamiento escribe en `modeling/artifacts/candidate`. Producción solo se sustituye tras validación explícita.
