# 3. Integración de fuentes

## Fuentes

### Jeff Sackmann

Se cargaron 144 archivos:

- ATP principal desde 1968.
- Futures desde 1991.
- Qualifying y Challenger desde 1978.

Total inicial Sackmann: **963.215 partidos**.

### TennisMyLife

Directorio:

```text
data/raw/tml_v2/current_season
```

Archivos:

```text
2026.csv                 -> ATP
2026_atp_quali.csv       -> ATP_QUALIFYING
2026_challenger.csv      -> CHALLENGER
```

Registros conservados tras integración y reconciliación:

```text
ATP                 2.132
ATP_QUALIFYING      1.103
CHALLENGER          5.234
Total               8.469
```

### tennis-data.co.uk

Se cargaron 68.264 partidos con cuotas entre 2001 y 2026. Se aceptaron 61.555 coincidencias.

## Salida integrada inicial

Antes del postproceso final:

```text
output_rows                 971.093
output_columns                  118
unique_logical_matches      971.093
odds_matches                 61.555
```

## Deduplicación

La clave lógica incluye:

```text
tourney_date
tourney_name normalizado
winner_name normalizado
loser_name normalizado
round
competition_type
```

La clave operativa final es:

```text
tourney_id + competition_type + match_num
```

Se conserva la fila con mayor prioridad y cobertura estadística. TML puede sustituir la versión Sackmann cuando representa una observación más actual o completa de la temporada corriente.

## Competición explícita

La columna `competition_type` evita mezclar:

```text
ATP
ATP_QUALIFYING
CHALLENGER
FUTURES
```

Esto resuelve colisiones de `tourney_id` y permite filtrar datasets profesionales sin depender exclusivamente de `tourney_level`.
