# PLAYER SNAPSHOTS V2 - DOCUMENTACION TECNICA

## Objetivo

Construir una capa de snapshots que represente el estado completo de un jugador inmediatamente antes de un partido.

Granularidad:

master_match_id
+
player_id

Resultado:

1 snapshot por jugador y partido.

Por tanto:

1 partido
=
2 snapshots

---

# Posición en el Pipeline

ingestion
↓
master_matches_v2

analytics
↓
player_per_match_statistics_v2
↓
player_statistics_v2
↓
tournament_statistics_v2
↓
elo_history_v2

prediction
↓
player_snapshots_v2

---

# Objetivo Funcional

Convertir información histórica agregada en variables utilizables para modelos de Machine Learning.

Los snapshots representan:

"Todo lo que sabíamos del jugador justo antes del partido"

El diseño evita data leakage.

No contiene información futura.

No contiene resultados posteriores al partido.

---

# Datasets de Entrada

## player_per_match_statistics_v2

Fuente base del snapshot.

Utilizado para:

- master_match_id
- match_date
- season
- surface
- tournament_level
- player_id
- player_key
- player_name
- player_rank
- player_rank_points
- player_age
- player_height

---

## player_statistics_v2

Se utiliza para incorporar estadísticas históricas agregadas.

### Career Statistics

Filtrado:

scope = CAREER

surface = ALL

tournament_level = ALL

Variables obtenidas:

- career_matches
- career_win_pct
- career_set_win_pct
- career_aces_per_match
- career_double_faults_per_match
- career_bp_saved_pct
- career_bp_conv_pct
- career_pressure_index

---

### Surface Statistics

Filtrado:

scope = CAREER

surface != ALL

tournament_level = ALL

Variables obtenidas:

- surface_matches
- surface_win_pct
- surface_set_win_pct

Merge mediante:

player_id
+
surface

---

### Season Statistics

Filtrado:

scope = SEASON

surface = ALL

tournament_level = ALL

Variables obtenidas:

- season_matches
- season_win_pct
- season_minutes_per_match

Merge mediante:

player_id
+
season

---

## elo_history_v2

Se utiliza para incorporar el estado Elo previo al partido.

Variables obtenidas:

- elo_overall_before
- elo_surface_before
- elo_mix_before
- career_matches_before
- days_since_last_match
- inactivity_adjusted
- expected_win_probability

---

# Construcción

## Paso 1

Generación de tabla base desde:

player_per_match_statistics_v2

Columnas iniciales:

- master_match_id
- match_date
- season
- surface
- tournament_level
- player_id
- player_key
- player_name
- player_rank
- player_rank_points
- player_age
- player_height

---

## Paso 2

Merge con:

elo_history_v2

Claves:

master_match_id
+
player_id

Variables añadidas:

- career_matches_before
- days_since_last_match
- inactivity_adjusted
- elo_overall_before
- elo_surface_before
- elo_mix_before
- expected_win_probability

---

## Paso 3

Merge con:

career_stats

Clave:

player_id

Variables añadidas:

- career_matches
- career_win_pct
- career_set_win_pct
- career_aces_per_match
- career_double_faults_per_match
- career_bp_saved_pct
- career_bp_conv_pct
- career_pressure_index

---

## Paso 4

Merge con:

surface_stats

Claves:

player_id
+
surface

Variables añadidas:

- surface_matches
- surface_win_pct
- surface_set_win_pct

---

## Paso 5

Merge con:

season_stats

Claves:

player_id
+
season

Variables añadidas:

- season_matches
- season_win_pct
- season_minutes_per_match

---

# Variables Finales

## Match Context

- master_match_id
- match_date
- season
- surface
- tournament_level

---

## Identidad

- player_id
- player_key
- player_name

---

## Ranking

- player_rank
- player_rank_points

---

## Biografía

- player_age
- player_height

---

## Elo

- elo_overall_before
- elo_surface_before
- elo_mix_before
- expected_win_probability

---

## Actividad

- career_matches_before
- days_since_last_match
- inactivity_adjusted

---

## Career Performance

- career_matches
- career_win_pct
- career_set_win_pct
- career_aces_per_match
- career_double_faults_per_match
- career_bp_saved_pct
- career_bp_conv_pct
- career_pressure_index

---

## Surface Performance

- surface_matches
- surface_win_pct
- surface_set_win_pct

---

## Season Performance

- season_matches
- season_win_pct
- season_minutes_per_match

---

# Validaciones Realizadas

## Integridad de Filas

PLAYER MATCH

133146

PLAYER SNAPSHOTS

133146

Resultado:

Sin pérdida de filas.

---

## Integridad de Jugadores

PLAYERS

1573

Resultado:

Consistente con player_mapping_v2.

---

## Integridad de Merges

Auditoría realizada:

elo_overall_before

surface_win_pct

season_win_pct

career_win_pct

Resultado:

0% valores nulos

Cobertura completa.

---

## Integridad Temporal

Todas las métricas Elo utilizadas corresponden a:

*_before

Es decir:

Estado previo al partido.

No existe leakage temporal.

---

# Dataset Resultante

player_snapshots_v2.parquet

Granularidad:

master_match_id
+
player_id

Filas:

133146

Jugadores:

1573

Columnas:

Snapshot completo pre-partido.

---

# Utilidad en Prediction

player_snapshots_v2 constituye la capa puente entre:

analytics
↓
prediction

Su objetivo es evitar recalcular estadísticas continuamente.

Permite construir:

- training_matches_v2
- prediction_features_v2

de forma eficiente.

---

# FUTURAS MEJORAS

## Recent Form

Actualmente no existe una métrica explícita de forma reciente.

Se añadirá una nueva capa rolling basada exclusivamente en partidos anteriores al snapshot.

Posibles variables:

- last_5_matches_win_pct
- last_10_matches_win_pct
- last_20_matches_win_pct

- last_10_surface_win_pct
- last_20_surface_win_pct

- recent_aces_per_match
- recent_double_faults_per_match

- recent_pressure_index

Implementación prevista:

Para cada snapshot:

Tomar únicamente partidos anteriores a match_date.

Ordenar cronológicamente.

Calcular rolling windows.

Ejemplo:

last_10_matches_win_pct

=

victorias últimos 10 partidos
/
10

---

## Head-To-Head (H2H)

Actualmente no existe información de enfrentamientos directos.

Se añadirá una capa H2H histórica libre de leakage.

Para cada partido:

Jugador A
vs
Jugador B

considerando únicamente enfrentamientos anteriores a la fecha actual.

Variables previstas:

- h2h_matches

- h2h_wins

- h2h_losses

- h2h_win_pct

- h2h_surface_matches

- h2h_surface_win_pct

- h2h_last_5_matches_win_pct

Ejemplo:

Djokovic vs Nadal

Antes de Roland Garros 2015

El snapshot sólo verá:

enfrentamientos Djokovic-Nadal anteriores a Roland Garros 2015.

Nunca posteriores.

---

## Fatigue Features

Posible ampliación futura.

Variables:

- matches_last_7_days
- matches_last_14_days
- matches_last_30_days

- minutes_last_7_days
- minutes_last_14_days
- minutes_last_30_days

Permiten estimar:

- desgaste
- acumulación de carga
- riesgo de bajada de rendimiento

---

## Serve / Return Strength

Considerada una de las futuras mejoras más potentes.

Variables:

Service

- service_games_won_pct
- service_points_won_pct
- first_serve_won_pct
- second_serve_won_pct

Return

- break_points_converted_pct
- return_games_won_pct

Versiones:

- Career
- Surface
- Rolling

---

# Roadmap

Estado actual:

✅ player_per_match_statistics_v2

✅ player_statistics_v2

✅ tournament_statistics_v2

✅ elo_history_v2

✅ player_snapshots_v2

Próximo paso:

build_training_matches_v2.py

Objetivo:

Transformar:

2 snapshots

↓

1 fila de entrenamiento

utilizando:

- elo_delta
- surface_elo_delta
- rank_delta
- rank_points_delta
- age_delta
- height_delta
- career_win_pct_delta
- surface_win_pct_delta
- season_win_pct_delta
- career_matches_delta
- days_since_last_match_delta

y la variable objetivo:

target

---

# Conclusión

player_snapshots_v2 consolida toda la información disponible sobre un jugador antes de disputar un partido.

Representa el estado actual del jugador en términos de:

- ranking
- experiencia
- rendimiento histórico
- rendimiento por superficie
- rendimiento de temporada
- actividad reciente
- Elo

y constituye la base fundamental para la construcción de datasets de Machine Learning en Smart Tennis Analytics V2.

