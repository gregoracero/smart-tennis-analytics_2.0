# Player Per Match Statistics V2

## Objetivo

Transformar la capa:

master_matches_v2.parquet

en una capa analítica con granularidad:

1 fila = 1 jugador en 1 partido

Esta capa constituye la base para:

- player_statistics_v2
- elo_history_v2
- player_snapshots_v2
- training_matches_v2
- betting_statistics_v2
- tournament_statistics_v2

---

# Input

data/parquet/v2/master_matches_v2.parquet

---

# Output

data/parquet/v2/player_per_match_statistics_v2.parquet

---

# Grain

1 fila = 1 jugador en 1 partido

Ejemplo:

Federer vs Nadal

produce:

- Federer
- Nadal

---

# Dataset Quality

Rows: 136.440

Columns: 43

Players: 1.584

Wins: 68.220

Losses: 68.220

---

# Match Identity

master_match_id

match_date

season

surface

tournament_id

tournament_key

tournament_name

tournament_level

round

---

# Player

player_key

player_name

player_rank

player_rank_points

player_age

player_height

player_hand

---

# Opponent

opponent_player_key

opponent_name

opponent_rank

opponent_rank_points

opponent_age

---

# Result

result

match

win

loss

---

# Sets

sets_won

sets_lost

---

# Service

aces

double_faults

first_serve_in

first_serve_points_won

second_serve_points_won

service_points_played

service_games_played

service_games_won

---

# Break Points

break_points_faced

break_points_saved

breaks_suffered

break_points_chances

break_points_converted

---

# Time On Court

minutes

---

# Betting

player_b365

opponent_b365

---

# Null Policy

Los valores faltantes NO se sustituyen por cero.

Se conservan como NULL para evitar sesgar:

- medias
- ratios
- estadísticas agregadas

---

# Analytics Layer Rules

## Totales

Siempre:

SUM(metric)

Ejemplos:

- aces
- double_faults
- minutes_played
- break_points_faced

---

## Métricas Per Match

Nunca:

metric_total / matches

Porque pueden existir partidos sin datos para esa métrica.

Siempre:

SUM(metric)
/
COUNT(metric)

Ejemplos:

aces_per_match

double_faults_per_match

minutes_per_match

break_points_faced_per_match

break_points_saved_per_match

break_points_chances_per_match

break_points_converted_per_match

---

## Porcentajes

Nunca:

AVG(match_pct)

Siempre:

SUM(numerator)
/
SUM(denominator)

Ejemplos:

first_serve_in_pct

=
SUM(first_serve_in)
/
SUM(service_points_played)

---

first_serve_won_pct

=
SUM(first_serve_points_won)
/
SUM(first_serve_in)

---

second_serve_won_pct

=
SUM(second_serve_points_won)
/
SUM(service_points_played - first_serve_in)

---

service_points_won_pct

=
SUM(first_serve_points_won + second_serve_points_won)
/
SUM(service_points_played)

---

break_points_saved_pct

=
SUM(break_points_saved)
/
SUM(break_points_faced)

---

break_points_converted_pct

=
SUM(break_points_converted)
/
SUM(break_points_chances)

---

# Data Quality Philosophy

La capa analítica debe preservar la calidad original de los datos.

Por tanto:

NULL != 0

Un partido sin estadística disponible no debe influir negativamente en una métrica agregada.

---

# Arquitectura

master_matches_v2
        ?
player_per_match_statistics_v2
        ?
player_statistics_v2
        ?
elo_history_v2
        ?
player_snapshots_v2
        ?
training_matches_v2

---

# Estado

? Implementado

? Validado

? 136.440 registros generados

? Listo para Analytics Layer

