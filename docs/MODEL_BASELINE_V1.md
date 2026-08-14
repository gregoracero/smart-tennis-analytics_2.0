# Smart Tennis Analytics - Baseline V1

## Objetivo

Construir modelos de predicción ATP utilizando únicamente información disponible antes del partido (anti-leakage).

---

## Dataset Fuente

### Master Matches

data/parquet/master_matches.parquet

Contiene:

- ganador
- perdedor
- ranking
- ranking points
- superficie
- fecha
- estadísticas ATP

### Player Matches

data/parquet/analytics_player_matches.parquet

Contiene:

- histórico por jugador
- estadísticas agregadas para feature engineering

---

## Filtros Aplicados

### Superficies

- Hard
- Clay
- Grass

### Niveles ATP

- G
- M
- 1000
- 500
- 250

### Fecha mínima

2010

---

## Anti Leakage

Para cada partido:

match_date

Se utilizan únicamente partidos anteriores:

player_match_date < match_date

Nunca se usan datos futuros.

---

## Ventanas Históricas

- 5 partidos
- 10 partidos
- 20 partidos
- 50 partidos

Siempre sobre la misma superficie.

---

## Features Generadas

### Ranking

- rank_a
- rank_b
- delta_rank

### Ranking Points

- rank_points_a
- rank_points_b
- delta_rank_points

### Edad

- age_a
- age_b
- delta_age

### Altura

- height_a
- height_b
- delta_height

### Mano Dominante

- is_left_handed_a
- is_left_handed_b

### Historial

- history_size_a
- history_size_b
- delta_history_size

### Win %

- delta_win_pct_5
- delta_win_pct_10
- delta_win_pct_20
- delta_win_pct_50

### Servicio

- delta_first_serve_in_pct
- delta_first_serve_won_pct
- delta_second_serve_won_pct
- delta_service_points_won_pct

para ventanas:

- 5
- 10
- 20
- 50

### Devolución

- delta_return_points_won_pct

para ventanas:

- 5
- 10
- 20
- 50

### Break Points

- delta_break_conversion_pct
- delta_bp_saved_pct

para ventanas:

- 5
- 10
- 20
- 50

### Tie Breaks

- delta_tiebreak_win_pct

para ventanas:

- 5
- 10
- 20
- 50

---

## Dataset Final

training_matches.parquet

Filas:

200118

Columnas:

70

Target:

0 = jugador A pierde

1 = jugador A gana

Balance:

100059 victorias

100059 derrotas

---

## Modelos V1

### ATP Hard

Modelo:

XGBoost

Accuracy:

0.6723

ROC AUC:

0.7362

---

### ATP Clay

Modelo:

Logistic Regression

Accuracy:

0.6510

ROC AUC:

0.7143

---

### ATP Grass

Modelo:

Logistic Regression

Accuracy:

0.6597

ROC AUC:

0.7238

---

## Principales Features

Las variables con mayor capacidad predictiva han sido:

- delta_rank_points
- delta_rank
- delta_win_pct
- delta_service_points_won_pct
- delta_return_points_won_pct

---

## Próxima Fase (V2)

### Elo Global

- elo_a
- elo_b
- delta_elo

### Elo por Superficie

- surface_elo_a
- surface_elo_b
- delta_surface_elo

### Head To Head

- h2h_a
- h2h_b
- delta_h2h

Objetivo:

Superar ROC AUC 0.75 en ATP Hard.
