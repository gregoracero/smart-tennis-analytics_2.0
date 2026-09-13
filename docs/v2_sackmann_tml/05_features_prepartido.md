# 5. Construcción de features prepartido

## Entrada y salida

Entrada:

```text
data/processed/jeff_sackmann_with_odds.parquet
```

Salida:

```text
data/processed/tennis_matches_with_player_stats.parquet
```

Resultado:

```text
Filas                                971.080
Columnas                                 163
Features orientadas/diferenciales        91
Elegibles para modelado              966.855
Excluidos                               4.225
```

## Estados acumulados

Cada jugador mantiene:

- Partidos, victorias y derrotas.
- Elo global.
- Partidos, victorias y Elo por superficie.
- Últimos 10 resultados.
- Minutos recientes.
- Fecha del último partido.
- Estadísticas acumuladas de servicio y retorno.

## Features principales

```text
career_matches_before
career_wins_before
career_win_rate_before
surface_matches_before
surface_wins_before
surface_win_rate_before
elo_before
surface_elo_before
form_last_5_before
form_last_10_before
avg_minutes_last_5_before
days_since_last_match
stat_matches_before
aces_per_service_game_before
double_faults_per_service_game_before
first_serve_in_pct_before
first_serve_win_pct_before
second_serve_win_pct_before
service_points_won_pct_before
break_points_saved_pct_before
return_points_won_pct_before
```

## H2H

```text
player_1_h2h_wins_before
player_2_h2h_wins_before
player_1_h2h_win_rate_before
```

## Orden temporal

Se ordena por:

```text
match_date
tourney_id
competition_order
round_order
match_num
original_row
```

Qualifying se procesa antes que el cuadro principal cuando no existe una fecha real separada.

## Partidos no elegibles

```text
walkover                  4.219
same_player_identity          6
missing_match_date            0
```

Estos partidos se conservan para auditoría, pero no actualizan Elo, forma, carrera, superficie, H2H ni estadísticas.

## Orientación

La versión utilizada para entrenar asigna `player_1` al jugador con ID numérico menor. La inferencia de Streamlit debe reproducir esa orientación y después mapear la probabilidad al orden elegido por el usuario.

## Validación final

```text
Duplicados operativos       0
player_1_id nulos           0
player_2_id nulos           0
Fecha máxima       2026-08-30
TML rows                8.469
```
