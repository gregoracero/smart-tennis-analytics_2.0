# Player Statistics V2

## Objetivo

Generar estadísticas agregadas de jugadores a partir de:

master_matches_v2.parquet

---

## Output

data/parquet/v2/player_statistics_v2.parquet

---

## Grain

player_key

surface

tournament_level

scope

season

---

## Scope

CAREER

SEASON

TOURNAMENT

---

## Surface

ALL

HARD

CLAY

GRASS

INDOOR

---

## Tournament Levels

ALL

GRAND_SLAM

ATP_1000

ATP_500

ATP_250

---

## Métricas

Match

- matches
- wins
- losses
- win_pct

Sets

- sets_won
- sets_lost
- set_win_pct

Games

- games_won
- games_lost
- game_win_pct

Dominance

- wins_to_love
- losses_to_love

First Set

- first_sets_won
- first_sets_lost
- first_set_win_pct

Service

- aces
- aces_per_match

- double_faults
- double_faults_per_match

- first_serve_in_pct

- first_serve_won_pct

- second_serve_won_pct

- service_points_won_pct

- service_games_won_pct

Return

- first_return_points_won_pct

- second_return_points_won_pct

- return_points_won_pct

- return_games_won

Break Points

- break_points_converted

- break_points_chances

- break_points_converted_pct

- break_points_saved

- break_points_faced

- break_points_saved_pct

Tie Breaks

- tiebreaks_played

- tiebreaks_won

- tiebreaks_lost

- tiebreak_win_pct

Pressure

- pressure_index

Time On Court

- minutes_played

- minutes_per_match

Opponent Strength

- avg_opponent_rank

- median_opponent_rank

- top10_matches
- top10_wins

- top20_matches
- top20_wins

- top50_matches
- top50_wins

- top100_matches
- top100_wins

Betting

- yield_total

- yield_as_favorite

- yield_as_underdog

