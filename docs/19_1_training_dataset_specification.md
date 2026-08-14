# 19.1 Training Dataset Specification

## Years

2010+

## Surfaces

- Hard
- Clay
- Grass

## Tournament Levels

Included

- G
- M
- 1000
- 500
- 250
- C

Excluded

- A
- D
- F
- O

## Models

ATP

- ATP_HARD
- ATP_CLAY
- ATP_GRASS

CH

- CH_HARD
- CH_CLAY
- CH_GRASS

## Windows

- Last 5 Matches
- Last 10 Matches
- Last 20 Matches
- Last 50 Matches

## Features

### Ranking

- rank

### Form

- win_pct

### Serve

- first_serve_in_pct
- first_serve_won_pct
- second_serve_won_pct
- service_points_won_pct

### Return

- return_points_won_pct

### Pressure

- break_conversion_pct
- bp_saved_pct
- tiebreak_win_pct

### Activity

- matches_last_30_days
- days_since_last_match

### Profile

- age
- height
- hand

## Target

Every historical match generates:

Winner vs Loser -> target=1

Loser vs Winner -> target=0

## Anti Leakage Rule

Only matches played before the match date
can be used to calculate features.

