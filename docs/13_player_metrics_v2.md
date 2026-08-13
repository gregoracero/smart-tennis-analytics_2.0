# Player Metrics V2

## Objetivo

Ampliar las métricas de jugador incorporando indicadores
de presión y capacidad de ruptura del servicio rival.

## Dataset de entrada

data/parquet/analytics_player_matches.parquet

## Dataset de salida

data/parquet/player_metrics.parquet

## Clave

player
surface
window

## Ventanas soportadas

### LAST_10_MATCHES

Últimos 10 partidos en la superficie.

### LAST_3_MONTHS

Últimos 3 meses en la superficie.

### LAST_6_MONTHS

Últimos 6 meses en la superficie.

### CAREER_SURFACE

Histórico completo en la superficie.

## Métricas oficiales V2

### Forma reciente

- matches
- wins
- losses
- surface_recent_win_pct

### Servicio

#### First Serve Won %

first_serve_won_pct

Porcentaje de puntos ganados cuando entra
el primer servicio.

#### First Serve In %

first_serve_in_pct

Porcentaje de primeros servicios que entran.

#### Second Serve Won %

second_serve_won_pct

Porcentaje de puntos ganados con segundo servicio.

### Break Points

#### BP Saved %

bp_saved_pct

Capacidad para salvar bolas de break.

#### Break Conversion %

break_conversion_pct

Capacidad para convertir oportunidades
de break en breaks reales.

### Tie-Break Performance

#### Tie-Breaks Played

tiebreaks_played

#### Tie-Breaks Won

tiebreaks_won

#### Tie-Break Win %

tiebreak_win_pct

Porcentaje de tie-breaks ganados.

### Resistencia

#### Average Match Duration

avg_minutes

Duración media de los partidos en la ventana.

## Métricas pendientes para futuras versiones

### Winners / Unforced Errors Ratio

winner_ue_ratio

Actualmente no disponible en el dataset TML.

Requiere una fuente adicional de datos.

## Interpretación orientativa

### First Serve Won %

- > 75% → Elite
- 70%-75% → Muy bueno
- 65%-70% → Correcto
- < 65% → Vulnerable

### First Serve In %

- > 65% → Muy consistente
- 60%-65% → Media ATP
- < 60% → Exceso de segundos servicios

### Second Serve Won %

- > 55% → Excelente
- 50%-55% → Bueno
- 45%-50% → Riesgo
- < 45% → Muy vulnerable

### Break Conversion %

- > 35% → Excelente retornador
- 30%-35% → Muy bueno
- 25%-30% → Correcto
- < 25% → Débil al resto

### Tie-Break Win %

- > 60% → Jugador clutch
- 50%-60% → Normal
- < 50% → Débil bajo presión

## Uso

Estas métricas serán la base principal de:

- Match Comparator
- Prediction Engine
- Dashboard
- Value Analysis

## Prioridad Analítica

Las métricas con mayor peso en el análisis prepartido son:

1. surface_recent_win_pct
2. first_serve_won_pct
3. second_serve_won_pct
4. break_conversion_pct
5. tiebreak_win_pct

Todas las métricas deben analizarse:

- por superficie
- por ventana temporal
- en contexto con el rival

Nunca utilizando únicamente valores de carrera agregados.
