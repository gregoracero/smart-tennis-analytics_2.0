# Dashboard Streamlit

## Objetivo

Visualizar métricas comparativas entre dos jugadores
de forma interactiva.

## Inputs

data/parquet/player_metrics.parquet

data/parquet/h2h_overall.parquet

data/parquet/h2h_surface.parquet

## Funcionalidades

### Filtros

- Player A
- Player B
- Surface
- Window

### Overview

- Record reciente
- Win %
- H2H Overall
- H2H Surface

### Servicio

- First Serve In %
- First Serve Won %
- Second Serve Won %

### Break Points

- BP Saved %
- Break Conversion %

### Pressure

- Tie Break Win %
- Tie Breaks Per Match

### Resistencia

- Average Match Duration

## Tecnología

- Streamlit
- Pandas
- Plotly

## Fuente Oficial

### Match Layer

data/parquet/master_matches.parquet

### Metrics Layer

data/parquet/player_metrics.parquet

### H2H Layer

data/parquet/h2h_overall.parquet

data/parquet/h2h_surface.parquet

## Roadmap

### Dashboard MVP

- Selector de jugadores
- Selector de superficie
- Selector de ventana
- Tabla comparativa

### Dashboard V2

- KPI Cards
- H2H Cards
- Edge Analysis

### Dashboard V3

- Radar Chart
- Visualización tipo ATP Tour
- Comparativa gráfica avanzada

## Ejecución

streamlit run dashboard/app.py


# Dashboard V2

## KPI Cards

Mostrar:

- Matches
- Wins
- Win %
- Avg Minutes

para ambos jugadores.

## Head To Head

Mostrar:

- H2H Overall
- H2H Surface

## Inputs

data/parquet/h2h_overall.parquet

data/parquet/h2h_surface.parquet

## Objetivo

Sustituir la tabla plana por indicadores visuales.


# Dashboard H2H

## Objetivo

Visualizar el Head To Head entre dos jugadores
con formato ATP Head2Head.

## Componentes

### H2H Overview

- Wins Player A
- Wins Player B
- Total Matches

### H2H Surface

- Hard
- Clay
- Grass

### Event Breakdown

Histórico completo de enfrentamientos.

## Inputs

data/parquet/h2h_overall.parquet

data/parquet/h2h_surface.parquet

## Salida

Dashboard interactivo H2H.


## Event Breakdown

Mostrar el histórico completo de enfrentamientos.

Campos:

- Date
- Tournament
- Surface
- Round
- Winner
- Score

Fuente:

data/parquet/h2h_matches.parquet


# Players Form

## Objetivo

Comparar la forma reciente de ambos jugadores.

## Input

data/parquet/player_metrics.parquet

## Métricas

### Overall

- Matches
- Wins
- Win %

### Serve

- Aces Per Match
- Double Faults Per Match
- First Serve In %
- First Serve Won %
- Second Serve Won %
- Service Points Won %
- Service Games Won %

### Return

- Return Points Won %
- First Return Points Won %
- Second Return Points Won %
- Return Games Won %

### Break Points

- Break Points Chances Per Match
- Break Points Converted Per Match
- Break Point Conversion %

### Misc

- Avg Minutes

## Fuente

player_metrics.parquet

