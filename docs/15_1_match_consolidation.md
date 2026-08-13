# Match Consolidation & Duplicate Detection

## Objetivo

Construir una fuente consolidada de partidos
combinando histórico y partidos recientes.

## Inputs

data/parquet/all_matches.parquet

data/raw/tml/ongoing_tourneys.csv

## Output

data/parquet/current_matches.parquet

## Proceso

1. Cargar histórico
2. Cargar ongoing tournaments
3. Normalizar tipos
4. Detectar duplicados
5. Conservar una única copia
6. Generar dataset consolidado

## Clave principal

tourney_id
match_num
winner_id
loser_id

## Clave fallback

tourney_date
winner_name
loser_name
score

## Auditoría

Mostrar:

- Historical rows
- Ongoing rows
- Duplicates removed
- New matches added
- Final rows

## Uso

Fuente oficial para:

- analytics_matches
- analytics_player_matches
- player_metrics


# ============================================================
# PIPELINE OPERATIVO
# ============================================================

## Actualización completa de datos

Ejecutar los siguientes pasos en orden.

### 1. Actualizar datos TML

python ingestion/update_tml_data.py

Actualiza:

- current year ATP
- current year Challenger
- ongoing_tourneys.csv

---

### 2. Reconstruir histórico

python ingestion/build_all_matches.py

Genera:

data/parquet/all_matches.parquet

Contiene:

- ATP histórico
- Challenger histórico

No incluye:

- ongoing_tourneys.csv

---

### 3. Construir dataset maestro

python ingestion/build_master_matches.py

Genera:

data/parquet/master_matches.parquet

Contiene:

all_matches.parquet
+
ongoing_tourneys.csv

Es la fuente oficial de partidos del proyecto.

---

### 4. Construir analytics matches

python ingestion/build_analytics_matches.py

Input:

data/parquet/master_matches.parquet

Output:

data/parquet/analytics_matches.parquet

---

### 5. Enriquecer tie-breaks

python ingestion/enrich_tiebreaks.py

Output:

data/parquet/analytics_matches_tiebreaks.parquet

---

### 6. Construir analytics player matches

python ingestion/build_analytics_player_matches.py

Output:

data/parquet/analytics_player_matches.parquet

---

### 7. Construir métricas de jugador

python ingestion/build_player_metrics.py

Output:

data/parquet/player_metrics.parquet

---

# PIPELINE COMPLETO

python ingestion/update_tml_data.py

python ingestion/build_all_matches.py

python ingestion/build_master_matches.py

python ingestion/build_analytics_matches.py

python ingestion/enrich_tiebreaks.py

python ingestion/build_analytics_player_matches.py

python ingestion/build_player_metrics.py

---

# SINGLE SOURCE OF TRUTH

## Match Data

data/parquet/master_matches.parquet

## Analytics

data/parquet/analytics_matches.parquet

data/parquet/analytics_player_matches.parquet

## Features

data/parquet/player_metrics.parquet

Estas fuentes deberán consumir:

- Match Comparator
- Match Edges
- Dashboard
- Future Prediction Engine

