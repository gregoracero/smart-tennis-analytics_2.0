# Master Matches V2

## Objetivo

Construir el dataset maestro del proyecto:

master_matches_v2.parquet

Este dataset será el Single Point Of Truth de Smart Tennis Analytics.

---

## Inputs

### Raw Sources

- tennis_data_co_uk_v2.parquet
- tml_v2.parquet

### Mapping Sources

- player_mapping_v2.parquet
- tournament_mapping_v2.parquet

### Matching Source

- match_link_audit_v2.parquet

---

## Output

data/parquet/v2/master_matches_v2.parquet

---

## Registros incluidos

Se incluirán únicamente los partidos con estado:

- MATCHED
- VALID_MULTIPLE_MATCH

Estados excluidos:

- NO_MATCH
- DUPLICATE_CANDIDATES
- NO_TML_COVERAGE
- WINNER_UNMATCHED
- LOSER_UNMATCHED
- NO_TOURNAMENT_MATCH

---

## Estrategia

build_master_matches_v2 NO realizará matching.

Consumirá directamente:

match_link_audit_v2.parquet

como fuente oficial de linkado entre Tennis Data y TML.

---

## Identidad Canónica

### Torneo

Procedente de:

tournament_mapping_v2.parquet

Columnas:

- tournament_id
- tournament_key

### Jugadores

Procedente de:

player_mapping_v2.parquet

Columnas:

- winner_player_key
- loser_player_key

Estas claves serán la identidad oficial de jugador dentro del proyecto.

---

## Master Match Id

Generar un identificador estable y reproducible:

master_match_id

Basado en:

- season
- tournament_key
- winner_player_key
- loser_player_key

Implementación recomendada:

SHA1(
  season +
  tournament_key +
  winner_player_key +
  loser_player_key
)

---

## Columnas de Auditoría

Conservar:

- match_link_status
- candidate_matches

---

## Datos Tennis Data

Todas las columnas originales de Tennis Data serán incluidas.

Prefijo:

td_

Ejemplos:

- td_Tournament
- td_Surface
- td_Date
- td_Winner
- td_Loser
- td_WRank
- td_LRank

etc.

---

## Datos TML

Todas las columnas originales de TML serán incluidas.

Prefijo:

tml_

Ejemplos:

- tml_tourney_name
- tml_surface
- tml_winner_name
- tml_loser_name
- tml_winner_rank
- tml_loser_rank
- tml_score

etc.

---

## Principios de Diseño

1. No perder columnas originales.
2. No aplicar reglas de precedencia en V2.
3. Mantener trazabilidad completa.
4. Mantener capacidad de auditoría.
5. Utilizar identificadores canónicos de jugador y torneo.
6. Dataset preparado para Feature Engineering, ELO y Modelos Predictivos.

---

## Flujo

tennis_data_co_uk_v2
                \
                 \
                  --> match_link_audit_v2
                 /
                /

tml_v2

        |
        v

build_master_matches_v2

        |
        v

master_matches_v2.parquet

---

## Pipeline

Fuentes
+-- tennis-data.co.uk
+-- TML

Ingesta
+-- downloader_tennis_data_v2
+-- build_tml_v2
+-- build_tennis_data_co_uk_v2

Normalización
+-- build_tournament_mapping_v2
+-- build_player_mapping_v2

Matching
+-- build_match_link_audit_v2

Master Layer
+-- build_master_matches_v2

---

## Estado Actual del Matching

MATCHED:
68.199

Cobertura:
99,90%

VALID_MULTIPLE_MATCH:
18

Los NO_MATCH restantes son mayoritariamente históricos y no bloquean la construcción del Master Layer.
