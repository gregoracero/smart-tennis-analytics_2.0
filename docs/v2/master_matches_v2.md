# Master Matches V2

## Objetivo

Construir el Single Point Of Truth del proyecto Smart Tennis Analytics.

Output:

data/parquet/v2/master_matches_v2.parquet

---

## Inputs

### Raw Sources

- tennis_data_co_uk_v2.parquet
- tml_v2.parquet

### Mapping Sources

- player_mapping_v2.parquet
- tournament_mapping_v2.parquet

### Audit Source

- match_link_audit_v2.parquet

---

## Registros incluidos

Estados incluidos:

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

## Identidad Canónica

### Torneo

- tournament_id
- tournament_key

Procedente de tournament_mapping_v2.

### Jugadores

- winner_player_key
- loser_player_key

Procedente de player_mapping_v2.

---

## Match Linking

El matching NO se realiza en este proceso.

Se consume directamente:

match_link_audit_v2.parquet

como fuente oficial de correspondencia entre Tennis Data y TML.

---

## Valid Multiple Match

Los casos VALID_MULTIPLE_MATCH representan partidos legítimos donde existen múltiples encuentros válidos entre los mismos jugadores dentro del mismo torneo y temporada.

Ejemplos:

- ATP Finals
- Masters Cup
- Round Robin + Final

Estos registros se materializan individualmente utilizando:

- tml_round
- tml_score

para identificar el candidato exacto.

---

## Master Match Id

Generado mediante SHA1:

year
tournament_key
winner_player_key
loser_player_key
tml_round
tml_score

Objetivo:

- estabilidad
- reproducibilidad
- unicidad

Resultado validado:

ROWS   : 68.220
UNIQUE : 68.220

---

## Estructura

### Canonical Layer

- master_match_id
- tournament_id
- tournament_key
- winner_player_key
- loser_player_key

### Audit Layer

- match_link_status
- candidate_matches

### Tennis Data Layer

Todas las columnas originales con prefijo:

td_

Ejemplos:

- td_Tournament
- td_Surface
- td_Winner
- td_Loser

### TML Layer

Todas las columnas originales con prefijo:

tml_

Ejemplos:

- tml_tourney_name
- tml_round
- tml_score
- tml_winner_rank

---

## Calidad de Datos

Master Matches V2

- Rows: 68.220
- Columns: 124
- Unique Match IDs: 68.220

Duplicados eliminados mediante master_match_id.

---

## Arquitectura

Raw Sources
+-- Tennis Data
+-- TML

Mappings
+-- Player Mapping
+-- Tournament Mapping

Audit
+-- Match Link Audit

Master Layer
+-- Master Matches V2

---

## Estado

? tennis_data_co_uk_v2

? tml_v2

? player_mapping_v2

? tournament_mapping_v2

? match_link_audit_v2

? master_matches_v2

Master Layer finalizada.
