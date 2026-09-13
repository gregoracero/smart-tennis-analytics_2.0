# Elo History V2

## Objetivo

Construir un sistema de ratings Elo inspirado en Tennis Abstract y Jeff Sackmann.

El objetivo es generar ratings dinámicos de fuerza para cada jugador que puedan ser utilizados en:

- Analytics
- Betting
- Feature Engineering
- Prediction Models

---

# Input

data/parquet/v2/player_per_match_statistics_v2.parquet

---

# Output

data/parquet/v2/elo_history_v2.parquet

---

# Grain

1 fila = 1 jugador en 1 partido

Ejemplo:

Federer

Australian Open 2020

?

elo_before
elo_after

---

# Filosofía

El sistema intenta aproximarse al funcionamiento de Tennis Abstract:

- Elo general
- Elo por superficie
- K dinámico
- Mezcla Elo General + Elo Surface
- Ajuste Best Of 5
- Penalización por inactividad

---

# Rating Inicial

Todos los jugadores comienzan con:

INITIAL_ELO = 1200

Motivos:

- ATP
- Challenger
- Qualifying
- ITF

comparten la misma escala histórica.

---

# Elo Ratings

## Elo General

elo_overall

Se actualiza en todos los partidos.

---

## Elo Hard

elo_hard

Actualiza únicamente en Hard.

---

## Elo Clay

elo_clay

Actualiza únicamente en Clay.

---

## Elo Grass

elo_grass

Actualiza únicamente en Grass.

---

## Elo Indoor

elo_indoor

Actualiza únicamente en Indoor.

---

# Elo Mixto

Para predicción se utilizará:

elo_mix

Fórmula:

elo_mix =
(
    elo_overall
    +
    elo_surface
)
/
2

Es decir:

50% Elo General

50% Elo Superficie

---

# Probabilidad Esperada

Fórmula clásica Elo:

expected =
1
/
(
    1
    +
    10 **
    (
        (
            opponent_elo
            -
            player_elo
        )
        /
        400
    )
)

---

# Resultado Observado

Victoria

result = 1

Derrota

result = 0

No se utilizan:

- sets
- juegos
- margen

Solo importa ganar o perder.

---

# Elo Update

new_elo =
old_elo
+
K
*
(
    result
    -
    expected
)

---

# Dynamic K

El sistema utiliza una aproximación inspirada en Tennis Abstract.

Fórmula:

K =
250
/
(
    (
        matches_played
        +
        5
    )
    **
    0.4
)

Propiedades:

- jugadores nuevos ? cambios rápidos
- jugadores experimentados ? cambios lentos
- transición continua
- sin saltos artificiales

---

# Surface Updates

Partido en Clay

Actualiza:

- elo_overall
- elo_clay

No actualiza:

- elo_hard
- elo_grass
- elo_indoor

---

# Best Of Five Adjustment

Los partidos Best Of Five benefician al favorito.

Primero se calcula:

expected_bo3

Posteriormente:

expected_bo5

utilizando la probabilidad equivalente para ganar tres sets antes que el rival.

Objetivo:

Aproximar el comportamiento de los Grand Slams masculinos.

---

# Inactivity Adjustment

El sistema incorpora regresión a la media para jugadores inactivos.

Se calcula:

days_since_last_match

---

## Activación

Se aplica cuando:

days_since_last_match > 90

---

## Fórmula

elo_adjusted =

1200

+

(
    elo_current
    -
    1200
)

*

exp(
    -0.0005
    *
    inactivity_days
)

---

## Objetivo

Reducir la confianza en ratings antiguos:

- lesiones
- parones prolongados
- inactividad estacional

---

# Output Columns

## Match Context

master_match_id

match_date

surface

result

player_key

opponent_player_key

---

## Experience

career_matches_before

k_factor

days_since_last_match

inactivity_adjusted

---

## Overall Elo

elo_overall_before

elo_overall_after

---

## Surface Elo

elo_surface_before

elo_surface_after

---

## Hard Elo

elo_hard_before

elo_hard_after

---

## Clay Elo

elo_clay_before

elo_clay_after

---

## Grass Elo

elo_grass_before

elo_grass_after

---

## Indoor Elo

elo_indoor_before

elo_indoor_after

---

## Mixed Elo

elo_mix_before

---

## Match Probability

expected_win_probability

---

# Validation Targets

Federer

Peak Elo Overall > 2200

---

Djokovic

Peak Elo Overall > 2250

---

Nadal

Peak Elo Clay > 2300

---

Isner

Elo Grass >> Elo Clay

---

Alcaraz

Elo Clay > Elo Hard
durante buena parte de su carrera

---

# Usage

Analytics

?

player_statistics_v2

---

Prediction

?

player_snapshots_v2

?

training_matches_v2

---

Betting

?

value detection

market pricing

yield studies

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

? Pendiente implementación

? Pendiente validación

? Pendiente calibración histórica

