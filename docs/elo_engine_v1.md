# ATP Elo Engine V1

## Estado

Production Ready

Fecha de congelación:

ELO Engine V1

---

## Objetivo

Construir un motor Elo histórico para ATP que permita generar variables predictivas para los modelos de Smart Tennis Analytics 2.0.

---

## Dataset origen

data/parquet/master_matches.parquet

---

## Limpieza aplicada

### Fechas

Eliminación de registros con fecha inválida.

### Duplicados

Clave utilizada:

- tourney_date
- winner_id
- loser_id
- tourney_name

---

## Identificación de jugadores

Se utilizan exclusivamente:

- winner_id
- loser_id

Nunca nombres.

---

## Elo General

### Configuración

```python
INITIAL_ELO = 1500
BASE_ELO = 1500
```

### Expected Score

```python
1 / (
    1 +
    10 ** (
        (elo_b - elo_a) / 400
    )
)
```

---

## Decay Global

### Configuración

```python
HALF_LIFE_DAYS = 730
```

### Objetivo

Reducir la confianza sobre jugadores con largos periodos de inactividad.

---

## Surface Elo

Superficies soportadas:

- Hard
- Clay
- Grass
- Carpet

### Configuración

```python
SURFACE_HALF_LIFE_DAYS = 3650
```

### Objetivo

Mantener conocimiento histórico sobre una superficie específica durante periodos prolongados.

---

## K Factor

### Grand Slam

```python
40
```

### Masters

```python
32
```

### ATP 500

```python
28
```

### ATP 250 y resto

```python
24
```

---

## Dynamic K

```python
if max_days > 180:
    k *= 2

elif max_days > 70:
    k *=*1.5
```

---

## Variables generad*s

### Elo*
- winner_elo_before
- loser_elo_b*fore

- winner_surface*elo_before
- loser_surface_elo_bef*re

### Elo almacenado

- winner_s*ored_elo
- loser_stored_elo

- win*er_surface_stored_elo
- loser_surf*ce_stored_elo

### Actividad

- wi*ner_days_inactive
- loser_days_ina*tive

- winner_surface_days_inacti*e
- loser_surface_days_inactive

#*# Experiencia

- winner_matches_pl*yed
- loser_matches_played

- winn*r_surface_matches_played
- loser_s*rface_matches_played

### Flags

-*winner_is_new_player
- loser_is_ne*_player

---

## Artefacto generad*

```text
data/parquet/player_elo_*istory.parquet
```

---

## Result*dos finales

### Partidos procesad*s

```text
366.065
```

### Jugado*es únicos

```text
11.683
```

###*Overall Elo

```*ext
Winner Mean = 1733
*oser Mean  = 1666

Winner Std  = 1*5*Loser Std   = 165

Max*Elo     = 2639
```

### Surface El**
#### Hard

```text
Std = 73
Max =**340
```

#### Clay

```text
Std = *1
Max = 2175
```

#### Grass

```t*xt*Std = 65
Max = 2126
```

####*Carpet

```text
Std = 55
Max = 195**```

---

## Decisiones de diseño
*### Overall Elo

Incl*ye:

- Decay
- Dynamic K*- Inactividad
- Experience

### Su*face Elo

Incluye:

- Rating indep*ndiente
- Surface Decay
- Surface *xperience
- Surface Inactivity

##* New Players

Los jugadores sin hi*torial previo reciben:

```python
*ays_inactive = -1
```

y generan*

- winner_is_new_player
- loser_i*_new_player*
---

## Próximas Features

- delt*_elo*- delta_surface_*lo

- delta_matches_played
- delta*surface_matches_played

- delta_in*ctivity_days
- delta_surface_inact*vity_days

---

## Próximo Sprint
*player_elo_history.parquet

↓

Mer*e con training_matches

↓

*eature Engineering

↓

XGBoost

↓
*Benchmark ROC AUC

Objetivo inicia*:

```text
Superar 0.7362
```

---*
## Vers*ón

ATP Elo Engine V1

Estado:

Fr*zen

Ready*for Feature Engineering