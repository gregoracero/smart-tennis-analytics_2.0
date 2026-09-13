# ELO HISTORY V2 - DOCUMENTACION TECNICA

## Objetivo

Construir un sistema ELO ATP inspirado en Tennis Abstract capaz de:

- Mantener ratings históricos coherentes.
- Gestionar múltiples superficies.
- Incorporar inactividad.
- Evolucionar correctamente a lo largo de toda la historia ATP.
- Ser consistente con futuras capas de predicción.

---

# Problema original detectado

Los primeros resultados mostraban:

Mean Elo ≈ 1191

Max Elo ≈ 1541

Federer ≈ 1322

Nadal ≈ 1361

Djokovic ≈ 1350

Valores completamente irreales.

Además:

days_since_last_match

min = -9185

max = 9101

Lo cual era imposible.

---

# Investigación realizada

## Hipótesis 1: Penalización por inactividad excesiva

Se observó:

inactivity_adjusted = True

60.498 veces

sobre:

136.440 registros

Lo que suponía aproximadamente:

44%

de todos los registros.

Inicialmente se pensó que la regresión a la media estaba destruyendo el sistema.

---

## Hipótesis 2: Error en el cálculo de inactividad

Aparecieron casos como:

Ruud      8813 días

Djokovic  7520 días

Federer   6902 días

Lo que parecía indicar:

- Fechas mal ordenadas
- Fechas mal parseadas

---

## Hipótesis 3: Colisiones de player_key

Se auditó player_key en todo el pipeline.

Aparecieron casos reales:

ruud_c

- Casper Ruud
- Christian Ruud

martin_a

- Alberto Martin
- Andrej Martin
- Andres Martin
- Alvaro Lopez San Martin

kuznetsov_a

- Alex Kuznetsov
- Andrey Kuznetsov

---

# Root Cause Encontrada

La causa principal NO era el algoritmo Elo.

La causa principal era:

player_key

como identificador de jugador.

El sistema estaba mezclando carreras completas de jugadores distintos.

Eso provocaba errores en:

- career_matches
- days_since_last_match
- elo_overall
- elo_surface
- K dinámico

---

# Solución Arquitectónica

## Nuevo identificador canónico

Se añadió:

player_id

en:

player_mapping_v2.parquet

Generado a nivel de:

tml_name

Resultado:

PLAYER_IDS = 1637

TML_NAMES = 1637

Es decir:

1 jugador = 1 player_id

---

# Propagación del player_id

## 1. build_player_mapping_v2

Añadido:

player_id

Columnas finales:

- player_id
- player_key
- tennis_data_name
- tml_name
- mapping_method
- confidence

---

## 2. build_tournament_mapping_v2

Sustitución de:

winner_key

loser_key

por:

winner_player_id

loser_player_id

para los cálculos de overlap.

---

## 3. build_match_link_audit_v2

Añadidas columnas:

winner_player_id

loser_player_id

manteniendo:

winner_player_key

loser_player_key

por compatibilidad.

---

## 4. build_master_matches_v2

Añadidas:

winner_player_id

loser_player_id

al dataset maestro.

Además:

master_match_id

pasó a construirse usando:

winner_player_id

loser_player_id

en lugar de:

winner_player_key

loser_player_key

---

## 5. build_player_per_match_statistics_v2

Añadidos:

player_id

opponent_player_id

junto a:

player_key

opponent_player_key

---

## 6. build_player_statistics_v2

Cambio fundamental.

Antes:

groupby(player_key)

Después:

groupby(player_id)

---

## 7. build_elo_history_v2

Toda la lógica Elo migrada a:

player_id

Se sustituyó:

elo_overall[player_key]

career_matches[player_key]

last_match_date[player_key]

por:

elo_overall[player_id]

career_matches[player_id]

last_match_date[player_id]

---

# Mejora de Inactividad

## Versión implementada

Sin penalización durante:

0-120 días

Lógica:

if inactivity_days <= 120:
    return rating

---

## Regresión suave

rating = INITIAL_ELO + (rating - INITIAL_ELO) * exp(
    -0.00035 * (inactivity_days - 120)
)

---

## Ajuste por regreso

Se añadió:

return_matches_remaining

Durante:

10 partidos

de regreso.

Aplicando:

K *= 1.5

---

# Bug encontrado durante la migración

El cálculo del K estaba invertido.

Incorrecto:

if return_matches_remaining > 0:

    winner_k *= 1.5

winner_k = dynamic_k(...)

Corregido a:

winner_k = dynamic_k(...)

if return_matches_remaining > 0:

    winner_k *= 1.5

---

# Estado Actual de la Inactividad

Resultados tras la migración:

mean = 20.77 días

median = 5 días

p75 = 14 días

p90 = 35 días

p95 = 77 días

p99 = 357 días

Además:

Negative Days = 0

Lo que confirma:

- Orden temporal correcto
- Fechas correctas
- Identidad correcta

---

# Validación del Sistema Elo

## Antes

Federer = 1322

Nadal = 1361

Djokovic = 1350

Max Elo = 1541

## Después

Federer = 2289

Nadal = 2229

Djokovic = 2381

Max Elo = 2381

---

# Top Históricos Actuales

1. Djokovic    2381
2. Federer     2289
3. Sinner      2273
4. Alcaraz     2271
5. Murray      2235
6. Nadal       2229

Resultados considerados plausibles.

---

# Validación de Superficies

## Nadal

Clay      1981

Hard      1550

Grass     1309

Carpet    1162

Comportamiento correcto:

Clay >>> Hard > Grass > Carpet

---

# Problemas Pendientes

## Caso Zhang

Existe un único conflicto residual.

player_key = zhang_z

para:

- Ze Zhang
- Zhizhen Zhang

Sin impacto funcional porque:

player_id

es ahora la identidad canónica de todo el sistema.

---

## Casos extremos de inactividad

Persisten algunos jugadores con:

- 2000+ días
- 3000+ días
- 8000+ días

Ejemplo:

Martin Damm

Pendiente revisar si se trata de:

- Carrera realmente larga
- Padre/hijo
- Mapping adicional a corregir

---

# Estado Actual del Sistema

✅ player_id implantado en todo el pipeline

✅ player_key pasa a ser únicamente descriptivo

✅ Eliminadas colisiones que contaminaban Elo

✅ Eliminadas inactividades negativas

✅ Career matches calculados correctamente

✅ Elo global consistente

✅ Elo por superficie consistente

✅ Picos históricos razonables

---

# Conclusión

La migración:

player_key → player_id

ha sido el cambio más importante del proyecto Elo V2.

Ha solucionado:

✅ Colisiones de jugadores

✅ Historiales mezclados

✅ Inactividad negativa

✅ Distorsión del K dinámico

✅ Elo comprimido

✅ Picos históricos irreales

Y ha permitido obtener un sistema Elo con valores coherentes y comparables a los observados en Tennis Abstract.

