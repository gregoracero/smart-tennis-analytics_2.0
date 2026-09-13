# TOURNAMENT STATISTICS V2 - DOCUMENTACION TECNICA

## Objetivo

Construir una capa analítica agregada por torneo y temporada.

Granularidad:

tournament_id
+
season

Una fila representa:

Un torneo específico en una temporada específica.

Ejemplo:

Cincinnati ATP 2026

Roland Garros 2024

Rotterdam 2016

---

# Fuente de Datos

Input:

player_per_match_statistics_v2.parquet

Motivo:

Es la tabla analítica más granular disponible.

Contiene:

- Jugadores
- Resultados
- Rankings
- Estadísticas de servicio
- Duración
- Cuotas de apuestas

Sin necesidad de realizar joins adicionales.

---

# Diseño

Agrupación principal:

groupby(

    tournament_id,
    tournament_key,
    tournament_name,
    season,
    surface,
    tournament_level

)

---

# Métricas de Participación

## Matches

Número de partidos únicos.

matches

Calculado mediante:

master_match_id.nunique()

---

## Players

Número de jugadores únicos.

players

Calculado mediante:

player_id.nunique()

---

# Métricas de Ranking

## Average Rank

avg_rank

Ranking medio de los participantes.

---

## Median Rank

median_rank

Ranking mediano de los participantes.

---

## Top Players

Jugadores únicos dentro del torneo:

top10_players

top20_players

top50_players

top100_players

Importante:

Se utilizan player_id únicos.

No se cuentan apariciones repetidas.

---

# Estadísticas de Juego

## Aces

total_aces

aces_per_match

---

## Double Faults

total_double_faults

double_faults_per_match

---

## Match Duration

total_minutes

avg_minutes

---

## Sets

avg_sets_per_match

Calculado mediante:

total_sets
/
matches

---

# Betting Model

## Normalización de Cuotas

Todas las cuotas se convierten a numéricas.

player_b365

opponent_b365

mediante:

pd.to_numeric()

---

## Favorite Definition

Un jugador se considera favorito cuando:

player_b365
<
opponent_b365

Es decir:

Cuota menor
=
Mayor probabilidad implícita
=
Favorito

---

## Favorite Mask

favorite_mask

Representa:

Jugador favorito.

---

## Underdog Mask

underdog_mask

Representa:

Jugador no favorito.

---

# Cobertura de Mercado

## Betting Matches

betting_matches

Partidos con cuotas disponibles para ambos jugadores.

---

## Betting Coverage

betting_coverage_pct

Calculado como:

betting_matches
/
matches

Permite identificar:

- Torneos históricos con poca cobertura
- Torneos modernos con cobertura completa

Resultados obtenidos:

mean ≈ 91.6%

median = 100%

---

# Favorite Statistics

## Favorite Matches

favorite_matches

Número de participaciones de favoritos.

---

## Favorite Wins

favorite_wins

Número de victorias obtenidas por favoritos.

---

## Favorite Win Percentage

favorite_win_pct

Resultado observado:

mean ≈ 68.5%

Valor consistente con mercados ATP reales.

---

# Underdog Statistics

## Underdog Matches

underdog_matches

---

## Underdog Wins

underdog_wins

---

## Underdog Win Percentage

underdog_win_pct

Resultado observado:

mean ≈ 32.4%

Valor consistente con mercados ATP reales.

---

# Betting ROI

## Favorite Profit

favorite_profit

Resultado acumulado de apostar:

1 unidad

a cada favorito.

---

## Favorite ROI

favorite_roi

Calculado como:

favorite_profit
/
favorite_matches

Resultado observado:

mean ≈ -5.7%

Consistente con margen de la casa.

---

## Underdog Profit

underdog_profit

Resultado acumulado de apostar:

1 unidad

a cada underdog.

---

## Underdog ROI

underdog_roi

Resultado observado:

mean ≈ -9.5%

También coherente con mercados eficientes.

---

# Validaciones Realizadas

## Match Distribution

matches

count = 1649

mean ≈ 40

median ≈ 31

max = 127

Resultado:

Correcto para:

- ATP 250
- ATP 500
- Masters 1000
- Grand Slams

---

## Player Distribution

players

median ≈ 32

max = 128

Resultado correcto.

---

## Average Rank

avg_rank

mean ≈ 80

Resultado coherente.

---

## Aces Per Match

aces_per_match

mean ≈ 11.4

Resultado coherente.

---

## Favorite Win Rate

favorite_win_pct

mean ≈ 68.5%

Resultado coherente.

---

## Underdog Win Rate

underdog_win_pct

mean ≈ 32.4%

Resultado coherente.

---

## Betting Coverage

betting_coverage_pct

mean ≈ 91.6%

Resultado muy satisfactorio.

---

# Dataset Resultante

tournament_statistics_v2.parquet

Incluye:

- Participación
- Rankings
- Servicio
- Duración
- Sets
- Mercado
- ROI
- Cobertura

---

# Próximas Mejoras Potenciales

## Champion

Nombre del campeón.

---

## Runner Up

Finalista.

---

## Tiebreak Statistics

tiebreaks_played

tiebreaks_per_match

---

## Elo Strength

Integración futura con:

elo_history_v2

para obtener:

avg_field_elo

median_field_elo

max_field_elo

---

# Conclusión

Tournament Statistics V2 proporciona una vista analítica completa de cada torneo y temporada.

Permite:

✅ Analizar la fuerza del cuadro.

✅ Analizar estadísticas de juego.

✅ Analizar comportamiento del mercado.

✅ Calcular ROI histórico.

✅ Alimentar dashboards de torneo.

✅ Servir como base para capas futuras de prediction y reporting.

