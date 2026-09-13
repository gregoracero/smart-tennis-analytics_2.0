# 11. Limitaciones y trabajo pendiente

## Identidades sintéticas

Quedan 37 IDs sintéticos. Algunos corresponden a jugadores con actividad y ranking relevantes. Prioridad:

1. Jugadores con ranking <= 500.
2. Jugadores con muchas apariciones TML.
3. Casos con candidato histórico inequívoco.
4. Variantes de país `URY`/`URU` o nombres sin IOC.

## Rendimiento TML

El rendimiento TML es inferior al subconjunto Sackmann. Se debe analizar por:

- Competición.
- Experiencia mínima.
- Cobertura estadística.
- ID sintético.
- Periodo temporal.
- Calidad de nombres e IOC.

## Modelo con mercado

La mejora frente al mercado no-vig es pequeña y el test contiene 1.618 filas. Sigue siendo experimental hasta acumular más datos o demostrar estabilidad con bootstrap.

## Calibración

Platt mejora marginalmente calibration, pero no mejora el test no market. Se usa una política conservadora y se selecciona `raw`.

## Orientación

La versión entrenada usa ID menor como `player_1`. Es reproducible, pero no intrínsecamente neutral. Una futura v3 podría usar un hash determinista independiente del ID. Ese cambio requeriría regenerar features, datasets, modelos y adaptar inferencia.

## Fecha de partido

Para muchos partidos históricos `tourney_date` representa el inicio del torneo, no la fecha exacta del encuentro. El orden de rondas reduce el problema, pero no lo elimina completamente.

## Pruebas automatizadas

Pendiente crear tests unitarios y de integración para:

- Orden temporal.
- No actualización por walkover.
- Simetría de inferencia.
- Esquema de features.
- Migración del calibrador.
- Escritura atómica.
