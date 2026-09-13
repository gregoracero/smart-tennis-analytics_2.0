# 4. Reconciliación de identidades y calidad

## Objetivo

TennisMyLife y Sackmann no usan necesariamente el mismo identificador de jugador. El proceso crea IDs estables y consolida automáticamente los casos de alta confianza.

## Resultado

```text
Mapeos canónicos aplicados       20
IDs sintéticos restantes         37
Jugadores únicos             29.600
```

## Criterios automáticos

La reconciliación compara:

- Nombre normalizado.
- IOC, cuando está disponible.
- Edad compatible.
- Historial y número de apariciones.
- Margen frente al segundo candidato.
- ID candidato no sintético.

La consolidación automática exige una evidencia conservadora. Los casos ambiguos permanecen sintéticos.

## Mojibake y alias

Se reparan secuencias como:

```text
NuÃ±ez -> Nuñez
```

Se admiten alias explícitos y auditables, por ejemplo:

```text
Aleksandr Shevchenko -> Alexander Shevchenko
Faris Zakaria        -> Fares Zakaria
```

## IDs sintéticos

Los IDs sintéticos se sitúan a partir de:

```text
90.000.000
```

La deuda de reconciliación queda registrada en:

```text
remaining_synthetic_players.csv
synthetic_player_candidates.csv
```

El test mostró peor rendimiento cuando participa un ID sintético, por lo que Streamlit debe mostrar una advertencia de cobertura.

## Rondas reclasificadas

Dos filas `R128` encontradas en el archivo qualifying fueron reclasificadas:

```text
ATP_QUALIFYING -> ATP
```

Resultado final TML:

```text
ATP                 2.132
ATP_QUALIFYING      1.103
```

## Estadísticas de calidad

Se corrigieron 848 valores TML codificados como cero:

```text
424 non_positive_w_SvGms
423 non_positive_l_SvGms
  1 non_positive_w_svpt
```

No se detectaron inconsistencias críticas en el postproceso final. Los ceros se transformaron en valores ausentes para impedir ratios artificiales.

## Resultado reconciliado

```text
output_rows                    971.080
output_columns                     119
unique_logical_matches         971.080
duplicate_operational_keys           0
canonical_mappings_applied          20
remaining_synthetic_players         37
rounds_reclassified                  2
duplicates_removed_postprocess      13
```
