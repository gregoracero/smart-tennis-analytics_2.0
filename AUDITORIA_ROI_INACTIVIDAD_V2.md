
# Smart Tennis Analytics 2.0
# Auditoria ROI, Edge e Inactividad
# Fecha: 2026

---

# 1. Objetivo inicial

Validar si el modelo realmente:

- supera al mercado
- genera edge real
- produce ROI sostenible
- est? libre de leakage evidente

---

# 2. Hallazgo inicial

Dataset analizado:

    atp_matches_with_predictions_nearest.parquet

Resultados iniciales:

    Accuracy modelo ? 83%
    ROI modelo ? +40%
    ROI edge >20% ? +100%

Los resultados parec?an excesivamente buenos.

---

# 3. Auditor?a de duplicados

Comprobaci?n:

    ROWS                4124
    UNIQUE_MATCH_KEYS   2062

Conclusi?n:

    El dataset contiene doble orientaci?n.

Ejemplo:

    A vs B
    B vs A

Hallazgo:

    Exist?a duplicaci?n.

---

# 4. Recalculo sobre partidos ?nicos

Metodolog?a:

    Mantener una ?nica fila por partido.

Resultado:

    UNIQUE MATCHES = 2062

    Accuracy ? 83%
    ROI ? +40%

Conclusi?n:

    El ROI no desaparece al eliminar duplicados.

---

# 5. Sospecha de leakage

Se detect?:

    P(modelo > 90%) = 908 partidos

    Win rate = 100%

Parec?a incompatible con una validaci?n real.

---

# 6. Auditor?a del pipeline de predicci?n

Archivo:

    betting/generate_model_probabilities_aligned.py

Proceso observado:

    train_df = training_matches_with_elo

    model.fit()

    predict sobre hist?rico

Conclusi?n:

    Estos parquets no deben usarse para validar ROI.

---

# 7. Validaci?n temporal

Script:

    betting/backtest_roi_temporal.py

Resultados:

    EDGE > 3%  ROI +34.5%
    EDGE > 5%  ROI +41.0%
    EDGE > 8%  ROI +52.1%
    EDGE >10%  ROI +58.7%

Conclusi?n:

    El ROI persiste fuera de muestra.

---

# 8. Walk Forward Validation

Script:

    ml/walk_forward_roi_elo_readiness.py

Resultados:

    2021 +41.99%
    2022 +46.09%
    2023 +42.40%
    2024 +36.66%
    2025 +49.46%

ROI medio:

    +43.32%

Conclusi?n:

    Se?al extremadamente estable.

---

# 9. Comparaci?n mercado vs modelo

Partidos ?nicos:

    6687

Resultado:

    MODEL ACCURACY  = 77.70%
    MARKET ACCURACY = 67.47%

    MODEL ROI  = +28.13%
    MARKET ROI = -4.80%

Conclusi?n:

    El modelo supera claramente al mercado.

---

# 10. Investigaci?n del edge

ROI por edge:

    0-5%     ROI negativo
    5-10%    ROI ~0
    15-20%   ROI +8%
    20-25%   ROI +15%
    25-30%   ROI +30%
    >30%     ROI +102%

Conclusi?n:

    Toda la rentabilidad proviene de edges muy altos.

---

# 11. Perfil de los top edges

Hallazgo:

    97% de los top edges corresponden a underdogs.

Promedio:

    Cuota ? 2.84

    Mercado ? 30%

    Modelo ? 93%

Win rate:

    ~94.5%

Conclusi?n:

    Los picks rentables son underdogs.

---

# 12. Hip?tesis Elo

Se analizan:

    delta_elo
    delta_surface_elo

Resultado:

    Top edges presentan Elo peor.

Media:

    delta_elo = -74

Conclusi?n:

    Elo NO explica la ventaja.

---

# 13. Hip?tesis forma reciente

Se analizan:

    delta_win_pct_5
    delta_win_pct_10
    delta_win_pct_20
    delta_win_pct_50

Resultado:

    Tambi?n negativos.

Conclusi?n:

    La forma reciente NO explica la ventaja.

---

# 14. Importancia de variables

Top importances:

    delta_surface_inactivity_days  24.4%
    delta_elo                       8.6%
    delta_inactivity_days           8.5%
    days_inactive_a                 7.2%
    days_inactive_b                 7.1%

M?s del 50% de la se?al:

    Variables de inactividad.

---

# 15. Experimento sin inactividad

Variables eliminadas:

    days_inactive_*
    surface_days_inactive_*
    delta_inactivity_days
    delta_surface_inactivity_days

Resultado:

    ROI = -1.49%

    Hit Rate = 44.8%

Conclusi?n:

    La inactividad explica pr?cticamente toda la ventaja.

---

# 16. Auditor?a de fechas

Descubrimiento cr?tico:

En build_training_matches.py

    match_date = tourney_date

Es decir:

    NO existe fecha real de partido.

Solo:

    fecha inicio torneo.

---

# 17. Evidencia emp?rica

Fechas con m?s partidos:

    2022-01-09 ? 534
    2021-02-14 ? 504
    2024-07-29 ? 426

Imposible si fueran fechas reales.

Conclusi?n:

    match_date = fecha semanal ATP.

---

# 18. Consecuencia

days_inactive y surface_days_inactive se calculan usando:

    tourney_date

No usando:

    fecha real partido.

Por tanto miden:

    actividad entre torneos

y NO:

    d?as reales entre partidos.

---

# 19. Tennis Data

Se observa que Tennis-Data s? contiene:

    Date

Fecha real del encuentro.

Cobertura:

    2001+

Incluye:

    Date
    Winner
    Loser
    Surface
    Odds

---

# 20. Decisi?n arquitect?nica

Construir:

    match_master_v2.parquet

Fuentes:

    Tennis-Data
    TML
    Ongoing

Periodo:

    2010-2026

---

# 21. Nuevo dise?o

Capa 1

    match_master_v2

Campos:

    date_real
    surface
    tournament
    winner
    loser

---

Capa 2

    player_activity_engine

Generar:

    days_inactive_real
    surface_days_inactive_real

usando fechas reales.

---

Capa 3

    training_matches_v2

Recalcular:

    Elo
    Surface Elo
    Ranking
    Form
    Inactividad real

---

# 22. Pr?ximo experimento

Comparar:

Modelo actual
    Inactividad mediante tourney_date

vs

Modelo V2
    Inactividad mediante Date real

Objetivo:

    determinar qu? parte del ROI es real
    y qu? parte procede del calendario ATP semanal.

---

# Conclusi?n ejecutiva

Se confirma:

- El modelo supera al mercado.
- El ROI depende casi completamente de variables de inactividad.
- Las variables de inactividad se calculan con tourney_date.
- tourney_date no es fecha real de partido.
- La siguiente evoluci?n correcta del proyecto es construir un dataset maestro basado en fechas reales Tennis-Data + TML para regenerar todas las features temporales.
