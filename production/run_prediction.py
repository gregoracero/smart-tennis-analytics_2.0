
from services.prediction_service import (
    predict_match
)

result = predict_match(

    player_a_id="S0S1",

    player_b_id="N0AE",

    tour="atp",

    surface="hard",

    match_date="2026-08-16"
)

print()

print("RESULT")

print(result)
