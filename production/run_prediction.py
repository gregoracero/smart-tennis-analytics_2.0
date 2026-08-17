
from services.player_lookup_service import (
    get_player_id
)

from services.prediction_service import (
    predict_match
)

player_a = "Carlos Alcaraz"

player_b = "Jannik Sinner"

player_a_id = get_player_id(
    player_a
)

player_b_id = get_player_id(
    player_b
)

result = predict_match(

    player_a_id=player_a_id,

    player_b_id=player_b_id,

    tour="atp",

    surface="hard",

    match_date="2026-08-16"
)

print()

print(player_a)
print(player_b)

print()

print(result)
