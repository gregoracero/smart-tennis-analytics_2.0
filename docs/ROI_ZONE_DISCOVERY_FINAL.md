# ROI Zone Discovery Final

## Objective

Identify the exact zones where Core Model V1 generates the majority of its economic edge.

---

# Core Model V1

Features

1. delta_surface_inactivity_days
2. delta_elo
3. delta_inactivity_days
4. days_inactive_a
5. days_inactive_b

---

# Main Finding

The largest ROI does not appear when a player has the highest Elo.

The largest ROI appears when:

- The player is more active.
- The player is significantly more active on the current surface.
- The market still discounts the player because of lower Elo.

This creates situations where:

Readiness Advantage

overcomes

Historical Quality Disadvantage.

---

# Premium ROI Zone

Conditions

delta_elo < 0

and

delta_surface_inactivity_days < -14

Interpretation

Player appears weaker according to Elo.

Player is substantially more active on the relevant surface.

---

## Results

delta_elo < -200

delta_surface_inactivity_days -30:-14

BETS

94

Hit Rate

61.70%

ROI

178.59%

Average Odds

5.21

---

delta_elo < -200

delta_surface_inactivity_days -14:-7

BETS

93

Hit Rate

58.06%

ROI

126.81%

Average Odds

4.90

---

delta_elo -200:-100

delta_surface_inactivity_days -30:-14

BETS

81

Hit Rate

85.19%

ROI

146.67%

Average Odds

2.97

---

delta_elo -200:-100

delta_surface_inactivity_days -14:-7

BETS

128

Hit Rate

69.53%

ROI

98.80%

Average Odds

3.17

---

# Strong ROI Zone

Conditions

delta_elo between -100 and 0

and

delta_surface_inactivity_days < -7

Results

delta_elo -100:0

delta_surface_inactivity_days -30:-14

BETS

107

ROI

101.59%

Hit Rate

83.18%

---

delta_elo -100:0

delta_surface_inactivity_days -14:-7

BETS

166

ROI

102.60%

Hit Rate

83.13%

---

# Neutral Zone

Conditions

delta_surface_inactivity_days between -7 and 0

Interpretation

Activity advantage still exists but is less powerful.

ROI becomes much less consistent.

---

# Avoid Zone

Conditions

delta_surface_inactivity_days > 7

Interpretation

Player is less active than opponent.

Even substantial Elo advantages often fail to compensate.

Examples

delta_elo 100:200

activity 0:7

ROI

-43.21%

---

delta_elo 100:200

activity 7:14

ROI

-45.89%

---

delta_elo < -200

activity 14:30

ROI

-100%

---

# Strategic Interpretation

Elo answers:

Who is the better player?

Readiness answers:

Who is more prepared to compete today?

The investigation suggests that markets price Elo relatively efficiently.

The largest market inefficiencies appear when:

- Elo favors one player.
- Activity favors the other player.
- Markets underestimate the impact of readiness.

---

# Updated Project Thesis

Original Thesis

Elo is the primary source of predictive power.

---

New Thesis

Competitive Readiness is the dominant source of economic edge.

Elo provides context.

Readiness determines whether a player is capable of performing at the level implied by Elo.

---

# Practical Betting Framework

Highest Priority

delta_surface_inactivity_days < -14

and

delta_elo < 0

---

Medium Priority

delta_surface_inactivity_days < -7

and

delta_elo <= 100

---

Avoid

delta_surface_inactivity_days > 7

regardless of Elo profile.

---

# Final Conclusion

The strongest opportunities discovered in the project are:

Underdog
+
Negative Elo Difference
+
Large Surface Activity Advantage

This profile consistently generates the highest ROI observed throughout the investigation and appears to represent the primary economic engine of Core Model V1.
