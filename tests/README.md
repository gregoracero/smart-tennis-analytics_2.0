# Validation Summary

## Walk Forward AUC

Full Model ............. 0.884
Elo + Readiness ........ 0.877
Elo Only ............... 0.840
Readiness Only ......... 0.817
Inactivity Only ........ 0.764
Delta Elo Only ......... 0.715
No Elo ................. 0.714
Rank Only .............. 0.687
Experience Only ........ 0.570

## Walk Forward ROI

Full Model ............. ~46%
Elo + Readiness ........ 43%
Elo Only ............... 41%
Readiness Only ......... 31%
Experience Only ........ -9%

## Key Findings

- Elo is the primary signal.
- Inactivity is the second strongest signal.
- Experience contributes almost no value.
- Edge quality increases monotonically with edge threshold.
- Calibration remains excellent even above 0.95 probability.
