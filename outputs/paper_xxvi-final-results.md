# Paper XXVI final confirmation

## Phase classification

- No-fitted-boundary classification: 55/56 (0.982)
- Original regime N=20,h=0.15: P(k*>1)=0.015, predicted ladder=False
- Ladder regime N=1000,h=0.15: P(k*>1)=0.997, predicted ladder=True
- High-h regime N=1000,h=0.80: P(k*>1)=0.215, predicted ladder=False

Misclassified cells:
- N=500, h=0.60: P(k*>1)=0.543, predicted ladder=False

## Hazard validation

- Overall held-out exponential survival MAE: 0.0538
- Peak median pointwise MAE: 0.119 at theta=-1.05

## Sweep composition

- Population-conditional theta MAE: 0.0029
- New-population median theta MAE: 0.0050

| Dwell | Frailty-corrected median | Prediction IQR | Median-hazard audit | Observed median |
|---:|---:|---:|---:|---:|
| 10 | -1.130 | [-1.130, -1.120] | -1.130 | -1.130 |
| 30 | -1.100 | [-1.100, -1.090] | -1.100 | -1.100 |
| 100 | -1.060 | [-1.067, -1.060] | -1.060 | -1.070 |
| 300 | -1.040 | [-1.050, -1.040] | -1.040 | -1.050 |

## Registered checks

- P1 phase reproduction: PASS
- P2 anchor regimes: PASS
- P3 hazard validation: PASS
- P4 sweep composition: PASS
- P5 dwell ordering: PASS

All claims are [R within the model].