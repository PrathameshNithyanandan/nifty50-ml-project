# Unit 4 and Unit 5 algorithms compared

Written by `compare_units_4_5.py` from `results/results.json`. Every model forecasts the same 822 RELIANCE test days. All of them train on days before the test period only.

## Next-day direction

The baseline is always predicting the majority class, which is right 51.95% of the time. An AUC between 0.460 and 0.540 is within the chance band.

| Rank | Algorithm | Unit | Accuracy | AUC | p against majority |
|---:|---|---:|---:|---:|---:|
| 1 | HMM, filtered probabilities | 4 | 0.5207 | 0.4970 | 0.486 |
| 2 | Majority class | baseline | 0.5195 | n/a | 0.514 |
| 3 | Gradient Boosting | 5 | 0.5195 | 0.5039 | 0.514 |
| 4 | BMA (logistic subsets) | 5 | 0.5134 | 0.5145 | 0.650 |
| 5 | Same as today | baseline | 0.5085 | 0.5078 | 0.746 |
| 6 | Stacking (logistic meta) | 5 | 0.5061 | 0.4778 | 0.789 |
| 7 | AdaBoost | 5 | 0.5024 | 0.4777 | 0.844 |
| 8 | Random Forest | 5 | 0.5000 | 0.4930 | 0.875 |
| 9 | Bagging | 5 | 0.4988 | 0.5146 | 0.889 |
| 10 | GAM (LogisticGAM) | 5 | 0.4988 | 0.4969 | 0.889 |
| 11 | CART (pruned) | 5 | 0.4696 | 0.4667 | 0.998 |

**Result:** Algorithms above the majority rate: 1 of 9. Significantly above it at the 5% level: **0 of 9**. The highest accuracy is HMM, filtered probabilities (Unit 4) at 52.07%, and the highest AUC is Bagging (Unit 5) at 0.5146. Neither unit's algorithms show a dependable edge on this stock.

The HMM forecast uses filtered probabilities, which only look at past days. The smoothed version scores 52.55%, but it uses future days, so it is left out of the ranking.

## Next-day closing price

Only Unit 4 forecasts the price: the Unit 5 models predict direction. So the Kalman filter is compared with persistence, which means using today's close as tomorrow's forecast.

| Model | RMSE | MAE | Diebold–Mariano p against persistence |
|---|---:|---:|---:|
| Persistence | 30.3051 | 20.8155 | baseline |
| Local level (MLE) | 30.3051 | 20.8155 | 0.982 |
| Local linear trend (MLE) | 30.2925 | 20.8011 | 0.756 |
| Local level, q = 0.01 (fixed by hand) | 73.8945 | 52.4919 | 0.000 |

**Result:** neither fitted Kalman model is significantly different from persistence (p = 0.982 and 0.756; RMSE 30.305 and 30.293 against 30.305). Maximum likelihood sets the local-level Kalman gain to 1.0000, which makes that model almost the same as persistence. The filter with a gain fixed by hand looks smoother but forecasts far worse.
