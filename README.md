# Flight Delay Risk & Duration Prediction

Predicts departure delay for US domestic flights three ways: whether a flight
leaves more than 15 minutes late (the D15 metric airlines report on), which of
four severity bands it lands in, and the delay in minutes. Built on the full
2019 BTS on-time performance file, 1.15M flights after filtering.

The main finding: almost all of the predictive power sits in two engineered
feature groups — **aircraft rotation** (tracking each tail number through its
day) and **airport/network congestion**. Dropping either group nearly halves
regression R2 (0.44 -> ~0.21), and the two are complementary rather than
redundant.

## Results at a glance

| question | model | headline result |
|---|---|---|
| More than 15 min late? | XGBoost classifier | ROC-AUC 0.843 (0.846 cross-val), top-5% precision 0.98 |
| Which severity band? | LightGBM + ordinal | macro-F1 0.56, quadratic kappa 0.55-0.57, macro-AUC 0.84 |
| How many minutes? | XGBoost regressor | R2 0.44, MAE 11.5 min, 83% of errors under 15 min |

All figures are on a time-based Nov-Dec hold-out (trained on Jan-Oct). The
trivial "always on time" baseline scores 0.82 accuracy and 0.23 macro-F1, so
the severity numbers should be read against that floor, not against 0.82
accuracy.

## Data

US BTS *Reporting Carrier On-Time Performance*, 2019, public. Filtered to 5
carriers (WN, DL, AA, OO, UA) and the 20 busiest airports; cancellations
dropped; 1.15M flights. The dataset itself is not included in this repo.

## Notebook layout

`flight_delay_prediction_bts.ipynb`:

1. Setup and data loading
2. Target construction (D15 binary + 4-class severity bands)
3. Baseline features (calendar, route/carrier identity, historical means)
4. Aircraft rotation features (previous-leg delay, turn slack, leg position)
5. Airport/network congestion features (rolling same-day delay state)
6. Exploratory analysis
7. Train/test split (time-based, Jan-Oct vs Nov-Dec) and feature-block ablations
8. Severity classification (direct 4-class + ordinal decomposition)
9. Binary triage (>15 min) with operating-point / lift analysis
10. Regression (minutes), including a two-stage hurdle model
11. Feature importance (gain + SHAP)
12. Validation: rolling-origin CV, calibration, permutation test, learning curve
13. Cost-based decision thresholds
14. Results summary

All history/congestion features are expanding statistics computed only from
prior flights (current row excluded), so the model only ever sees information
that would have been available at prediction time.

## Limitations

- No weather. BTS weather fields are assigned after the fact and would leak;
  a real forecast feed is the obvious next addition.
- Prediction horizon is 1-3 hours — the rotation features need the inbound
  leg to have already landed. Day-ahead prediction would need to chain leg
  by leg and would score lower.
- Top-2% precision is partly mechanical: a plane already hours late will
  leave late again. The more informative range is the middle of the risk
  distribution, where precision runs 0.54-0.64.
- 2019 only, 20 airports, cancellations excluded — results shouldn't be
  assumed to transfer to other years, regional airports, or the post-2020
  network.

## License

MIT — see [LICENSE](LICENSE).
