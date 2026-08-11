# Reported Results

The values below are the results documented in the submitted capstone report. They are recorded here for transparency and are not hard-coded into the experiment script.

## Official FD004 Test Result

| Measure | Reported value |
|---|---:|
| Test engines | 248 |
| Selected warning horizon | 50 cycles |
| Selected probability threshold | 0.10 |
| ROC-AUC | 0.9738 |
| PR-AUC | 0.9518 |
| Precision | 0.6475 |
| Recall | 0.9875 |
| F1-score | 0.7822 |
| True negatives | 125 |
| False positives | 43 |
| False negatives | 1 |
| True positives | 79 |

## Interpretation

The model ranked engine-failure risk strongly, as shown by the ROC-AUC and PR-AUC values. The operating threshold then determined how many engines were actually flagged. At the selected threshold, 79 of 80 positive test engines were detected, while one positive engine was missed. The trade-off was 43 false alarms.

This is appropriate to discuss as an operational decision rather than as a universally optimal threshold. In a real maintenance system, the threshold would depend on the relative costs of missed failures, inspections, downtime, maintenance capacity, and safety requirements.

## Explainability

The report's SHAP analysis identified the following top features by mean absolute SHAP value:

| Rank | Feature | Mean absolute SHAP |
|---:|---|---:|
| 1 | s13 | 0.1210 |
| 2 | s11 | 0.0625 |
| 3 | s4 | 0.0567 |
| 4 | s15 | 0.0531 |
| 5 | s14 | 0.0480 |
| 6 | s8 | 0.0405 |
| 7 | s3 | 0.0374 |
| 8 | s9 | 0.0354 |
| 9 | s21 | 0.0310 |
| 10 | s12 | 0.0296 |

## Limitations

- FD004 is simulated rather than unrestricted real aircraft maintenance data.
- The official benchmark contains 248 test engines, so performance is benchmark-specific.
- The pseudo-test validation uses controlled RUL targets rather than an independent operational validation campaign.
- The 5:1 false-negative cost ratio is illustrative; real maintenance costs should be supplied by domain experts.
- The project focuses on classical DATA110 models rather than LSTM, GRU, Transformer, or other sequence architectures.
