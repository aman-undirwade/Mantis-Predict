# Results

Generated experiment outputs belong in `results/generated/` and should be produced by `src/run_experiment.py` after the NASA data is placed in `data/raw/`.

Expected outputs:

- `model_comparison.csv` — model/horizon comparison
- `official_test_predictions.csv` — one prediction per official test engine
- `official_test_summary.csv` — final metrics
- `random_forest_fd004.joblib` — serialized selected model

`reported_results.md` contains the metrics already documented in the final report.
