# DATA110 Capstone — Aircraft Engine Predictive Maintenance

## Project Overview
This repository contains the DATA110 advanced project on early aircraft-engine failure detection using NASA's C-MAPSS FD004 dataset. The project reframes predictive maintenance as a binary early-warning classification problem and compares classical machine-learning models taught in DATA110.

The central research question is: **How early can an engine-failure warning be generated while keeping missed failures acceptably low and controlling false alarms?**

## Advanced Project Requirements Addressed
- Problem formulation and objectives
- NASA C-MAPSS FD004 data preparation
- Engine-aware validation to reduce leakage
- Feature engineering for multivariate sensor trajectories
- Comparison of Logistic Regression, Gaussian Naive Bayes, KNN, Decision Tree, and Random Forest
- Explicit warning-horizon comparison: 10, 20, 30, and 50 cycles
- ROC-AUC, PR-AUC, precision, recall, and F1 evaluation
- Cost-sensitive threshold selection
- Official FD004 test evaluation
- Random Forest explainability with SHAP
- Reproducible code and documentation

## Repository Structure

```text
DATA110-Capstone/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── README.md
├── src/
│   ├── predictive_maintenance.py
│   └── run_experiment.py
├── results/
│   ├── README.md
│   └── reported_results.md
├── figures/
│   └── README.md
├── labs/
│   └── README.md
├── assignments/
│   └── README.md
└── intermediate/
    └── README.md
```

## Dataset
NASA C-MAPSS is a simulated turbofan-engine run-to-failure benchmark. This project uses **FD004**, which contains multiple operating conditions and two degradation modes. The raw NASA ZIP is intentionally not committed because the repository should remain lightweight and reproducible.

See `data/README.md` for the exact download and placement instructions.

## Method

1. Load the FD004 train/test trajectories and official RUL values.
2. Add engine-cycle and trajectory information.
3. Construct RUL-aware binary labels for four warning horizons.
4. Create sensor-derived features while preserving engine identity.
5. Split validation by engine rather than randomly mixing rows.
6. Compare five classical classifiers.
7. Rank models using ROC-AUC and PR-AUC and inspect operational metrics.
8. Select the decision threshold using an illustrative 5:1 false-negative-to-false-positive cost ratio.
9. Retrain the selected Random Forest on the complete FD004 training set.
10. Evaluate once on the official test set.
11. Use SHAP to interpret the final model when the optional dependency is installed.

## Key Reported Result
The final report selected a **50-cycle warning horizon** and a **0.10 probability threshold**. On the official FD004 test set, the reported Random Forest results were ROC-AUC **0.9738**, PR-AUC **0.9518**, precision **0.6475**, recall **0.9875**, and F1 **0.7822**. The confusion matrix contained 125 true negatives, 43 false positives, 1 false negative, and 79 true positives.

These figures are documented in `results/reported_results.md`; the repository code is provided so the methodology can be reproduced and inspected rather than treating the reported numbers as hard-coded outputs.

## Running the Project

```bash
pip install -r requirements.txt
python src/run_experiment.py --data-dir data/raw --dataset FD004
```

The script expects the standard NASA files (`train_FD004.txt`, `test_FD004.txt`, and `RUL_FD004.txt`) under `data/raw/`.

## Academic Integrity
This repository is organized for the individual DATA110 capstone submission. The code and documentation are intended to make the methodology transparent and reproducible. The raw NASA benchmark is referenced rather than redistributed.
