# Notebooks

The capstone's reproducible implementation is kept in `src/` so that the core logic is version-controlled as normal Python modules. If a Jupyter notebook is used for the final demonstration, it should import the same functions from `src/` rather than maintaining a second copy of the modelling logic.

Recommended notebook order:

1. Data loading and schema check
2. Exploratory analysis
3. RUL and warning-label construction
4. Feature engineering
5. Model comparison across warning horizons
6. Threshold selection and error analysis
7. Official FD004 test evaluation
8. SHAP explainability

This keeps the notebook aligned with the written report and avoids duplicated, potentially inconsistent code.
