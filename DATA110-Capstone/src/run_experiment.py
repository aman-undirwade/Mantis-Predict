"""Run the DATA110 FD004 early-warning classification experiment.

Example:
    python src/run_experiment.py --data-dir data/raw --dataset FD004

The script compares five classical classifiers across warning horizons and then
performs an official-test evaluation for the selected Random Forest operating point.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from predictive_maintenance import add_test_rul, add_training_rul, add_warning_label, build_feature_matrix, load_fd004

warnings.filterwarnings("ignore")

HORIZONS = (10, 20, 30, 50)
RANDOM_STATE = 42


def make_models() -> dict[str, Pipeline]:
    """Return the DATA110 model set with sensible preprocessing."""
    scaled = lambda estimator: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])
    tree_based = lambda estimator: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])
    return {
        "Logistic Regression": scaled(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        "Gaussian Naive Bayes": scaled(GaussianNB()),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=11, weights="distance")),
        "Decision Tree": tree_based(DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE)),
        "Random Forest": tree_based(RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    }


def engine_holdout(df: pd.DataFrame, test_size: float = 0.2):
    """Split complete engines, preventing row-level leakage across train/validation."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, val_idx = next(splitter.split(df, groups=df["unit"]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def metrics(y_true, probability, threshold: float = 0.5) -> dict[str, float]:
    pred = (probability >= threshold).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y_true, probability),
        "PR_AUC": average_precision_score(y_true, probability),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
    }


def select_cost_threshold(y_true, probability, fn_cost: float = 5.0, fp_cost: float = 1.0):
    """Select a probability threshold using an illustrative asymmetric cost."""
    best = None
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (probability >= threshold).astype(int)
        fn = int(((y_true == 1) & (pred == 0)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        cost = fn_cost * fn + fp_cost * fp
        candidate = (cost, threshold, fn, fp)
        if best is None or candidate < best:
            best = candidate
    return {"cost": best[0], "threshold": best[1], "false_negatives": best[2], "false_positives": best[3]}


def prepare(df: pd.DataFrame, horizon: int):
    labelled = add_warning_label(df, horizon)
    X, feature_cols = build_feature_matrix(labelled)
    y = labelled["target"].to_numpy()
    return X, y, feature_cols


def compare_models(train_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Compare models on an engine-held-out validation set."""
    fit_df, val_df = engine_holdout(train_df)
    X_train, y_train, _ = prepare(fit_df, horizon)
    X_val, y_val, _ = prepare(val_df, horizon)
    rows = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_val)[:, 1]
        m = metrics(y_val, probability)
        m.update({"Model": name, "Horizon": horizon})
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--dataset", default="FD004")
    parser.add_argument("--output-dir", default="results/generated")
    args = parser.parse_args()

    if args.dataset.upper() != "FD004":
        raise ValueError("This submission pipeline is configured for FD004.")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train, test, official_rul = load_fd004(data_dir)
    train = add_training_rul(train)
    test = add_test_rul(test, official_rul)

    all_results = []
    for horizon in HORIZONS:
        print(f"Evaluating warning horizon: {horizon} cycles")
        all_results.append(compare_models(train, horizon))

    comparison = pd.concat(all_results, ignore_index=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    # The report identifies RF at a 50-cycle horizon as the selected operating point.
    selected_horizon = 50
    X_train, y_train, _ = prepare(train, selected_horizon)

    # Build test features from complete trajectories first so rolling statistics retain history.
    test_labelled = add_warning_label(test, selected_horizon)
    X_test_all, _, _ = prepare(test_labelled, selected_horizon)
    last_mask = test_labelled["cycle"].eq(test_labelled.groupby("unit")["cycle"].transform("max"))
    X_test_last = X_test_all.loc[last_mask].reset_index(drop=True)
    test_last = test_labelled.loc[last_mask].sort_values("unit").reset_index(drop=True)

    rf = make_models()["Random Forest"]
    rf.fit(X_train, y_train)

    # Threshold selection is performed on a separate engine-held-out validation set.
    fit_df, val_df = engine_holdout(train)
    X_fit, y_fit, _ = prepare(fit_df, selected_horizon)
    X_val, y_val, _ = prepare(val_df, selected_horizon)
    rf_for_threshold = clone(rf)
    rf_for_threshold.fit(X_fit, y_fit)
    val_probability = rf_for_threshold.predict_proba(X_val)[:, 1]
    chosen = select_cost_threshold(y_val, val_probability)

    test_probability = rf.predict_proba(X_test_last)[:, 1]
    test_y = (test_last["rul"].to_numpy() <= selected_horizon).astype(int)
    test_metrics = metrics(test_y, test_probability, threshold=chosen["threshold"])

    final = pd.DataFrame({
        "unit": test_last["unit"].to_numpy(),
        "cycle": test_last["cycle"].to_numpy(),
        "rul": test_last["rul"].to_numpy(),
        "probability": test_probability,
        "prediction": (test_probability >= chosen["threshold"]).astype(int),
    })
    final.to_csv(output_dir / "official_test_predictions.csv", index=False)
    joblib.dump(rf, output_dir / "random_forest_fd004.joblib")

    summary = pd.DataFrame([{
        "selected_horizon": selected_horizon,
        "selected_threshold_from_validation": chosen["threshold"],
        **test_metrics,
    }])
    summary.to_csv(output_dir / "official_test_summary.csv", index=False)

    print("\nModel comparison written to", output_dir / "model_comparison.csv")
    print("Official test summary written to", output_dir / "official_test_summary.csv")
    print("Selected threshold:", chosen["threshold"])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
