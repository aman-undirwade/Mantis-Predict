"""SHAP explainability for the selected FD004 Random Forest.

Run after the data is installed. The script writes a global feature-importance
CSV and a bar chart to the results directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from predictive_maintenance import add_training_rul, add_warning_label, build_feature_matrix, load_fd004


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="results/generated")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train, _, _ = load_fd004(args.data_dir)
    train = add_training_rul(train)
    train = add_warning_label(train, 50)
    X, feature_names = build_feature_matrix(train)

    imputer = SimpleImputer(strategy="median")
    X_clean = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)
    y = train["target"].to_numpy()

    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_clean, y)

    # A bounded sample keeps explainability practical on the full trajectory table.
    sample = X_clean.sample(min(5000, len(X_clean)), random_state=42)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[1]
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(output_dir / "shap_feature_importance.csv", index=False)

    top = importance.head(15).sort_values("mean_abs_shap")
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Global SHAP feature importance — FD004")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_feature_importance.png", dpi=180)
    plt.close()

    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
