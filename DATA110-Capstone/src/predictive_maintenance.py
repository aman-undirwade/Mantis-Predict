"""Reusable DATA110 predictive-maintenance utilities for NASA C-MAPSS FD004.

The module keeps engine identity explicit and performs feature construction without
mixing observations from different engines. It supports the early-warning framing
used in the capstone report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COLUMNS = [
    "unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3",
    *[f"s{i}" for i in range(1, 22)],
]

SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
SETTING_COLUMNS = ["op_setting_1", "op_setting_2", "op_setting_3"]


def load_cmapss_file(path: str | Path) -> pd.DataFrame:
    """Read a whitespace-delimited NASA C-MAPSS trajectory file."""
    path = Path(path)
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Some copies contain trailing blank columns; retain only the standard 26 fields.
    df = df.iloc[:, : len(COLUMNS)].copy()
    df.columns = COLUMNS
    return df


def load_fd004(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load FD004 train trajectories, test trajectories, and official test RUL."""
    data_dir = Path(data_dir)
    train = load_cmapss_file(data_dir / "train_FD004.txt")
    test = load_cmapss_file(data_dir / "test_FD004.txt")
    rul = pd.read_csv(data_dir / "RUL_FD004.txt", header=None, names=["rul"])["rul"]
    return train, test, rul


def add_training_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Attach exact remaining useful life to each training row."""
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = max_cycle - out["cycle"]
    return out


def add_test_rul(df: pd.DataFrame, official_rul: pd.Series) -> pd.DataFrame:
    """Attach RUL to test rows using the official RUL at each engine's last cycle."""
    out = df.copy()
    final_rul = dict(enumerate(official_rul.to_numpy(), start=1))
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = out["unit"].map(final_rul) + (max_cycle - out["cycle"])
    return out


def add_warning_label(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create a binary early-warning target: 1 when RUL <= horizon."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    out = df.copy()
    out["target"] = (out["rul"] <= horizon).astype(int)
    return out


def add_sensor_features(
    df: pd.DataFrame,
    rolling_windows: Iterable[int] = (5, 10),
) -> pd.DataFrame:
    """Create trajectory-aware sensor features.

    For each sensor, rolling mean/std and first difference are calculated within
    each engine. The first row of each trajectory uses a zero difference. Missing
    rolling statistics are back-filled within the same engine and then filled with
    the sensor's global training median by the caller if required.
    """
    out = df.sort_values(["unit", "cycle"]).copy()
    grouped = out.groupby("unit", sort=False)

    for sensor in SENSOR_COLUMNS:
        out[f"{sensor}_diff"] = grouped[sensor].diff().fillna(0.0)
        for window in rolling_windows:
            rolling = grouped[sensor].rolling(window, min_periods=1)
            out[f"{sensor}_mean_{window}"] = (
                rolling.mean().reset_index(level=0, drop=True).to_numpy()
            )
            out[f"{sensor}_std_{window}"] = (
                rolling.std().reset_index(level=0, drop=True).fillna(0.0).to_numpy()
            )

    return out.reset_index(drop=True)


def build_feature_matrix(
    df: pd.DataFrame,
    include_settings: bool = True,
    rolling_windows: Iterable[int] = (5, 10),
) -> tuple[pd.DataFrame, list[str]]:
    """Return numeric features and their names."""
    engineered = add_sensor_features(df, rolling_windows=rolling_windows)
    feature_cols = ["cycle"]
    if include_settings:
        feature_cols.extend(SETTING_COLUMNS)
    feature_cols.extend(SENSOR_COLUMNS)
    feature_cols.extend(
        [
            f"{s}_diff" for s in SENSOR_COLUMNS
        ]
        + [
            f"{s}_mean_{w}" for s in SENSOR_COLUMNS for w in rolling_windows
        ]
        + [
            f"{s}_std_{w}" for s in SENSOR_COLUMNS for w in rolling_windows
        ]
    )
    return engineered[feature_cols].replace([np.inf, -np.inf], np.nan), feature_cols


def last_cycle_per_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Return the final observed row for every engine."""
    idx = df.groupby("unit")["cycle"].idxmax()
    return df.loc[idx].sort_values("unit").reset_index(drop=True)


if __name__ == "__main__":
    print("Import this module from run_experiment.py; it is not a standalone CLI.")
