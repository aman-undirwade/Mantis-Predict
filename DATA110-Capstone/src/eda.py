"""Generate core exploratory plots for the FD004 capstone."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from predictive_maintenance import SENSOR_COLUMNS, add_training_rul, load_fd004


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="figures/generated")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train, _, _ = load_fd004(args.data_dir)
    train = add_training_rul(train)

    # Class balance at each warning horizon.
    horizons = [10, 20, 30, 50]
    rates = [(train["rul"] <= h).mean() for h in horizons]
    plt.figure(figsize=(7, 4))
    plt.plot(horizons, rates, marker="o")
    plt.xlabel("Warning horizon (cycles)")
    plt.ylabel("Positive-class proportion")
    plt.title("Positive-class rate by warning horizon")
    plt.tight_layout()
    plt.savefig(out / "warning_horizon_balance.png", dpi=180)
    plt.close()

    # Representative sensor trajectories for a small set of engines.
    sample_units = train["unit"].drop_duplicates().head(5)
    plt.figure(figsize=(9, 5))
    for unit in sample_units:
        subset = train[train["unit"] == unit]
        plt.plot(subset["cycle"], subset["s13"], label=f"Engine {unit}")
    plt.xlabel("Cycle")
    plt.ylabel("Sensor s13")
    plt.title("Representative s13 degradation trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "sensor_trajectory_s13.png", dpi=180)
    plt.close()

    # Distribution of the 21 raw sensors.
    train[SENSOR_COLUMNS].describe().T.to_csv(out / "sensor_summary.csv")
    print(f"Wrote EDA outputs to {out}")


if __name__ == "__main__":
    main()
