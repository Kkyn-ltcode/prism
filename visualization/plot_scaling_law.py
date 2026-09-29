"""
Publication-Quality Plotting for Scaling Laws & Benchmark Figures
=================================================================
Plots Figure 1: Test MAE vs. Training Size N in log-log scale with power-law fit.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def power_law(N, beta, gamma):
    return beta * (N ** (-gamma))

def generate_scaling_plot(results_dir=None, target="gap", output_path=None):
    if results_dir is None:
        results_dir = str(PROJECT_ROOT / "results")
    if output_path is None:
        output_path = str(PROJECT_ROOT / "results" / "figure1_scaling_law.png")

    files = glob.glob(os.path.join(results_dir, f"*_{target}_N*.json"))
    if not files:
        print(f"No result JSON files found for target '{target}' in '{results_dir}'.")
        return

    data = {"eqgnn": {"N": [], "mae": []}, "egnn": {"N": [], "mae": []}}
    unit = "eV"

    for fpath in files:
        with open(fpath, "r") as f:
            rec = json.load(f)
            model = rec["model"]
            n = rec["num_train"]
            mae = rec["test_mae"]
            unit = rec.get("unit", unit)
            if model in data:
                data[model]["N"].append(n)
                data[model]["mae"].append(mae)

    # Sort by N
    for m in data:
        sorted_pairs = sorted(zip(data[m]["N"], data[m]["mae"]))
        if sorted_pairs:
            data[m]["N"], data[m]["mae"] = zip(*sorted_pairs)
            data[m]["N"] = np.array(data[m]["N"], dtype=float)
            data[m]["mae"] = np.array(data[m]["mae"], dtype=float)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    colors = {"eqgnn": "#1f77b4", "egnn": "#ff7f0e"}
    labels = {"eqgnn": "Q-PRISM (Quantum, 4-Qubit PQC)", "egnn": "EGNN (Classical SOTA Baseline)"}
    markers = {"eqgnn": "o", "egnn": "s"}

    for m in ["egnn", "eqgnn"]:
        n_vals = data[m]["N"]
        mae_vals = data[m]["mae"]
        if len(n_vals) > 0:
            ax.scatter(n_vals, mae_vals, color=colors[m], marker=markers[m], s=60, label=f"{labels[m]} Data", zorder=3)
            
            # Fit power law MAE(N) = beta * N^(-gamma)
            if len(n_vals) >= 2:
                try:
                    popt, _ = curve_fit(power_law, n_vals, mae_vals, p0=[1.0, 0.2])
                    beta, gamma = popt
                    n_dense = np.linspace(n_vals.min() * 0.9, n_vals.max() * 1.1, 100)
                    ax.plot(n_dense, power_law(n_dense, beta, gamma), color=colors[m], linestyle="--", alpha=0.8,
                            label=f"{m.upper()} Fit (γ = {gamma:.3f})")
                except Exception as e:
                    ax.plot(n_vals, mae_vals, color=colors[m], linestyle="--", alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Training Molecules ($N$)", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Test MAE [{unit}] (Log Scale)", fontsize=12, fontweight="bold")
    ax.set_title(f"Sample Efficiency Scaling Law: QM9 HOMO-LUMO Gap ($\Delta\epsilon$)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Saved scaling plot to: {output_path}")

if __name__ == "__main__":
    generate_scaling_plot()
