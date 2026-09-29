"""
Publication-Quality Scaling Curves & Crossover Analysis
========================================================
Generates figures from the full-dataset scaling sweep results.

Usage:
    pixi run python visualization/plot_scaling_crossover.py
    pixi run python visualization/plot_scaling_crossover.py --benchmark qm9
    pixi run python visualization/plot_scaling_crossover.py --benchmark md17
    pixi run python visualization/plot_scaling_crossover.py --benchmark nbody
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# =============================================================================
# Styling
# =============================================================================
MODEL_COLORS = {
    "eqgnn_v2": "#1f77b4", "eqgnn": "#1f77b4",
    "egnn_pm": "#9467bd",
    "egnn_std": "#ff7f0e", "egnn": "#ff7f0e",
    "noneq": "#d62728",
}
MODEL_LABELS = {
    "eqgnn_v2": "Q-PRISM (Ours, 2,527 params)",
    "eqgnn": "Q-PRISM (Ours)",
    "egnn_pm": "EGNN Param-Matched",
    "egnn_std": "EGNN Standard (4,289 params)",
    "egnn": "EGNN Standard",
    "noneq": "Non-Equivariant GNN",
}
MODEL_MARKERS = {
    "eqgnn_v2": "o", "eqgnn": "o",
    "egnn_pm": "D",
    "egnn_std": "s", "egnn": "s",
    "noneq": "^",
}

TARGET_DISPLAY = {
    "homo": r"HOMO Energy ($\epsilon_{\mathrm{HOMO}}$)",
    "lumo": r"LUMO Energy ($\epsilon_{\mathrm{LUMO}}$)",
    "gap": r"HOMO-LUMO Gap ($\Delta\epsilon$)",
    "u0": r"Internal Energy ($U_0$)",
}
TARGET_UNITS = {"homo": "eV", "lumo": "eV", "gap": "eV", "u0": "eV"}


def power_law(N, beta, gamma):
    """Power law: MAE(N) = beta * N^(-gamma)"""
    return beta * (N ** (-gamma))


def find_crossover(n_range, params_a, params_b):
    """
    Find crossover point N* where model A and model B curves intersect.
    Returns N* or None if no crossing in range.
    """
    n_dense = np.linspace(n_range[0], n_range[-1], 10000)
    try:
        curve_a = power_law(n_dense, *params_a)
        curve_b = power_law(n_dense, *params_b)
        diff = curve_a - curve_b
        # Find sign change
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            # Linear interpolation for better precision
            x1, x2 = n_dense[idx], n_dense[idx + 1]
            y1, y2 = diff[idx], diff[idx + 1]
            n_star = x1 - y1 * (x2 - x1) / (y2 - y1)
            return float(n_star)
    except Exception:
        pass
    return None


# =============================================================================
# QM9 Plotting
# =============================================================================
def plot_qm9_scaling(results_path, output_dir):
    """Generate multi-panel QM9 scaling figure."""
    with open(results_path, "r") as f:
        results = json.load(f)

    # Detect which targets have data
    targets = [t for t in ["homo", "lumo", "gap", "u0"] if t in results and isinstance(results[t], dict)]
    if not targets:
        print("No QM9 target data found.")
        return

    models = ["eqgnn_v2", "egnn_pm", "egnn_std"]
    n_targets = len(targets)
    ncols = min(n_targets, 2)
    nrows = (n_targets + ncols - 1) // ncols

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), dpi=300)
    if n_targets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    crossover_results = {}

    for ax_idx, target in enumerate(targets):
        ax = axes[ax_idx]
        target_data = results[target]
        unit = target_data.get("unit", "eV")
        fits = {}

        for model_key in models:
            n_vals = []
            mean_maes = []
            std_maes = []

            # Collect data across all N values
            for key, val in target_data.items():
                if key.startswith("N") and key[1:].isdigit():
                    n = int(key[1:])
                    model_data = val.get(model_key, {})
                    if "mean_mae" in model_data:
                        n_vals.append(n)
                        mean_maes.append(model_data["mean_mae"])
                        std_maes.append(model_data["std_mae"])

            if not n_vals:
                continue

            # Sort by N
            sort_idx = np.argsort(n_vals)
            n_vals = np.array(n_vals)[sort_idx]
            mean_maes = np.array(mean_maes)[sort_idx]
            std_maes = np.array(std_maes)[sort_idx]

            color = MODEL_COLORS.get(model_key, "#333333")
            marker = MODEL_MARKERS.get(model_key, "o")
            label = MODEL_LABELS.get(model_key, model_key)

            # Plot data points with error bars
            ax.errorbar(n_vals, mean_maes, yerr=std_maes,
                        fmt=marker, color=color, markersize=7, capsize=4,
                        label=label, zorder=3, linewidth=1.5)

            # Fit power law
            if len(n_vals) >= 3:
                try:
                    popt, _ = curve_fit(power_law, n_vals, mean_maes,
                                        p0=[1.0, 0.2], maxfev=5000)
                    fits[model_key] = popt
                    n_dense = np.logspace(np.log10(n_vals.min() * 0.8),
                                          np.log10(n_vals.max() * 1.2), 200)
                    ax.plot(n_dense, power_law(n_dense, *popt),
                            color=color, linestyle="--", alpha=0.6, linewidth=1.2,
                            label=f"  fit: $\\gamma={popt[1]:.3f}$")
                except Exception:
                    pass

        # Find and annotate crossover points
        if "eqgnn_v2" in fits:
            for baseline in ["egnn_pm", "egnn_std"]:
                if baseline in fits:
                    n_star = find_crossover(
                        [min(n_vals), max(n_vals) * 2],
                        fits["eqgnn_v2"], fits[baseline]
                    )
                    if n_star and n_star > 0:
                        crossover_results[f"{target}_{baseline}"] = n_star
                        mae_star = power_law(n_star, *fits["eqgnn_v2"])
                        ax.axvline(x=n_star, color="gray", linestyle=":", alpha=0.7, linewidth=1)
                        ax.annotate(f"$N^*\\approx{n_star:.0f}$",
                                    xy=(n_star, mae_star), fontsize=9,
                                    xytext=(10, 15), textcoords="offset points",
                                    arrowprops=dict(arrowstyle="->", color="gray"),
                                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of Training Molecules ($N$)", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"Test MAE [{unit}]", fontsize=11, fontweight="bold")
        ax.set_title(TARGET_DISPLAY.get(target, target), fontsize=12, fontweight="bold", pad=10)
        ax.legend(fontsize=8, frameon=True, loc="upper right")
        ax.grid(True, which="both", ls="--", alpha=0.4)

    # Hide unused axes
    for i in range(len(targets), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Sample Efficiency Scaling Law: QM9 Molecular Properties\n"
                 "$N \\in \\{100, 250, 500, 1000, 2500, 5000, 10000\\}$ — 5 Seeds",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "figure1_scaling_law.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved QM9 scaling figure to: {out_path}")

    # Copy to paper/figures
    paper_fig_dir = os.path.join(str(PROJECT_ROOT), "paper", "figures")
    os.makedirs(paper_fig_dir, exist_ok=True)
    os.system(f"cp '{out_path}' '{paper_fig_dir}/'")

    # Print crossover summary
    if crossover_results:
        print("\n📐 Crossover Points (N*):")
        for key, n_star in crossover_results.items():
            print(f"   {key}: N* ≈ {n_star:.0f}")

    return crossover_results


# =============================================================================
# MD17 Plotting
# =============================================================================
def plot_md17_scaling(results_path, output_dir):
    """Generate MD17 scaling figure."""
    with open(results_path, "r") as f:
        results = json.load(f)

    molecules = [m for m in ["ethanol", "aspirin"] if m in results and isinstance(results[m], dict)]
    if not molecules:
        print("No MD17 data found.")
        return

    models = ["eqgnn", "egnn_pm", "egnn", "noneq"]
    n_mols = len(molecules)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, n_mols, figsize=(7 * n_mols, 5.5), dpi=300)
    if n_mols == 1:
        axes = [axes]

    for ax, mol in zip(axes, molecules):
        mol_data = results[mol]

        for model_key in models:
            n_vals = []
            mean_maes = []
            std_maes = []

            for key, val in mol_data.items():
                if key.startswith("N") and key[1:].isdigit():
                    n = int(key[1:])
                    model_data = val.get(model_key, {})
                    if "mean_mae" in model_data:
                        n_vals.append(n)
                        mean_maes.append(model_data["mean_mae"])
                        std_maes.append(model_data["std_mae"])

            if not n_vals:
                continue

            sort_idx = np.argsort(n_vals)
            n_vals = np.array(n_vals)[sort_idx]
            mean_maes = np.array(mean_maes)[sort_idx]
            std_maes = np.array(std_maes)[sort_idx]

            color = MODEL_COLORS.get(model_key, "#333333")
            marker = MODEL_MARKERS.get(model_key, "o")
            label = MODEL_LABELS.get(model_key, model_key)

            ax.errorbar(n_vals, mean_maes, yerr=std_maes,
                        fmt=f"{marker}-", color=color, markersize=7, capsize=4,
                        label=label, zorder=3, linewidth=1.5)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of Training Conformations ($N$)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Energy MAE [kcal/mol]", fontsize=11, fontweight="bold")
        ax.set_title(f"{mol.capitalize()}", fontsize=13, fontweight="bold", pad=10)
        ax.legend(fontsize=9, frameon=True, loc="upper right")
        ax.grid(True, which="both", ls="--", alpha=0.4)

    fig.suptitle("MD17 Conformational Energy Scaling",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "figure_md17_scaling.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved MD17 scaling figure to: {out_path}")

    paper_fig_dir = os.path.join(str(PROJECT_ROOT), "paper", "figures")
    os.makedirs(paper_fig_dir, exist_ok=True)
    os.system(f"cp '{out_path}' '{paper_fig_dir}/'")


# =============================================================================
# N-Body Plotting
# =============================================================================
def plot_nbody_scaling(results_path, output_dir):
    """Generate N-body scaling figure."""
    with open(results_path, "r") as f:
        results = json.load(f)

    models = ["eqgnn", "egnn_pm", "egnn", "noneq"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    for model_key in models:
        n_vals = []
        mean_mses = []
        std_mses = []

        for key, val in results.items():
            if key.startswith("N") and key[1:].isdigit():
                n = int(key[1:])
                model_data = val.get(model_key, {})
                if "mean_mse" in model_data:
                    n_vals.append(n)
                    mean_mses.append(model_data["mean_mse"])
                    std_mses.append(model_data["std_mse"])

        if not n_vals:
            continue

        sort_idx = np.argsort(n_vals)
        n_vals = np.array(n_vals)[sort_idx]
        mean_mses = np.array(mean_mses)[sort_idx]
        std_mses = np.array(std_mses)[sort_idx]

        color = MODEL_COLORS.get(model_key, "#333333")
        marker = MODEL_MARKERS.get(model_key, "o")
        label = MODEL_LABELS.get(model_key, model_key)

        ax.errorbar(n_vals, mean_mses, yerr=std_mses,
                    fmt=f"{marker}-", color=color, markersize=7, capsize=4,
                    label=label, zorder=3, linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlabel("Number of Training Trajectories ($N$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Coordinate Prediction MSE", fontsize=11, fontweight="bold")
    ax.set_title("N-Body Trajectory Prediction Scaling", fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9, frameon=True, loc="upper right")
    ax.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "figure_nbody_scaling.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved N-body scaling figure to: {out_path}")

    paper_fig_dir = os.path.join(str(PROJECT_ROOT), "paper", "figures")
    os.makedirs(paper_fig_dir, exist_ok=True)
    os.system(f"cp '{out_path}' '{paper_fig_dir}/'")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Plot scaling sweep results")
    parser.add_argument("--benchmark", type=str, default="all",
                        choices=["all", "qm9", "md17", "nbody"])
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()

    benchmarks = ["qm9", "md17", "nbody"] if args.benchmark == "all" else [args.benchmark]

    for bench in benchmarks:
        if bench == "qm9":
            path = os.path.join(args.results_dir, "qm9_scaling_sweep.json")
            if os.path.exists(path):
                plot_qm9_scaling(path, args.output_dir)
            else:
                print(f"⚠️  QM9 results not found at {path}")

        elif bench == "md17":
            path = os.path.join(args.results_dir, "md17_scaling_sweep.json")
            if os.path.exists(path):
                plot_md17_scaling(path, args.output_dir)
            else:
                print(f"⚠️  MD17 results not found at {path}")

        elif bench == "nbody":
            path = os.path.join(args.results_dir, "nbody_scaling_sweep.json")
            if os.path.exists(path):
                plot_nbody_scaling(path, args.output_dir)
            else:
                print(f"⚠️  N-body results not found at {path}")


if __name__ == "__main__":
    main()
