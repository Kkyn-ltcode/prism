"""
Publication-Quality Figure: P2.1 Large-Scale Evaluation
========================================================
Addresses Reviewer Concern P2.1 (R2, R3, R5):
Demonstrates that PRISM (SIREN) maintains its generalization
advantage over Classical Twin (tanh) and Q-PRISM (PQC) at large training
set sizes (up to N=10,000).

Outputs:
    paper/figures/figure_p21_scaling.png
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_p21_scaling(json_path=None, output_path=None):
    if json_path is None:
        json_path = PROJECT_ROOT / "results" / "p21_large_scale_eval.json"
    else:
        json_path = Path(json_path)

    if output_path is None:
        output_path = PROJECT_ROOT / "paper" / "figures" / "figure_p21_scaling.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    # Model configuration according to specifications
    models_cfg = [
        {
            "key": "trig_twin_siren",
            "label": "PRISM (SIREN)",
            "color": "#4CAF50",
            "marker": "o",
            "linestyle": "-",
        },
        {
            "key": "classical_twin",
            "label": "Classical Twin (tanh)",
            "color": "#F44336",
            "marker": "s",
            "linestyle": "--",
        },
        {
            "key": "eqgnn_v2",
            "label": "Q-PRISM (PQC)",
            "color": "#2196F3",
            "marker": "^",
            "linestyle": ":",
        },
    ]

    targets = [
        ("homo", "HOMO"),
        ("lumo", "LUMO"),
        ("gap", "Gap"),
    ]

    # Clean publication styling
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#2b2b2b"
    plt.rcParams["axes.linewidth"] = 1.1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    x_ticks = [1000, 5000, 10000]
    x_tick_labels = ["1,000", "5,000", "10,000"]

    for idx, (target_key, target_title) in enumerate(targets):
        ax = axes[idx]
        ax.set_axisbelow(True)
        target_data = data.get(target_key, {})

        for mcfg in models_cfg:
            model_key = mcfg["key"]
            n_vals = []
            mae_vals = []

            for n_str, runs in target_data.items():
                try:
                    n_val = int(n_str.replace("N=", ""))
                except ValueError:
                    continue

                for run_key, run_info in runs.items():
                    if run_key.startswith(model_key):
                        # Filter: only use data points with epochs == 20
                        # Ignore N=1000 dry-run artifacts with epochs == 2
                        if run_info.get("epochs") == 20:
                            n_vals.append(n_val)
                            mae_vals.append(run_info["test_mae"])

            if n_vals:
                sorted_idx = np.argsort(n_vals)
                n_arr = np.array(n_vals)[sorted_idx]
                mae_arr = np.array(mae_vals)[sorted_idx]

                ax.plot(
                    n_arr,
                    mae_arr,
                    color=mcfg["color"],
                    marker=mcfg["marker"],
                    linestyle=mcfg["linestyle"],
                    linewidth=2.5,
                    markersize=9,
                    markeredgewidth=1.5,
                    markeredgecolor="white",
                    label=mcfg["label"],
                    zorder=4,
                )

        # X-axis configuration
        ax.set_xscale("log")
        ax.set_xlim(800, 12500)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels, fontsize=12)
        ax.minorticks_off()

        # Labels and title
        ax.set_xlabel(r"$N$ (training molecules)", fontsize=14, labelpad=8)
        ax.set_ylabel("Test MAE (eV)", fontsize=14, labelpad=8)
        ax.set_title(target_title, fontsize=15, fontweight="bold", pad=10)

        # Tick styling
        ax.tick_params(axis="both", which="major", labelsize=12)

        # Grid
        ax.grid(True, which="major", linestyle="--", alpha=0.3, color="#b0b0b0", zorder=1)

        # Legend in first subplot only
        if idx == 0:
            ax.legend(
                fontsize=11,
                loc="upper left",
                frameon=True,
                framealpha=0.95,
                edgecolor="#cccccc",
                facecolor="#ffffff",
            )

    fig.suptitle(
        "Large-Scale Evaluation: Periodic Activation Advantage Persists at N=10,000",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Publication figure saved to: {output_path.resolve()}")


if __name__ == "__main__":
    plot_p21_scaling()
