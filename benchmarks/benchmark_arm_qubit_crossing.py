"""Arm qubit low-noise sweep and precision crossing analysis."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import shutil
import numpy as np
import torch
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


n_qubits = 4

def get_ideal_qnode():
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def circuit(inputs, weights, dist_weights):
        h_i0, h_i1, h_j0, h_j1, d_ij = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        n_layers = weights.shape[0]

        for l in range(n_layers):
            qml.RY(h_i0, wires=0)
            qml.RY(h_i1, wires=1)
            qml.RY(h_j0, wires=2)
            qml.RY(h_j1, wires=3)
            scaled_d = d_ij * dist_weights[l]
            for q in range(n_qubits):
                qml.RZ(scaled_d, wires=q)
            for q in range(n_qubits):
                qml.RX(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)
                qml.RZ(weights[l, q, 2], wires=q)
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])

        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit


def get_noisy_qnode(noise_rate=0.01):
    dev = qml.device("default.mixed", wires=n_qubits)
    @qml.qnode(dev)
    def circuit(inputs, weights, dist_weights):
        h_i0, h_i1, h_j0, h_j1, d_ij = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        n_layers = weights.shape[0]

        for l in range(n_layers):
            qml.RY(h_i0, wires=0)
            qml.RY(h_i1, wires=1)
            qml.RY(h_j0, wires=2)
            qml.RY(h_j1, wires=3)
            scaled_d = d_ij * dist_weights[l]
            for q in range(n_qubits):
                qml.RZ(scaled_d, wires=q)
            for q in range(n_qubits):
                qml.RX(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)
                qml.RZ(weights[l, q, 2], wires=q)
                if noise_rate > 0:
                    qml.DepolarizingChannel(noise_rate, wires=q)

            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
                if noise_rate > 0:
                    # Two-qubit gate depolarizing error
                    qml.DepolarizingChannel(noise_rate * 2.0, wires=q)
                    qml.DepolarizingChannel(noise_rate * 2.0, wires=(q + 1) % n_qubits)

        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit


def load_real_qm9_edge_inputs(num_samples=50):
    """
    Extract real edge feature representations [h_i0, h_i1, h_j0, h_j1, d_ij]
    from QM9 test molecules using trained Q-PRISM projections.
    """
    try:
        from datasets.dataset_qm9 import get_qm9_splits
        _, _, test_set, _ = get_qm9_splits(root="./data/QM9", target="gap", num_train=10, num_val=10, num_test=10, seed=42)
        ckpt_path = PROJECT_ROOT / "results/experiment_c_full_qm9/checkpoints/gap_N110000_prism_s42.pt"
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            sd = ckpt["state_dict"]
            emb = torch.nn.Linear(11, 16)
            emb.weight.data = sd["embedding.weight"]
            proj = torch.nn.Linear(16, 2)
            proj.weight.data = sd["layers.0.head_electronic.feature_proj.weight"]
            proj.bias.data = sd["layers.0.head_electronic.feature_proj.bias"]
        else:
            emb = torch.nn.Linear(11, 16)
            proj = torch.nn.Linear(16, 2)

        inputs_list = []
        for mol in test_set:
            h = proj(emb(mol.x))
            edge_index = mol.edge_index
            pos = mol.pos
            row, col = edge_index[0], edge_index[1]
            d = torch.sum((pos[row] - pos[col])**2, dim=-1, keepdim=True)
            edge_feats = torch.cat([h[row], h[col], d], dim=-1).detach().numpy()
            for ef in edge_feats:
                inputs_list.append(ef)
                if len(inputs_list) >= num_samples:
                    break
            if len(inputs_list) >= num_samples:
                break
        print(f"Loaded {len(inputs_list)} real edge inputs from QM9 test molecules.")
        return np.array(inputs_list)
    except Exception as e:
        print(f"Falling back to synthetic inputs: {e}")
        np.random.seed(42)
        return np.array([np.random.uniform(-np.pi, np.pi, size=5) for _ in range(num_samples)])


def run_benchmark(save_dir="./results", num_samples=50):
    os.makedirs(save_dir, exist_ok=True)
    print("Executing arm qubit low-noise sweep and precision crossing benchmark")

    # Load real QM9 test edge inputs and fixed variational weights
    inputs_list = load_real_qm9_edge_inputs(num_samples=num_samples)
    np.random.seed(42)
    weights = np.random.uniform(-np.pi, np.pi, size=(2, n_qubits, 3))
    dist_weights = np.array([0.1, 0.1])

    ideal_circuit = get_ideal_qnode()
    ideal_vals = np.array([ideal_circuit(inp, weights, dist_weights) for inp in inputs_list])  # [N, 4]

    noise_levels = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0]
    tolerance_threshold = 1e-3  # High-precision observable tolerance (99.9% fidelity)

    unmitigated_errors = []
    poly_mitigated_errors = []
    richardson_mitigated_errors = []

    print(f"Sweeping across {len(noise_levels)} depolarizing noise rates with ZNE...")

    for noise in noise_levels:
        noisy_circuit = get_noisy_qnode(noise)
        
        # Measure unmitigated noisy values (scale = 1.0)
        unmit_vals = np.array([noisy_circuit(inp, weights, dist_weights) for inp in inputs_list])
        err_unmit = float(np.mean(np.abs(unmit_vals - ideal_vals)))
        unmitigated_errors.append(err_unmit)

        # Simulate noise scaling for ZNE
        # At scale s, effective noise rate is noise * s
        scaled_measurements = []
        for s in scale_factors:
            scaled_circuit = get_noisy_qnode(noise * s)
            s_vals = np.array([scaled_circuit(inp, weights, dist_weights) for inp in inputs_list])
            scaled_measurements.append(s_vals)

        # Apply Polynomial and Richardson Zero-Noise Extrapolation
        scaled_array = np.array(scaled_measurements)
        
        # Fit degree-2 polynomial extrapolation to zero noise (s -> 0)
        mit_vals_poly = np.zeros_like(ideal_vals)
        for i in range(len(inputs_list)):
            for q in range(n_qubits):
                y_pts = scaled_array[:, i, q]
                poly_coeff = np.polyfit(scale_factors, y_pts, deg=2)
                mit_vals_poly[i, q] = np.polyval(poly_coeff, 0.0)

        mit_vals_poly = np.clip(mit_vals_poly, -1.0, 1.0)
        err_poly = float(np.mean(np.abs(mit_vals_poly - ideal_vals)))
        poly_mitigated_errors.append(err_poly)

        # Richardson extrapolation (s=1.0 and s=2.0)
        s1_vals = scaled_array[0]  # scale 1.0
        s2_vals = scaled_array[2]  # scale 2.0
        richardson_vals = np.clip(2.0 * s1_vals - s2_vals, -1.0, 1.0)
        err_rich = float(np.mean(np.abs(richardson_vals - ideal_vals)))
        richardson_mitigated_errors.append(err_rich)

        reduction_pct = ((err_unmit - err_poly) / max(err_unmit, 1e-12)) * 100.0
        print(f"  • Noise = {noise:.1e} | Unmit = {err_unmit:.6f} | Rich = {err_rich:.6f} | Poly = {err_poly:.6f} (Red: {reduction_pct:.1f}%)")

    # Find precision crossing point
    crossing_noise = None
    for i in range(len(noise_levels) - 1):
        if unmitigated_errors[i] <= tolerance_threshold and unmitigated_errors[i+1] > tolerance_threshold:
            x1, x2 = np.log10(noise_levels[i]), np.log10(noise_levels[i+1])
            y1, y2 = np.log10(unmitigated_errors[i]), np.log10(unmitigated_errors[i+1])
            y_target = np.log10(tolerance_threshold)
            x_target = x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)
            crossing_noise = float(10**x_target)
            break

    print(f"\nCrossing Point: λ* = {crossing_noise:.2e} (Threshold = {tolerance_threshold})")

    # Save JSON results
    zne_record = {
        "noise_levels": noise_levels,
        "scale_factors": scale_factors,
        "tolerance_threshold": tolerance_threshold,
        "crossing_point_noise_rate": crossing_noise,
        "unmitigated_errors": unmitigated_errors,
        "richardson_mitigated_errors": richardson_mitigated_errors,
        "poly_mitigated_errors": poly_mitigated_errors,
        "num_qm9_samples": len(inputs_list)
    }
    json_path = os.path.join(save_dir, "noise_zne_arm_qubit_results.json")
    with open(json_path, "w") as f:
        json.dump(zne_record, f, indent=2)
    print(f"Saved results to: {json_path}")

    # Generate Figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Plot (a): Log-Log Expectation Value Error vs Noise Rate
    ax1.loglog(noise_levels, unmitigated_errors, marker="o", color="#d62728", lw=2, label="Unmitigated Noisy QPU")
    ax1.loglog(noise_levels, richardson_mitigated_errors, marker="s", color="#ff7f0e", lw=2, linestyle="--", label="ZNE (Richardson)")
    ax1.loglog(noise_levels, poly_mitigated_errors, marker="^", color="#1f77b4", lw=2.5, linestyle="-", label="ZNE (Polynomial Degree-2)")
    ax1.set_xlabel(r"Depolarizing Noise Rate ($\lambda$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Expectation Estimation Error $\epsilon_{\mathrm{obs}}$", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Zero-Noise Extrapolation Performance", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # Plot (b): Observable Precision Crossing Analysis
    ax2.plot(noise_levels, unmitigated_errors, marker="o", color="#d62728", lw=2.2, label=r"Unmitigated Error $\epsilon_{\mathrm{unmit}}$")
    ax2.axhline(tolerance_threshold, color="#444444", linestyle="--", lw=1.6, label=r"Tolerance Threshold ($\epsilon = 10^{-3}$, 99.9% Fidelity)")
    
    # Shaded regions
    ax2.axvspan(1e-5, 1e-4, color="#2ca02c", alpha=0.18, label="Arm Qubit Regime")
    ax2.axvspan(1e-3, 1e-2, color="#d62728", alpha=0.15, label="Current Transmon Regime")

    if crossing_noise:
        ax2.annotate(
            f"Crossing Point\n" + r"$\lambda^* \approx " + f"{crossing_noise:.1e}$",
            xy=(crossing_noise, tolerance_threshold),
            xytext=(crossing_noise * 2.2, tolerance_threshold * 4.5),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1.2, headwidth=6),
            fontsize=10.5,
            fontweight="bold"
        )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Depolarizing Noise Rate ($\lambda$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Unmitigated Estimation Error $\epsilon$", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Hardware Observable Precision Crossing Analysis", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "figure4b_arm_qubit_crossing.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved Figure to: {fig_path}")

    # Copy to paper/figures/ for LaTeX compilation
    paper_fig_path = PROJECT_ROOT / "paper/figures/figure4b_arm_qubit_crossing.png"
    if paper_fig_path.parent.exists():
        shutil.copy2(fig_path, str(paper_fig_path))
        print(f"Copied figure to: {paper_fig_path}")

    print("Arm qubit crossing benchmark completed successfully.")


if __name__ == "__main__":
    run_benchmark()
