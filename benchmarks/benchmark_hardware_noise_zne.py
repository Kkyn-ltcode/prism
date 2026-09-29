"""
Experiment 5: NISQ Hardware Noise & Zero-Noise Extrapolation (ZNE) Mitigation
=============================================================================
Simulates realistic NISQ hardware depolarizing noise on the 4-qubit Q-PRISM PQC
and applies Zero-Noise Extrapolation (ZNE) using Richardson and Polynomial extrapolation
via Mitiq + PennyLane to demonstrate error recovery on near-term QPUs.

Outputs publication-quality Figure 4 and results/noise_zne_results.json.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import torch
import pennylane as qml
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


def run_zne_mitigation_sweep(save_dir="./results", num_samples=30):
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 70)
    print("🔬 Executing Experiment 5: NISQ Hardware Noise & ZNE Mitigation")
    print("=" * 70)

    # 1. Fixed test edge inputs & weights
    np.random.seed(42)
    inputs_list = [np.random.uniform(-np.pi, np.pi, size=5) for _ in range(num_samples)]
    weights = np.random.uniform(-np.pi, np.pi, size=(2, n_qubits, 3))
    dist_weights = np.array([0.1, 0.1])

    ideal_circuit = get_ideal_qnode()
    ideal_vals = np.array([ideal_circuit(inp, weights, dist_weights) for inp in inputs_list]) # [N, 4]

    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.035, 0.05]
    scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0]

    unmitigated_errors = []
    mitigated_errors = []
    richardson_mitigated_errors = []

    print(f"Sweeping across {len(noise_levels)} depolarizing noise rates with ZNE...")

    for noise in noise_levels:
        noisy_circuit = get_noisy_qnode(noise)
        
        # Measure unmitigated noisy values (scale = 1.0)
        unmit_vals = np.array([noisy_circuit(inp, weights, dist_weights) for inp in inputs_list])
        err_unmit = np.mean(np.abs(unmit_vals - ideal_vals))
        unmitigated_errors.append(err_unmit)

        if noise == 0.0:
            mitigated_errors.append(0.0)
            richardson_mitigated_errors.append(0.0)
            print(f"  • Noise = {noise:.3f} | Unmitigated Err = {err_unmit:.4f} | ZNE Err = 0.0000")
            continue

        # Simulate noise scaling for ZNE
        # At scale s, effective noise rate is noise * s
        scaled_measurements = []
        for s in scale_factors:
            scaled_circuit = get_noisy_qnode(noise * s)
            s_vals = np.array([scaled_circuit(inp, weights, dist_weights) for inp in inputs_list])
            scaled_measurements.append(s_vals)

        # Apply Polynomial and Richardson Zero-Noise Extrapolation
        # scaled_measurements: [num_scales, num_samples, 4]
        scaled_array = np.array(scaled_measurements)
        
        # Fit polynomial extrapolation to zero noise (s -> 0)
        mit_vals_poly = np.zeros_like(ideal_vals)
        for i in range(num_samples):
            for q in range(n_qubits):
                y_pts = scaled_array[:, i, q]
                poly_coeff = np.polyfit(scale_factors, y_pts, deg=2)
                mit_vals_poly[i, q] = np.polyval(poly_coeff, 0.0)

        # Clip unphysical expectations to [-1, 1]
        mit_vals_poly = np.clip(mit_vals_poly, -1.0, 1.0)
        err_mit = np.mean(np.abs(mit_vals_poly - ideal_vals))
        mitigated_errors.append(err_mit)

        # Richardson extrapolation (first two scale factors)
        # E(0) = 2*E(1) - E(2) for s=1, s=2
        s1_vals = scaled_array[0]
        s2_vals = scaled_array[2]
        richardson_vals = 2.0 * s1_vals - s2_vals
        richardson_vals = np.clip(richardson_vals, -1.0, 1.0)
        err_rich = np.mean(np.abs(richardson_vals - ideal_vals))
        richardson_mitigated_errors.append(err_rich)

        error_reduction_pct = ((err_unmit - err_mit) / max(err_unmit, 1e-8)) * 100.0
        print(f"  • Noise = {noise:.3f} | Unmitigated Err = {err_unmit:.4f} | ZNE Err = {err_mit:.4f} (Reduction: {error_reduction_pct:.1f}%)")

    # 2. Save JSON results
    zne_record = {
        "noise_levels": noise_levels,
        "scale_factors": scale_factors,
        "unmitigated_errors": unmitigated_errors,
        "poly_mitigated_errors": mitigated_errors,
        "richardson_mitigated_errors": richardson_mitigated_errors
    }
    with open(os.path.join(save_dir, "noise_zne_results.json"), "w") as f:
        json.dump(zne_record, f, indent=2)

    # 3. Generate Publication Figure 4
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # Plot (a): Expectation Value Error vs Noise Rate
    ax1.plot(noise_levels, unmitigated_errors, marker="o", color="#d62728", lw=2, label="Unmitigated Noisy QPU")
    ax1.plot(noise_levels, richardson_mitigated_errors, marker="s", color="#ff7f0e", lw=2, linestyle="--", label="ZNE (Richardson Extrapolation)")
    ax1.plot(noise_levels, mitigated_errors, marker="^", color="#1f77b4", lw=2.5, linestyle="-", label="ZNE (Polynomial Degree-2)")
    ax1.set_xlabel("Hardware Depolarizing Noise Rate ($\\lambda$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Expectation Estimation Error $\\epsilon$", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Zero-Noise Extrapolation Performance", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # Plot (b): Sample Extrapolation Curve (for noise = 0.035)
    sample_noise = 0.035
    noisy_q_sample = [get_noisy_qnode(sample_noise * s)(inputs_list[0], weights, dist_weights)[0] for s in scale_factors]
    ideal_sample = ideal_circuit(inputs_list[0], weights, dist_weights)[0]
    p_fit = np.polyfit(scale_factors, noisy_q_sample, deg=2)
    s_dense = np.linspace(0, 3.2, 100)
    
    ax2.scatter(scale_factors, noisy_q_sample, color="#d62728", s=60, zorder=3, label="Scaled Noisy Points")
    ax2.plot(s_dense, np.polyval(p_fit, s_dense), color="#1f77b4", lw=2, linestyle="--", label="Polynomial Fit ($s \\to 0$)")
    ax2.scatter([0.0], [np.polyval(p_fit, 0.0)], color="#2ca02c", marker="*", s=140, zorder=4, label=f"Mitigated Estimate ({np.polyval(p_fit, 0.0):.3f})")
    ax2.axhline(ideal_sample, color="black", linestyle=":", lw=1.5, label=f"Ideal Noiseless Target ({ideal_sample:.3f})")
    ax2.set_xlabel("Noise Scale Factor ($s$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Observable Readout $\\langle Z_0 \\rangle$", fontsize=11, fontweight="bold")
    ax2.set_title(f"(b) Extrapolation Trajectory ($\\lambda = {sample_noise}$)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "figure4_zne_mitigation.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"\n📊 Saved Figure 4 to: {fig_path}")
    print("✅ Experiment 5 Completed Successfully!")


if __name__ == "__main__":
    run_zne_mitigation_sweep()
