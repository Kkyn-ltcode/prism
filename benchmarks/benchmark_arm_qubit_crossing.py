"""
Experiment: Arm Qubit Low-Noise Sweep & Crossing-Point Analysis
===============================================================
Simulates realistic NISQ hardware depolarizing noise on the 4-qubit Q-PRISM PQC
and applies Zero-Noise Extrapolation (ZNE). Also maps expectation error
to an estimated molecular MAE to identify the crossing point for chemical accuracy.

Outputs publication-quality Figure and results/noise_zne_arm_qubit_results.json.
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


def run_benchmark(save_dir="./results", num_samples=30):
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 70)
    print("🔬 Executing Arm Qubit Low-Noise Sweep")
    print("=" * 70)

    # 1. Fixed test edge inputs & weights
    np.random.seed(42)
    inputs_list = [np.random.uniform(-np.pi, np.pi, size=5) for _ in range(num_samples)]
    weights = np.random.uniform(-np.pi, np.pi, size=(2, n_qubits, 3))
    dist_weights = np.array([0.1, 0.1])

    ideal_circuit = get_ideal_qnode()
    ideal_vals = np.array([ideal_circuit(inp, weights, dist_weights) for inp in inputs_list]) # [N, 4]

    noise_levels = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    scale_factors = [1.0, 1.5, 2.0, 2.5, 3.0]
    scale_factor_eV = 0.8

    unmitigated_errors = []
    poly_mitigated_errors = []
    richardson_mitigated_errors = []
    molecular_mae_estimates = []

    print(f"Sweeping across {len(noise_levels)} depolarizing noise rates with ZNE...")

    for noise in noise_levels:
        noisy_circuit = get_noisy_qnode(noise)
        
        # Measure unmitigated noisy values (scale = 1.0)
        unmit_vals = np.array([noisy_circuit(inp, weights, dist_weights) for inp in inputs_list])
        err_unmit = np.mean(np.abs(unmit_vals - ideal_vals))
        unmitigated_errors.append(float(err_unmit))
        
        molecular_mae = err_unmit * scale_factor_eV
        molecular_mae_estimates.append(float(molecular_mae))

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
        poly_mitigated_errors.append(float(err_mit))

        # Richardson extrapolation (first two scale factors)
        # E(0) = 2*E(1) - E(2) for s=1, s=2
        s1_vals = scaled_array[0]
        s2_vals = scaled_array[1] # s=1.0 and s=1.5 is standard, but the original logic uses s1_vals and s2_vals. Wait, in original it was: s1_vals = scaled_array[0], s2_vals = scaled_array[2]. scale_factors are 1.0, 1.5, 2.0... so index 2 is s=2.0.
        s2_vals = scaled_array[2] 
        richardson_vals = 2.0 * s1_vals - s2_vals
        richardson_vals = np.clip(richardson_vals, -1.0, 1.0)
        err_rich = np.mean(np.abs(richardson_vals - ideal_vals))
        richardson_mitigated_errors.append(float(err_rich))

        error_reduction_pct = ((err_unmit - err_mit) / max(err_unmit, 1e-8)) * 100.0
        print(f"  • Noise = {noise:.1e} | Unmitigated Err = {err_unmit:.4f} | ZNE Err = {err_mit:.4f} (Reduction: {error_reduction_pct:.1f}%) | MAE = {molecular_mae:.4f} eV")

    # 2. Save JSON results
    zne_record = {
        "noise_levels": noise_levels,
        "scale_factors": scale_factors,
        "unmitigated_errors": unmitigated_errors,
        "poly_mitigated_errors": poly_mitigated_errors,
        "richardson_mitigated_errors": richardson_mitigated_errors,
        "molecular_mae_estimates": molecular_mae_estimates
    }
    with open(os.path.join(save_dir, "noise_zne_arm_qubit_results.json"), "w") as f:
        json.dump(zne_record, f, indent=2)

    # 3. Generate Publication Figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Plot (a): Log-Log Expectation Value Error vs Noise Rate
    ax1.loglog(noise_levels, unmitigated_errors, marker="o", color="#d62728", lw=2, label="Unmitigated")
    ax1.loglog(noise_levels, richardson_mitigated_errors, marker="s", color="#ff7f0e", lw=2, linestyle="--", label="ZNE (Richardson)")
    ax1.loglog(noise_levels, poly_mitigated_errors, marker="^", color="#1f77b4", lw=2.5, linestyle="-", label="ZNE (Polynomial)")
    ax1.set_xlabel("Depolarizing Noise Rate ($\\lambda$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Expectation Estimation Error $\\epsilon$", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Zero-Noise Extrapolation Performance", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # Plot (b): Estimated Molecular MAE (eV) vs Noise Rate
    ax2.plot(noise_levels, molecular_mae_estimates, marker="o", color="#d62728", lw=2, label="Estimated MAE")
    
    # Horizontal line at 0.043 eV
    ax2.axhline(0.043, color="gray", linestyle="--", lw=1.5, label="1 kcal/mol Chemical Accuracy")
    
    # Shaded regions
    ax2.axvspan(1e-5, 1e-4, color="green", alpha=0.2, label="Arm Qubit Regime")
    ax2.axvspan(1e-3, 1e-2, color="pink", alpha=0.3, label="Current Transmon Regime")

    # Find crossing point
    crossing_noise = None
    for i in range(len(noise_levels) - 1):
        if molecular_mae_estimates[i] <= 0.043 and molecular_mae_estimates[i+1] > 0.043:
            # Linear interpolation in log-log space (approx)
            x1, x2 = np.log10(noise_levels[i]), np.log10(noise_levels[i+1])
            y1, y2 = np.log10(molecular_mae_estimates[i]), np.log10(molecular_mae_estimates[i+1])
            y_target = np.log10(0.043)
            x_target = x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)
            crossing_noise = 10**x_target
            break

    if crossing_noise:
        ax2.annotate(f"Crossing Point\\n~ {crossing_noise:.1e}", 
                     xy=(crossing_noise, 0.043), 
                     xytext=(crossing_noise*2, 0.08),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                     fontsize=10, fontweight="bold")

    ax2.set_xscale("log")
    ax2.set_xlabel("Depolarizing Noise Rate ($\\lambda$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Estimated Molecular MAE (eV)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Chemical Accuracy Crossing Analysis", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "figure4b_arm_qubit_crossing.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"\\n📊 Saved Figure to: {fig_path}")
    print("✅ Experiment Completed Successfully!")


if __name__ == "__main__":
    run_benchmark()
