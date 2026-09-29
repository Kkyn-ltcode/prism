"""
Experiment: Parasitic ZZ Crosstalk Noise Channel
================================================
Simulates realistic NISQ hardware depolarizing noise alongside parasitic
ZZ crosstalk on the 4-qubit Q-PRISM PQC.

Outputs publication-quality Figure 4c and results/noise_zz_crosstalk_results.json.
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


def get_noisy_qnode(noise_rate=0.01, zeta_hz=0.0, dt=200e-9):
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
                    qml.DepolarizingChannel(noise_rate * 2.0, wires=q)
                    qml.DepolarizingChannel(noise_rate * 2.0, wires=(q + 1) % n_qubits)
            
            # Idle period: Apply ZZ crosstalk between adjacent pairs
            if zeta_hz > 0:
                phi = zeta_hz * dt
                for q in range(n_qubits):
                    qml.IsingZZ(phi, wires=[q, (q + 1) % n_qubits])

        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return circuit


def run_crosstalk_benchmark(save_dir="./results", num_samples=30):
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 70)
    print("🔬 Executing Parasitic ZZ Crosstalk Benchmark")
    print("=" * 70)

    # Setup inputs
    np.random.seed(42)
    inputs_list = [np.random.uniform(-np.pi, np.pi, size=5) for _ in range(num_samples)]
    weights = np.random.uniform(-np.pi, np.pi, size=(2, n_qubits, 3))
    dist_weights = np.array([0.1, 0.1])

    ideal_circuit = get_ideal_qnode()
    ideal_vals = np.array([ideal_circuit(inp, weights, dist_weights) for inp in inputs_list])

    zeta_kHz_values = [0.4, 5, 10, 50, 100]
    noise_levels = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2]
    
    error_grid = np.zeros((len(zeta_kHz_values), len(noise_levels)))
    dt = 200e-9

    arm_qubit_error = 0.0
    transmon_error = 0.0
    
    print(f"Sweeping 2D grid: {len(zeta_kHz_values)} ZZ rates × {len(noise_levels)} Depol rates...")

    for i, zeta_kHz in enumerate(zeta_kHz_values):
        zeta_hz = 2 * np.pi * zeta_kHz * 1e3
        for j, noise in enumerate(noise_levels):
            noisy_circuit = get_noisy_qnode(noise, zeta_hz, dt)
            
            noisy_vals = np.array([noisy_circuit(inp, weights, dist_weights) for inp in inputs_list])
            err = np.mean(np.abs(noisy_vals - ideal_vals))
            error_grid[i, j] = err
            
            # Capture specific points
            if np.isclose(zeta_kHz, 0.4) and np.isclose(noise, 8.6e-5, atol=1e-6):
                arm_qubit_error = err
            elif np.isclose(zeta_kHz, 50) and np.isclose(noise, 1e-3, atol=1e-6):
                transmon_error = err
                
            print(f"  • ζ/(2π)={zeta_kHz} kHz, λ={noise:.1e} | Err = {err:.4f}")

    # Save JSON results
    results_record = {
        "zeta_kHz_values": zeta_kHz_values,
        "noise_levels": noise_levels,
        "error_grid": error_grid.tolist(),
        "arm_qubit_error": float(arm_qubit_error),
        "transmon_error": float(transmon_error)
    }
    with open(os.path.join(save_dir, "noise_zz_crosstalk_results.json"), "w") as f:
        json.dump(results_record, f, indent=2)

    # Generate Figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    X, Y = np.meshgrid(noise_levels, zeta_kHz_values)
    c = ax.pcolormesh(X, Y, error_grid, shading='nearest', cmap='viridis')
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Expectation Estimation Error $\\epsilon$", fontsize=11, fontweight="bold")

    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Mark points
    ax.scatter([8.6e-5], [0.4], color='lime', marker='*', s=200, edgecolor='black', zorder=5)
    ax.annotate("Arm Qubit", (8.6e-5, 0.4), xytext=(10, -10), textcoords="offset points", 
                color='white', fontweight='bold', fontsize=10, 
                path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')])
                
    ax.scatter([1e-3], [50], color='red', marker='*', s=200, edgecolor='black', zorder=5)
    ax.annotate("IBM Transmon", (1e-3, 50), xytext=(10, -10), textcoords="offset points", 
                color='white', fontweight='bold', fontsize=10,
                path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')])

    ax.set_xlabel("Hardware Depolarizing Noise Rate ($\\lambda$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Parasitic ZZ Coupling $\\zeta / (2\\pi)$ [kHz]", fontsize=11, fontweight="bold")
    ax.set_title("(c) Parasitic ZZ Crosstalk × Depolarizing Noise Impact", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "figure4c_zz_crosstalk_heatmap.png")
    plt.savefig(fig_path)
    plt.close()
    
    print(f"\n📊 Saved Figure 4c to: {fig_path}")
    print("✅ Benchmark Completed Successfully!")


if __name__ == "__main__":
    run_crosstalk_benchmark()
