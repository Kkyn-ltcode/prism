"""Device-realistic noise validation against IBM Brisbane calibrated noise model."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import argparse
import numpy as np
import torch
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

try:
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane
except ImportError:
    try:
        from qiskit_ibm_runtime.fake_provider import FakeBrisbaneV2 as FakeBrisbane
    except ImportError:
        from qiskit.providers.fake_provider import FakeBrisbane

# Q-PRISM 4-qubit PQC
N_QUBITS = 4
N_LAYERS = 2


def build_qiskit_pqc(inputs, weights, dist_weights):
    """Build Q-PRISM V1 PQC as a Qiskit QuantumCircuit."""
    qc = QuantumCircuit(N_QUBITS)
    h_i0, h_i1, h_j0, h_j1, d_ij = inputs

    for l in range(N_LAYERS):
        # Feature encoding (RY)
        qc.ry(float(h_i0), 0)
        qc.ry(float(h_i1), 1)
        qc.ry(float(h_j0), 2)
        qc.ry(float(h_j1), 3)

        # Distance re-uploading (RZ on all qubits)
        scaled_d = float(d_ij * dist_weights[l])
        for q in range(N_QUBITS):
            qc.rz(scaled_d, q)

        # Variational rotations (RX, RY, RZ)
        for q in range(N_QUBITS):
            qc.rx(float(weights[l, q, 0]), q)
            qc.ry(float(weights[l, q, 1]), q)
            qc.rz(float(weights[l, q, 2]), q)

        # Ring CNOT entanglement
        for q in range(N_QUBITS):
            qc.cx(q, (q + 1) % N_QUBITS)

    return qc


def get_pennylane_ideal_qnode():
    """PennyLane ideal (noiseless) QNode — matches existing benchmarks."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def circuit(inputs, weights, dist_weights):
        h_i0, h_i1, h_j0, h_j1, d_ij = (
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        )
        for l in range(N_LAYERS):
            qml.RY(h_i0, wires=0)
            qml.RY(h_i1, wires=1)
            qml.RY(h_j0, wires=2)
            qml.RY(h_j1, wires=3)
            scaled_d = d_ij * dist_weights[l]
            for q in range(N_QUBITS):
                qml.RZ(scaled_d, wires=q)
            for q in range(N_QUBITS):
                qml.RX(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)
                qml.RZ(weights[l, q, 2], wires=q)
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
        return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]

    return circuit


def get_pennylane_noisy_qnode(noise_rate):
    """PennyLane noisy QNode — matches existing depolarizing benchmarks."""
    dev = qml.device("default.mixed", wires=N_QUBITS)

    @qml.qnode(dev)
    def circuit(inputs, weights, dist_weights):
        h_i0, h_i1, h_j0, h_j1, d_ij = (
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        )
        for l in range(N_LAYERS):
            qml.RY(h_i0, wires=0)
            qml.RY(h_i1, wires=1)
            qml.RY(h_j0, wires=2)
            qml.RY(h_j1, wires=3)
            scaled_d = d_ij * dist_weights[l]
            for q in range(N_QUBITS):
                qml.RZ(scaled_d, wires=q)
            for q in range(N_QUBITS):
                qml.RX(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)
                qml.RZ(weights[l, q, 2], wires=q)
                if noise_rate > 0:
                    qml.DepolarizingChannel(noise_rate, wires=q)
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
                if noise_rate > 0:
                    qml.DepolarizingChannel(noise_rate * 2.0, wires=q)
                    qml.DepolarizingChannel(
                        noise_rate * 2.0, wires=(q + 1) % N_QUBITS
                    )
        return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]

    return circuit





def run_qiskit_ideal(qc, shots=100_000):
    """Run circuit on ideal Qiskit Aer statevector simulator."""
    sim = AerSimulator(method="statevector")
    # Compute exact expectation values via statevector
    qc_copy = qc.copy()
    qc_copy.save_statevector()
    result = sim.run(qc_copy, shots=0).result()
    sv = result.get_statevector()
    # Calculate <Z_q> for each qubit
    expvals = []
    for q in range(N_QUBITS):
        op = SparsePauliOp.from_list(
            [("I" * (N_QUBITS - 1 - q) + "Z" + "I" * q, 1.0)]
        )
        expvals.append(float(np.real(sv.expectation_value(op))))
    return np.array(expvals)


def run_qiskit_brisbane_noise(qc, noise_model, sim, shots=100_000):
    """Run 4-qubit circuit on Qiskit Aer with calibrated IBM Brisbane noise model."""
    transpiled = transpile(
        qc,
        basis_gates=noise_model.basis_gates,
        optimization_level=1,
        seed_transpiler=42,
    )

    qc_meas = transpiled.copy()
    qc_meas.measure_all()

    result = sim.run(qc_meas, shots=shots, seed_simulator=42).result()
    counts = result.get_counts()

    # Calculate <Z_q> from shot counts
    expvals = np.zeros(N_QUBITS)
    total = sum(counts.values())
    for bitstring, count in counts.items():
        for q in range(N_QUBITS):
            bit = int(bitstring[-(q + 1)])  # qubit q
            expvals[q] += (1 - 2 * bit) * count
    expvals /= total
    return expvals


def extract_brisbane_noise_stats(backend):
    """Extract and print key noise parameters from Brisbane calibration."""
    stats = {}
    
    # Coherence times (T1, T2)
    t1_vals = []
    t2_vals = []
    for q in range(min(4, backend.num_qubits)):
        try:
            qp = backend.qubit_properties(q)
            if qp:
                t1_vals.append(qp.t1 * 1e6)  # µs
                t2_vals.append(qp.t2 * 1e6)  # µs
        except Exception:
            pass

    # Single-qubit gate errors (sx)
    sx_errors = []
    if 'sx' in backend.target:
        for q in range(min(4, backend.num_qubits)):
            try:
                op_prop = backend.target['sx'].get((q,))
                if op_prop and op_prop.error is not None:
                    sx_errors.append(float(op_prop.error))
            except Exception:
                pass

    # Two-qubit gate errors (ecr or cx)
    two_q_errors = []
    for two_q_gate in ['ecr', 'cx', 'cz']:
        if two_q_gate in backend.target:
            for pair in [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]:
                try:
                    op_prop = backend.target[two_q_gate].get(pair)
                    if op_prop and op_prop.error is not None:
                        two_q_errors.append(float(op_prop.error))
                except Exception:
                    pass
            if two_q_errors:
                break

    # Readout errors
    readout_errors = []
    if 'measure' in backend.target:
        for q in range(min(4, backend.num_qubits)):
            try:
                op_prop = backend.target['measure'].get((q,))
                if op_prop and op_prop.error is not None:
                    readout_errors.append(float(op_prop.error))
            except Exception:
                pass

    stats = {
        "t1_us": t1_vals,
        "t2_us": t2_vals,
        "sx_gate_errors": sx_errors,
        "two_qubit_gate_errors": two_q_errors,
        "readout_errors": readout_errors,
        "mean_sx_error": float(np.mean(sx_errors)) if sx_errors else None,
        "mean_two_qubit_error": float(np.mean(two_q_errors)) if two_q_errors else None,
        "mean_readout_error": float(np.mean(readout_errors)) if readout_errors else None,
    }
    return stats


def run_experiment(save_dir="./results", fig_dir="./figures", num_samples=50):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("Device-realistic noise validation experiment")
    print("Comparing: Ideal vs IBM Brisbane vs Depolarizing(λ=1e-3, 1e-5)")

    # ── Fixed random inputs & weights (same seed as other benchmarks) ──
    np.random.seed(42)
    inputs_list = [
        np.random.uniform(-np.pi, np.pi, size=5) for _ in range(num_samples)
    ]
    weights = np.random.uniform(-np.pi, np.pi, size=(N_LAYERS, N_QUBITS, 3))
    dist_weights = np.array([0.1, 0.1])

    # ── 1. PennyLane ideal baseline ─────────────────────────────────────
    print("\n[1/4] Running PennyLane ideal (noiseless) simulation...")
    ideal_circuit = get_pennylane_ideal_qnode()
    ideal_vals = np.array(
        [ideal_circuit(inp, weights, dist_weights) for inp in inputs_list]
    )
    print(f"  {num_samples} samples, mean |<Z>| = {np.mean(np.abs(ideal_vals)):.4f}")

    # ── 2. PennyLane depolarizing λ=1e-3 (transmon regime) ─────────────
    print("\n[2/4] Running PennyLane depolarizing (λ=1e-3, transmon regime)...")
    depol_1e3_circuit = get_pennylane_noisy_qnode(1e-3)
    depol_1e3_vals = np.array(
        [depol_1e3_circuit(inp, weights, dist_weights) for inp in inputs_list]
    )
    err_depol_1e3 = np.mean(np.abs(depol_1e3_vals - ideal_vals))
    print(f"  Mean expectation error vs ideal: {err_depol_1e3:.6f}")

    # ── 3. PennyLane depolarizing λ=1e-5 (arm qubit regime) ────────────
    print("\n[3/4] Running PennyLane depolarizing (λ=1e-5, arm qubit regime)...")
    depol_1e5_circuit = get_pennylane_noisy_qnode(1e-5)
    depol_1e5_vals = np.array(
        [depol_1e5_circuit(inp, weights, dist_weights) for inp in inputs_list]
    )
    err_depol_1e5 = np.mean(np.abs(depol_1e5_vals - ideal_vals))
    print(f"  Mean expectation error vs ideal: {err_depol_1e5:.6f}")

    # ── 4. Qiskit Aer with IBM Brisbane noise model ────────────────────
    print("\n[4/4] Running Qiskit Aer with IBM Brisbane calibrated noise...")
    brisbane_backend = FakeBrisbane()
    noise_model = NoiseModel.from_backend(brisbane_backend)
    sim = AerSimulator(noise_model=noise_model)

    # Extract and display device noise parameters
    print("  Extracting Brisbane calibration data...")
    try:
        noise_stats = extract_brisbane_noise_stats(brisbane_backend)
        print(f"    Mean SX gate error:        {noise_stats.get('mean_sx_error', 'N/A')}")
        print(f"    Mean 2-qubit gate error:   {noise_stats.get('mean_two_qubit_error', 'N/A')}")
        print(f"    Mean readout error:        {noise_stats.get('mean_readout_error', 'N/A')}")
        if noise_stats.get("t1_us"):
            print(f"    T1 (qubits 0-3):           {['%.1f µs' % t for t in noise_stats['t1_us']]}")
        if noise_stats.get("t2_us"):
            print(f"    T2 (qubits 0-3):           {['%.1f µs' % t for t in noise_stats['t2_us']]}")
    except Exception as e:
        print(f"    (Could not extract detailed stats: {e})")
        noise_stats = {}

    brisbane_vals = np.zeros((num_samples, N_QUBITS))
    shots = 100_000

    for i, inp in enumerate(inputs_list):
        qc = build_qiskit_pqc(inp, weights, dist_weights)
        brisbane_vals[i] = run_qiskit_brisbane_noise(qc, noise_model=noise_model, sim=sim, shots=shots)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    Sample {i+1}/{num_samples} done")

    err_brisbane = np.mean(np.abs(brisbane_vals - ideal_vals))
    print(f"  Mean expectation error vs ideal: {err_brisbane:.6f}")

    # ── 5. Compute per-qubit errors for detailed analysis ──────────────
    per_qubit_err_brisbane = np.mean(np.abs(brisbane_vals - ideal_vals), axis=0)
    per_qubit_err_depol = np.mean(np.abs(depol_1e3_vals - ideal_vals), axis=0)
    per_qubit_err_arm = np.mean(np.abs(depol_1e5_vals - ideal_vals), axis=0)

    # ── 6. Summary statistics ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<35} {'Mean |ΔZ|':>12} {'Max |ΔZ|':>12}")
    print(f"  {'─' * 35} {'─' * 12} {'─' * 12}")
    print(f"  {'IBM Brisbane (calibrated noise)':<35} "
          f"{err_brisbane:>12.6f} "
          f"{np.max(np.abs(brisbane_vals - ideal_vals)):>12.6f}")
    print(f"  {'Depolarizing λ=1e-3 (transmon)':<35} "
          f"{err_depol_1e3:>12.6f} "
          f"{np.max(np.abs(depol_1e3_vals - ideal_vals)):>12.6f}")
    print(f"  {'Depolarizing λ=1e-5 (arm qubit)':<35} "
          f"{err_depol_1e5:>12.6f} "
          f"{np.max(np.abs(depol_1e5_vals - ideal_vals)):>12.6f}")

    ratio = err_brisbane / max(err_depol_1e3, 1e-10)
    print(f"\n  Brisbane / Depol(1e-3) error ratio: {ratio:.2f}x")
    if 0.3 < ratio < 3.0:
        print("  Depolarizing model is a reasonable approximation of Brisbane noise")
    elif ratio < 0.3:
        print("  Brisbane noise is less severe than depolarizing(1e-3)")
    else:
        print("  Brisbane noise is more severe than depolarizing(1e-3)")

    # ── 7. Save JSON results ───────────────────────────────────────────
    results = {
        "experiment": "device_noise_validation",
        "description": (
            "Comparison of ideal vs IBM Brisbane calibrated noise vs "
            "uniform depolarizing noise on the 4-qubit Q-PRISM PQC"
        ),
        "num_samples": num_samples,
        "n_qubits": N_QUBITS,
        "n_layers": N_LAYERS,
        "shots_brisbane": shots,
        "errors": {
            "brisbane_mean": float(err_brisbane),
            "brisbane_max": float(np.max(np.abs(brisbane_vals - ideal_vals))),
            "depol_1e3_mean": float(err_depol_1e3),
            "depol_1e3_max": float(np.max(np.abs(depol_1e3_vals - ideal_vals))),
            "depol_1e5_mean": float(err_depol_1e5),
            "depol_1e5_max": float(np.max(np.abs(depol_1e5_vals - ideal_vals))),
            "brisbane_to_depol_ratio": float(ratio),
        },
        "per_qubit_errors": {
            "brisbane": per_qubit_err_brisbane.tolist(),
            "depol_1e3": per_qubit_err_depol.tolist(),
            "depol_1e5": per_qubit_err_arm.tolist(),
        },
        "brisbane_noise_stats": noise_stats,
        "ideal_mean_abs_Z": float(np.mean(np.abs(ideal_vals))),
        "raw_data": {
            "ideal_vals": ideal_vals.tolist(),
            "brisbane_vals": brisbane_vals.tolist(),
            "depol_1e3_vals": depol_1e3_vals.tolist(),
            "depol_1e5_vals": depol_1e5_vals.tolist(),
        },
    }

    json_path = os.path.join(save_dir, "device_noise_validation.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {json_path}")

    # ── 8. Generate publication figure ─────────────────────────────────
    generate_figure(results, fig_dir)

    print("\nExperiment completed successfully.")
    return results





def generate_figure(results, fig_dir="./figures"):
    """Generate Figure 9: Device-Realistic Noise Validation."""
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)

    raw = results["raw_data"]
    ideal = np.array(raw["ideal_vals"])
    brisbane = np.array(raw["brisbane_vals"])
    depol_1e3 = np.array(raw["depol_1e3_vals"])
    depol_1e5 = np.array(raw["depol_1e5_vals"])

    # ── Panel (a): Scatter — Ideal vs Noisy expectations ───────────────
    ax = axes[0]
    ax.scatter(
        ideal.flatten(), brisbane.flatten(),
        alpha=0.4, s=15, c="#e74c3c", label="IBM Brisbane", zorder=3,
    )
    ax.scatter(
        ideal.flatten(), depol_1e3.flatten(),
        alpha=0.4, s=15, c="#3498db", label=r"Depol $\lambda=10^{-3}$",
        marker="^", zorder=2,
    )
    ax.scatter(
        ideal.flatten(), depol_1e5.flatten(),
        alpha=0.3, s=10, c="#2ecc71", label=r"Depol $\lambda=10^{-5}$",
        marker="s", zorder=1,
    )
    lims = [-1.05, 1.05]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="Ideal (y=x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(r"Ideal $\langle Z_q \rangle$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Noisy $\langle Z_q \rangle$", fontsize=11, fontweight="bold")
    ax.set_title("(a) Ideal vs Noisy Expectations", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # ── Panel (b): Per-qubit error bar comparison ──────────────────────
    ax = axes[1]
    x = np.arange(N_QUBITS)
    width = 0.25
    pq_bris = results["per_qubit_errors"]["brisbane"]
    pq_dep3 = results["per_qubit_errors"]["depol_1e3"]
    pq_dep5 = results["per_qubit_errors"]["depol_1e5"]

    bars1 = ax.bar(x - width, pq_bris, width, label="IBM Brisbane",
                   color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, pq_dep3, width, label=r"Depol $\lambda=10^{-3}$",
                   color="#3498db", alpha=0.85, edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, pq_dep5, width, label=r"Depol $\lambda=10^{-5}$",
                   color="#2ecc71", alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Qubit Index", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Mean $|\Delta \langle Z_q \rangle|$", fontsize=11,
                  fontweight="bold")
    ax.set_title("(b) Per-Qubit Expectation Error", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$q_{q}$" for q in range(N_QUBITS)])
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, axis="y", alpha=0.3)

    # ── Panel (c): Error distribution histogram ────────────────────────
    ax = axes[2]
    err_bris = np.abs(brisbane - ideal).flatten()
    err_dep3 = np.abs(depol_1e3 - ideal).flatten()
    err_dep5 = np.abs(depol_1e5 - ideal).flatten()

    bins = np.linspace(0, max(err_bris.max(), err_dep3.max()) * 1.1, 30)
    ax.hist(err_bris, bins=bins, alpha=0.6, color="#e74c3c",
            label="IBM Brisbane", density=True, edgecolor="black", linewidth=0.3)
    ax.hist(err_dep3, bins=bins, alpha=0.6, color="#3498db",
            label=r"Depol $\lambda=10^{-3}$", density=True,
            edgecolor="black", linewidth=0.3)
    ax.hist(err_dep5, bins=bins, alpha=0.4, color="#2ecc71",
            label=r"Depol $\lambda=10^{-5}$", density=True,
            edgecolor="black", linewidth=0.3)

    # Annotate means
    ax.axvline(err_bris.mean(), color="#e74c3c", linestyle="--", lw=1.5,
               label=f"Brisbane mean: {err_bris.mean():.4f}")
    ax.axvline(err_dep3.mean(), color="#3498db", linestyle="--", lw=1.5,
               label=f"Depol mean: {err_dep3.mean():.4f}")

    ax.set_xlabel(r"Expectation Error $|\Delta \langle Z \rangle|$",
                  fontsize=11, fontweight="bold")
    ax.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax.set_title("(c) Error Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, frameon=True, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "figure9_device_noise_validation.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"\nSaved Figure 9 to: {fig_path}")


if __name__ == "__main__":
    run_experiment(
        save_dir="./results",
        fig_dir="./figures",
        num_samples=50,
    )
