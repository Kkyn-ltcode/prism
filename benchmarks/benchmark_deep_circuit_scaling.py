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
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader

from datasets.dataset_qm9 import get_qm9_splits
from benchmarks.benchmark_ablations import AblationEQGNNModelV2, train_eval_model

def compute_noise_mae(ideal_mae, L, sensitivity, lam, zeta_hz):
    mean_Z = 0.5  # Approximation for mean(|<Z>_ideal|)
    n_channels_per_layer = 16
    n_total = n_channels_per_layer * L
    
    # Depolarizing error
    expectation_error_depol = mean_Z * (1 - (1 - 4*lam/3)**n_total)
    
    # ZZ Crosstalk error
    dt = 200e-9
    total_phase = zeta_hz * dt * L
    expectation_error_zz = np.sin(total_phase / 2)**2 * mean_Z
    
    total_expectation_error = expectation_error_depol + expectation_error_zz
    noise_mae = ideal_mae + total_expectation_error * sensitivity
    return noise_mae

def main():
    print("========================================")
    print("🚀 Running Deep Circuit Scaling Benchmark")
    print("========================================")
    
    os.makedirs("./results", exist_ok=True)
    
    L_values = [1, 2, 4, 8, 16]
    targets = ["gap", "u0"]
    sensitivities = {"gap": 0.8, "u0": 100.0}
    
    results = {
        "L_values": L_values,
        "gap_mae_ideal": [],
        "gap_mae_arm": [],
        "gap_mae_transmon": [],
        "u0_mae_ideal": [],
        "u0_mae_arm": [],
        "u0_mae_transmon": [],
        "param_counts": [],
        "quantum_param_counts": [],
        "circuit_time_arm_ns": [],
        "circuit_time_ibm_ns": [],
        "coherence_ratio_arm": [],
        "coherence_ratio_ibm": []
    }
    
    for L in L_values:
        # Timing
        n_single = L * 40
        n_two = L * 8
        time_arm = n_single * 5e-9 + n_two * 17e-9
        time_ibm = n_single * 25e-9 + n_two * 300e-9
        results["circuit_time_arm_ns"].append(time_arm * 1e9)
        results["circuit_time_ibm_ns"].append(time_ibm * 1e9)
        results["coherence_ratio_arm"].append(time_arm / 380e-6)
        results["coherence_ratio_ibm"].append(time_ibm / 50e-6)
    
    for target in targets:
        print(f"\nEvaluating target: {target}")
        train_set, val_set, test_set, (mean, std) = get_qm9_splits(
            root="./data/QM9", target=target, num_train=250, num_val=100, num_test=250, seed=42
        )
        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
        
        for i, L in enumerate(L_values):
            print(f"  Training L={L}...")
            # Set seed for reproducibility
            torch.manual_seed(42)
            np.random.seed(42)
            
            model = AblationEQGNNModelV2(
                in_features=11, hidden_features=16, num_layers=2, n_q_layers=L,
                use_rbf=True, n_heads=2, use_correlators=True, use_angles=True, use_tanh_gate=True
            )
            
            if target == "gap":
                # Record params only once
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                # Count quantum params (weights in q_edge_layer)
                q_params = 0
                for name, param in model.named_parameters():
                    if "q_edge_layer" in name:
                        q_params += param.numel()
                results["param_counts"].append(total_params)
                results["quantum_param_counts"].append(q_params)
            
            ideal_mae = train_eval_model(model, train_loader, val_loader, test_loader, mean, std, epochs=15, lr=1e-3)
            
            # Analytical Noise Degradation
            sensitivity = sensitivities[target]
            
            # Arm Qubit
            lam_arm = 1e-5
            zeta_arm = 2 * np.pi * 0.4e3
            arm_mae = compute_noise_mae(ideal_mae, L, sensitivity, lam_arm, zeta_arm)
            
            # Transmon
            lam_transmon = 1e-3
            zeta_transmon = 2 * np.pi * 50e3
            transmon_mae = compute_noise_mae(ideal_mae, L, sensitivity, lam_transmon, zeta_transmon)
            
            results[f"{target}_mae_ideal"].append(ideal_mae)
            results[f"{target}_mae_arm"].append(arm_mae)
            results[f"{target}_mae_transmon"].append(transmon_mae)
            
            print(f"    Ideal MAE: {ideal_mae:.4f} | Arm: {arm_mae:.4f} | Transmon: {transmon_mae:.4f}")

    with open("./results/deep_circuit_scaling.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n✅ Saved results to ./results/deep_circuit_scaling.json")

def plot_deep_circuit_scaling(results, fig_path="./results/figure7_deep_circuit_scaling.png"):
    L_values = results["L_values"]
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
    
    # (a) Gap MAE vs Depth
    ax = axes[0, 0]
    ax.plot(L_values, results["gap_mae_ideal"], marker='o', color='blue', label='Ideal')
    ax.plot(L_values, results["gap_mae_arm"], marker='^', color='green', label='Arm Qubit')
    ax.plot(L_values, results["gap_mae_transmon"], marker='s', color='red', label='Transmon')
    
    # Mark existing ablation reference point at L=2 with a star
    idx_L2 = L_values.index(2)
    ax.plot(2, results["gap_mae_ideal"][idx_L2], marker='*', color='gold', markersize=15, linestyle='None', label='Baseline Ref')
    
    ax.set_xlabel("Circuit Depth (L)", fontweight="bold")
    ax.set_ylabel("Test MAE (eV)", fontweight="bold")
    ax.set_title("(a) HOMO-LUMO Gap MAE vs Circuit Depth", fontweight="bold")
    ax.set_xticks(L_values)
    ax.legend()
    
    # (b) U0 MAE vs Depth
    ax = axes[0, 1]
    ax.plot(L_values, results["u0_mae_ideal"], marker='o', color='blue', label='Ideal')
    ax.plot(L_values, results["u0_mae_arm"], marker='^', color='green', label='Arm Qubit')
    ax.plot(L_values, results["u0_mae_transmon"], marker='s', color='red', label='Transmon')
    
    ax.set_xlabel("Circuit Depth (L)", fontweight="bold")
    ax.set_ylabel("Test MAE (eV)", fontweight="bold")
    ax.set_title(r"(b) Internal Energy $U_0$ MAE vs Circuit Depth", fontweight="bold")
    ax.set_xticks(L_values)
    ax.legend()
    
    # (c) Parameter Count vs Depth
    ax = axes[1, 0]
    x = np.arange(len(L_values))
    width = 0.35
    ax.bar(x - width/2, results["quantum_param_counts"], width, label='Quantum Params', color='#1f77b4')
    ax.bar(x + width/2, results["param_counts"], width, label='Total Params', color='#ff7f0e')
    
    ax.set_xlabel("Circuit Depth (L)", fontweight="bold")
    ax.set_ylabel("Parameter Count", fontweight="bold")
    ax.set_title("(c) Model Capacity Scaling", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(L_values)
    ax.legend()
    
    # (d) Circuit-to-Coherence Ratio
    ax = axes[1, 1]
    ax.plot(L_values, results["coherence_ratio_arm"], marker='o', color='green', label=r'Arm Qubit $\tau/T_2^E$')
    ax.plot(L_values, results["coherence_ratio_ibm"], marker='s', color='red', label=r'IBM Transmon $\tau/T_2$')
    ax.set_yscale('log')
    ax.axhline(0.01, color='green', linestyle='--', label='1% Threshold')
    ax.axhline(0.10, color='red', linestyle='--', label='10% Threshold')
    ax.fill_between(L_values, 1e-6, 0.01, color='green', alpha=0.1)
    ax.fill_between(L_values, 0.10, max(results["coherence_ratio_ibm"]) * 10, color='red', alpha=0.1)
    
    ax.set_xlabel("Circuit Depth (L)", fontweight="bold")
    ax.set_ylabel("Circuit-to-Coherence Ratio", fontweight="bold")
    ax.set_title("(d) Circuit-to-Coherence Budget", fontweight="bold")
    ax.set_xticks(L_values)
    ax.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path)
    plt.close()
    print(f"📊 Saved Figure 7 to: {fig_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="Re-generate plots from existing JSON results")
    args = parser.parse_args()
    
    if args.plot_only:
        with open("./results/deep_circuit_scaling.json", "r") as f:
            data = json.load(f)
        plot_deep_circuit_scaling(data)
        print("✅ Plot re-generated successfully!")
    else:
        main()
