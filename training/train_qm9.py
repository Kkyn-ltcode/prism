"""
Training & Evaluation Engine for Q-PRISM & Classical Baselines on QM9
=====================================================================
Automated training sweeps, loss logging, and MAE reporting in physical units.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from datasets.dataset_qm9 import get_qm9_splits, QM9_TARGET_INDICES
from models.eq_gnn import QPRISMModel, EQGNNModel
from models.classical_trig_twin import PRISMModel
from models.egnn_baseline import EGNNModel


def train_one_epoch(model, loader, optimizer, criterion, mean, std, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Normalized target for stable training
        y_norm = (batch.y - mean) / std

        pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
        loss = criterion(pred, y_norm)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, mean, std, device):
    model.eval()
    total_mae = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        pred_norm = model(batch.x, batch.pos, batch.edge_index, batch.batch)
        
        # Denormalize to true physical units (eV, Debye, etc.)
        pred = pred_norm * std + mean
        mae = torch.abs(pred - batch.y).sum().item()

        total_mae += mae
        total_samples += batch.num_graphs

    return total_mae / total_samples


def main():
    parser = argparse.ArgumentParser(description="Train Q-PRISM, PRISM, or EGNN on QM9")
    parser.add_argument("--model", type=str, default="q_prism", choices=["q_prism", "prism", "eqgnn", "egnn"])
    parser.add_argument("--target", type=str, default="gap", choices=list(QM9_TARGET_INDICES.keys()))
    parser.add_argument("--num-train", type=int, default=100)
    parser.add_argument("--num-val", type=int, default=50)
    parser.add_argument("--num-test", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--q-layers", type=int, default=2, help="Number of variational layers in quantum circuit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default="./data/QM9")
    parser.add_argument("--save-dir", type=str, default="./results")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cpu")  # Quantum simulator default.qubit runs on CPU / lightning

    model_display = {
        "q_prism": "Q-PRISM (Quantum PQC)",
        "eqgnn": "Q-PRISM (EQ-GNN legacy)",
        "prism": "PRISM (Classical Periodic SIREN)",
        "egnn": "EGNN (Classical baseline)",
    }.get(args.model, args.model.upper())

    print("=" * 65)
    print(f"🚀 Starting Benchmark: Model={model_display} | Target={args.target} | N_train={args.num_train}")
    print(f"   Epochs={args.epochs} | BatchSize={args.batch_size} | LR={args.lr} | HiddenDim={args.hidden_dim}")
    print("=" * 65)

    # 1. Load Data
    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root=args.data_root,
        target=args.target,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        seed=args.seed
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # 2. Instantiate Model
    if args.model in ("q_prism", "eqgnn"):
        model = QPRISMModel(
            in_features=11,
            hidden_features=args.hidden_dim,
            num_layers=args.num_layers,
            n_q_layers=args.q_layers,
            out_dim=1
        ).to(device)
    elif args.model == "prism":
        model = PRISMModel(
            in_features=11,
            hidden_features=args.hidden_dim,
            num_layers=args.num_layers,
            n_q_layers=args.q_layers,
            out_dim=1,
            head_type="siren",
        ).to(device)
    else:
        model = EGNNModel(
            in_features=11,
            hidden_features=args.hidden_dim,
            num_layers=args.num_layers,
            out_dim=1
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model Architecture: {model_display} with {total_params:,} trainable parameters.")

    # 3. Optimizer & Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    # 4. Training Loop
    history = {"train_loss": [], "val_mae": [], "epoch_times": []}
    best_val_mae = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, mean, std, device)
        val_mae = evaluate(model, val_loader, mean, std, device)
        scheduler.step()
        ep_time = time.time() - ep_start

        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)
        history["epoch_times"].append(ep_time)

        if val_mae < best_val_mae:
            best_val_mae = val_mae

        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{ep_time:.1f}s] - Train Loss (MSE): {train_loss:.4f} | Val MAE: {val_mae:.4f}")

    total_training_time = time.time() - start_time
    test_mae = evaluate(model, test_loader, mean, std, device)

    unit = "eV" if args.target in ["gap", "homo", "lumo"] else ("Debye" if args.target == "mu" else "Hartree")
    print("=" * 65)
    print(f"🏁 Benchmark Complete in {total_training_time:.1f}s!")
    print(f"   Best Val MAE: {best_val_mae:.4f} {unit}")
    print(f"   Final Test MAE: {test_mae:.4f} {unit}")
    print("=" * 65)

    # 5. Save Results
    result_record = {
        "model": args.model,
        "target": args.target,
        "unit": unit,
        "num_train": args.num_train,
        "num_val": args.num_val,
        "num_test": args.num_test,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "total_params": total_params,
        "best_val_mae": best_val_mae,
        "test_mae": test_mae,
        "total_time_seconds": total_training_time,
        "history": history
    }

    result_filename = os.path.join(args.save_dir, f"{args.model}_{args.target}_N{args.num_train}.json")
    with open(result_filename, "w") as f:
        json.dump(result_record, f, indent=2)
    print(f"Saved benchmark summary to: {result_filename}")


if __name__ == "__main__":
    main()
