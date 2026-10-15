"""QM9 molecular dataset pipeline for spatial graph processing."""

import os
import argparse
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

QM9_TARGET_INDICES = {
    "mu": 0,          # Dipole moment (Debye)
    "alpha": 1,       # Isotropic polarizability (Bohr^3)
    "homo": 2,        # HOMO energy (Hartree)
    "lumo": 3,        # LUMO energy (Hartree)
    "gap": 4,         # HOMO-LUMO gap (Hartree)
    "r2": 5,          # Electronic spatial extent (Bohr^2)
    "zpve": 6,        # Zero point vibrational energy (Hartree)
    "u0": 7,          # Internal energy at 0K (Hartree)
    "u": 8,           # Internal energy at 298.15K (Hartree)
    "h": 9,           # Enthalpy at 298.15K (Hartree)
    "g": 10,          # Free energy at 298.15K (Hartree)
    "cv": 11,         # Heat capacity (cal/(mol K))
}

# Conversion factor from Hartree to eV for gap, homo, lumo
HARTREE_TO_EV = 27.211386245988

QM9_ELECTRONIC_TARGETS = ["homo", "lumo", "gap"]

def build_fully_connected_edges(num_nodes):
    """Create bidirectional edges for a fully connected molecular clique."""
    adj = torch.ones(num_nodes, num_nodes, dtype=torch.long) - torch.eye(num_nodes, dtype=torch.long)
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    return edge_index

def build_knn_edges(pos, k=8):
    """Build sparse k-nearest-neighbor edges for molecular systems."""
    N = pos.size(0)
    k_actual = min(k, N - 1)  # Cannot have more neighbors than other nodes

    # Pairwise squared distances [N, N]
    dist = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)
    # Set self-distance to infinity to exclude self-loops
    dist.fill_diagonal_(float("inf"))

    # k-nearest neighbors per node
    _, indices = dist.topk(k_actual, dim=-1, largest=False)  # [N, k]

    # Build edge index: source -> target (directed)
    row = torch.arange(N, device=pos.device).unsqueeze(1).expand(-1, k_actual).flatten()
    col = indices.flatten()
    edge_index = torch.stack([row, col], dim=0)  # [2, N*k]

    # Add reverse edges for bidirectionality and deduplicate
    edge_index_rev = edge_index.flip(0)
    edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    return edge_index

class ProcessedQM9Subset:
    """In-memory dataset wrapper for QM9."""
    def __init__(self, raw_data_list, target_name="gap"):
        self.data_list = []
        target_idx = QM9_TARGET_INDICES[target_name]

        for item in raw_data_list:
            n_nodes = item.x.size(0)
            edge_index = build_fully_connected_edges(n_nodes)
            y_val = item.y[:, target_idx:target_idx+1]

            mol_data = Data(
                x=item.x.float(),       # [N, 11]
                pos=item.pos.float(),   # [N, 3]
                edge_index=edge_index,  # [2, E]
                y=y_val.float()         # [1, 1]
            )
            self.data_list.append(mol_data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def get_qm9_splits(root="./data/QM9", target="gap", num_train=1000, num_val=500, num_test=1000, seed=42):
    """Load QM9 dataset and create standardized train/val/test splits."""
    print(f"Loading QM9 dataset from '{root}'...")
    raw_dataset = QM9(root=root)
    total_len = len(raw_dataset)
    print(f"Total QM9 molecules available: {total_len}")

    torch.manual_seed(seed)
    perm = torch.randperm(total_len)

    train_indices = perm[:num_train]
    val_indices = perm[num_train : num_train + num_val]
    test_indices = perm[num_train + num_val : num_train + num_val + num_test]

    train_raw = [raw_dataset[i] for i in train_indices]
    val_raw = [raw_dataset[i] for i in val_indices]
    test_raw = [raw_dataset[i] for i in test_indices]

    train_set = ProcessedQM9Subset(train_raw, target_name=target)
    val_set = ProcessedQM9Subset(val_raw, target_name=target)
    test_set = ProcessedQM9Subset(test_raw, target_name=target)

    # Compute target normalization statistics on training split
    train_y = torch.cat([d.y for d in train_set.data_list], dim=0)
    mean = train_y.mean().item()
    std = train_y.std().item()

    print(f"Split Summary: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    print(f"Target '{target}' Train Stats: Mean={mean:.4f}, Std={std:.4f}")

    return train_set, val_set, test_set, (mean, std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test QM9 Dataset Loading")
    parser.add_argument("--root", type=str, default="./data/QM9")
    parser.add_argument("--target", type=str, default="gap", choices=list(QM9_TARGET_INDICES.keys()))
    parser.add_argument("--num-train", type=int, default=100)
    parser.add_argument("--num-val", type=int, default=50)
    parser.add_argument("--num-test", type=int, default=50)
    args = parser.parse_args()

    train_data, val_data, test_data, stats = get_qm9_splits(
        root=args.root,
        target=args.target,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test
    )

    loader = DataLoader(train_data, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    print(f"Sample Batch: x={batch.x.shape}, pos={batch.pos.shape}, edge_index={batch.edge_index.shape}, y={batch.y.shape}, batch={batch.batch.shape}")
    print("QM9 data pipeline functional and verified.")
