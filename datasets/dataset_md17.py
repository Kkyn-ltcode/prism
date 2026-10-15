"""MD17 molecular dynamics dataset pipeline."""

import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader


# Atomic number vocabulary for one-hot encoding (covers H=1, C=6, N=7, O=8)
MAX_ATOMIC_NUM = 10


def one_hot_atomic(z, max_z=MAX_ATOMIC_NUM):
    """One-hot encode atomic numbers."""
    one_hot = torch.zeros(z.size(0), max_z, dtype=torch.float32)
    # Clamp z to valid range
    z_clamped = z.clamp(0, max_z - 1).long()
    one_hot.scatter_(1, z_clamped.unsqueeze(-1), 1.0)
    return one_hot


def build_knn_edges(pos, k=8):
    """Build k-nearest neighbor graph edges from positions."""
    n = pos.size(0)
    # If molecule is small enough, use fully connected
    if n <= k + 1:
        # Fully connected (excluding self-loops)
        adj = torch.ones(n, n, dtype=torch.long) - torch.eye(n, dtype=torch.long)
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        return edge_index

    # Pairwise distances
    dist = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)  # [n, n]
    # Set self-distance to infinity
    dist.fill_diagonal_(float('inf'))
    # Get k nearest neighbors for each atom
    _, knn_idx = dist.topk(k, largest=False, dim=-1)  # [n, k]

    # Build edge index
    row = torch.arange(n).unsqueeze(-1).expand(-1, k).reshape(-1)
    col = knn_idx.reshape(-1)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


def prepare_md17_data(molecule_name='ethanol', root='./data/MD17', k_nn=8):
    """Load MD17 dataset and prepare graph data with one-hot features."""
    print(f"Loading MD17 '{molecule_name}' dataset from '{root}'...")
    dataset = MD17(root=root, name=molecule_name)
    print(f"Total MD17 '{molecule_name}' conformations: {len(dataset)}")

    # Get atomic numbers from first sample (same for all conformations)
    z = dataset[0].z
    node_features = one_hot_atomic(z)  # [N_atoms, MAX_ATOMIC_NUM]
    n_atoms = z.size(0)

    # Build edge index once (topology is fixed for all conformations of same molecule)
    # Use first conformation's positions as reference for k-NN
    ref_edge_index = build_knn_edges(dataset[0].pos, k=k_nn)

    print(f"  Atoms: {n_atoms}, Edges: {ref_edge_index.shape[1]}, Node features: {node_features.shape[1]}")
    print(f"  Atomic numbers: {z.tolist()}")

    return dataset, node_features, ref_edge_index, n_atoms


def get_md17_splits(molecule_name='ethanol', root='./data/MD17',
                    num_train=1000, num_val=250, num_test=500,
                    seed=42, k_nn=8):
    """Create train/val/test splits from MD17 conformations."""
    dataset, node_features, ref_edge_index, n_atoms = prepare_md17_data(
        molecule_name=molecule_name, root=root, k_nn=k_nn
    )

    total_needed = num_train + num_val + num_test
    assert total_needed <= len(dataset), f"Requested {total_needed} samples but only {len(dataset)} available"

    # Random permutation for split
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(dataset))[:total_needed]

    train_indices = perm[:num_train]
    val_indices = perm[num_train:num_train + num_val]
    test_indices = perm[num_train + num_val:num_train + num_val + num_test]

    # Compute training energy statistics for normalization
    train_energies = torch.stack([dataset[int(i)].energy for i in train_indices])
    energy_mean = train_energies.mean().item()
    energy_std = train_energies.std().item()
    print(f"  Energy Train Stats: Mean={energy_mean:.4f}, Std={energy_std:.4f} kcal/mol")

    def make_data_list(indices):
        data_list = []
        for idx in indices:
            sample = dataset[int(idx)]
            # Rebuild edge_index per conformation (positions change geometry slightly)
            edge_index = build_knn_edges(sample.pos, k=k_nn)

            data = Data(
                x=node_features.clone(),           # [N_atoms, MAX_ATOMIC_NUM]
                pos=sample.pos.clone(),             # [N_atoms, 3]
                edge_index=edge_index,              # [2, E]
                energy=sample.energy.clone(),       # [1] raw energy
                force=sample.force.clone(),         # [N_atoms, 3] raw forces
                energy_norm=(sample.energy.clone() - energy_mean) / max(energy_std, 1e-8),  # Normalized energy
            )
            data_list.append(data)
        return data_list

    train_data = make_data_list(train_indices)
    val_data = make_data_list(val_indices)
    test_data = make_data_list(test_indices)

    print(f"  Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    return train_data, val_data, test_data, energy_mean, energy_std


if __name__ == "__main__":
    for mol in ['ethanol', 'aspirin']:
        print(f"\nTesting MD17 pipeline for: {mol}")
        train, val, test, e_mean, e_std = get_md17_splits(
            molecule_name=mol, num_train=100, num_val=50, num_test=50
        )
        s = train[0]
        print(f"  Sample: x={s.x.shape}, pos={s.pos.shape}, energy={s.energy.shape}, "
              f"force={s.force.shape}, edge_index={s.edge_index.shape}")
        print(f"  Energy mean={e_mean:.4f}, std={e_std:.4f}")
    print("\nMD17 dataset pipeline verified.")
