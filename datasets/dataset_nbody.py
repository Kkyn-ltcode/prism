"""3D charged N-body particle system generator and dataset pipeline."""

import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def simulate_charged_trajectory(n_particles=5, timesteps=1000, dt=0.001, loc_std=1.0, vel_std=0.5):
    """Simulate a single 5-particle trajectory under 3D Coulomb force laws."""
    pos = np.random.normal(0, loc_std, size=(n_particles, 3))
    vel = np.random.normal(0, vel_std, size=(n_particles, 3))
    
    # Remove center of mass velocity to ensure system is bound
    vel -= np.mean(vel, axis=0, keepdims=True)
    
    # Random discrete charges in {-1, +1}
    charges = np.random.choice([-1.0, 1.0], size=(n_particles, 1))

    # Velocity Verlet numerical integration
    def compute_forces(x):
        forces = np.zeros_like(x)
        for i in range(n_particles):
            for j in range(n_particles):
                if i != j:
                    diff = x[i] - x[j]
                    dist_sq = np.sum(diff**2) + 1e-3  # Softening parameter to prevent division by zero
                    dist = np.sqrt(dist_sq)
                    # Coulomb law: F_ij = q_i * q_j * r_ij / r_ij^3
                    f_ij = charges[i, 0] * charges[j, 0] * diff / (dist**3)
                    forces[i] += f_ij
        return forces

    forces = compute_forces(pos)
    for _ in range(timesteps):
        pos += vel * dt + 0.5 * forces * (dt**2)
        new_forces = compute_forces(pos)
        vel += 0.5 * (forces + new_forces) * dt
        forces = new_forces

    return pos


def generate_nbody_dataset(n_samples=2000, n_particles=5, timesteps=1000, dt=0.001, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    data_list = []
    
    # Fully connected edge index for 5 particles (excluding self-loops)
    adj = torch.ones(n_particles, n_particles, dtype=torch.long) - torch.eye(n_particles, dtype=torch.long)
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()

    for idx in range(n_samples):
        # Initial state
        init_pos = np.random.normal(0, 1.0, size=(n_particles, 3))
        init_vel = np.random.normal(0, 0.5, size=(n_particles, 3))
        init_vel -= np.mean(init_vel, axis=0, keepdims=True)
        charges = np.random.choice([-1.0, 1.0], size=(n_particles, 1))

        # Integrate forward to get final position target
        # Local simulation for this sample
        pos = init_pos.copy()
        vel = init_vel.copy()

        def compute_forces(x):
            forces = np.zeros_like(x)
            for i in range(n_particles):
                for j in range(n_particles):
                    if i != j:
                        diff = x[i] - x[j]
                        dist_sq = np.sum(diff**2) + 1e-3
                        dist = np.sqrt(dist_sq)
                        f_ij = charges[i, 0] * charges[j, 0] * diff / (dist**3)
                        forces[i] += f_ij
            return forces

        forces = compute_forces(pos)
        for _ in range(timesteps):
            pos += vel * dt + 0.5 * forces * (dt**2)
            new_forces = compute_forces(pos)
            vel += 0.5 * (forces + new_forces) * dt
            forces = new_forces

        target_pos = pos

        # Invariant Node Features: [charges, speed^2]
        speed_sq = np.sum(init_vel**2, axis=-1, keepdims=True)
        node_feat = np.concatenate([charges, speed_sq], axis=-1)  # [5, 2]

        sample_data = Data(
            x=torch.tensor(node_feat, dtype=torch.float32),      # [5, 2] Invariant features
            pos=torch.tensor(init_pos, dtype=torch.float32),     # [5, 3] Initial coordinates
            vel=torch.tensor(init_vel, dtype=torch.float32),     # [5, 3] Initial velocities
            target_pos=torch.tensor(target_pos, dtype=torch.float32), # [5, 3] Target future position
            edge_index=edge_index                                # [2, 20] Graph clique
        )
        data_list.append(sample_data)

    return data_list


def get_nbody_splits(root="./data/nbody", num_train=1000, num_val=250, num_test=500, seed=42):
    os.makedirs(root, exist_ok=True)
    cache_file = os.path.join(root, "nbody_5p_dataset.pt")

    total_samples = num_train + num_val + num_test
    if os.path.exists(cache_file):
        print(f"Loading cached N-Body dataset from '{cache_file}'...")
        try:
            data_list = torch.load(cache_file, weights_only=False)
        except Exception:
            data_list = generate_nbody_dataset(n_samples=total_samples, seed=seed)
            torch.save(data_list, cache_file)
        if len(data_list) < total_samples:
            print("Cached dataset smaller than requested, regenerating...")
            data_list = generate_nbody_dataset(n_samples=total_samples, seed=seed)
            torch.save(data_list, cache_file)
    else:
        print(f"Generating 3D Charged N-Body dataset ({total_samples} simulations)...")
        data_list = generate_nbody_dataset(n_samples=total_samples, seed=seed)
        torch.save(data_list, cache_file)
        print(f"Saved generated N-Body dataset to '{cache_file}'.")

    torch.manual_seed(seed)
    perm = torch.randperm(len(data_list))

    train_data = [data_list[i] for i in perm[:num_train]]
    val_data = [data_list[i] for i in perm[num_train : num_train + num_val]]
    test_data = [data_list[i] for i in perm[num_train + num_val : num_train + num_val + num_test]]

    print(f"N-Body Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data


if __name__ == "__main__":
    train_data, val_data, test_data = get_nbody_splits(num_train=100, num_val=50, num_test=50)
    sample = train_data[0]
    print(f"Sample: x={sample.x.shape}, pos={sample.pos.shape}, target_pos={sample.target_pos.shape}, edge_index={sample.edge_index.shape}")
    print("3D charged N-body dataset generator verified.")
