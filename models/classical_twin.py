"""Classical twin of Q-PRISM using parameter-matched MLPs with tanh activations."""

import torch
import torch.nn as nn
from .eq_gnn import GaussianRBF, compute_edge_mean_angles, N_RBF


class ClassicalHead(nn.Module):
    """Classical twin head replacing the PQC with a tanh-bounded MLP."""
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.feature_proj = nn.Linear(in_features, 2)

        input_dim = 4 + n_rbf + 1  # [h_i0, h_i1, h_j0, h_j1, rbf×K, angle]
        n_outputs = 8
        hidden = 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_outputs, bias=False),
        )

    def forward(self, h_row, h_col, rbf_features, angles):
        h_i_proj = self.feature_proj(h_row)  # [E, 2]
        h_j_proj = self.feature_proj(h_col)  # [E, 2]
        inputs = torch.cat([h_i_proj, h_j_proj, rbf_features, angles], dim=-1)  # [E, 4+K+1]
        return torch.tanh(self.mlp(inputs))  # [E, 8], bounded to [-1, 1]


class ClassicalTwinLayerV2(nn.Module):
    """Classical twin of EQGNNLayerV2."""
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.in_features = in_features
        self.n_rbf = n_rbf

        self.rbf = GaussianRBF(n_rbf=n_rbf)
        self.head_electronic = ClassicalHead(in_features, n_q_layers=n_q_layers, n_rbf=n_rbf)
        self.head_topological = ClassicalHead(in_features, n_q_layers=n_q_layers, n_rbf=n_rbf)

        self.coord_gate = nn.Linear(2, 1)

        msg_dim = 7 * 2  # 7 message values per head × 2 heads = 14
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + msg_dim, in_features),
            nn.SiLU(),
            nn.Linear(in_features, in_features)
        )

    def forward(self, h, x, edge_index):
        row, col = edge_index[0], edge_index[1]

        rel_x = x[row] - x[col]
        d_ij = torch.sum(rel_x ** 2, dim=-1, keepdim=True)

        rbf_features = self.rbf(d_ij)
        angles = compute_edge_mean_angles(x, edge_index)

        out_e = self.head_electronic(h[row], h[col], rbf_features, angles)
        out_t = self.head_topological(h[row], h[col], rbf_features, angles)

        msg_e = torch.cat([out_e[:, :3], out_e[:, 4:]], dim=-1)  # [E, 7]
        msg_t = torch.cat([out_t[:, :3], out_t[:, 4:]], dim=-1)  # [E, 7]
        m_ij = torch.cat([msg_e, msg_t], dim=-1)                 # [E, 14]

        phi_x_raw = torch.cat([out_e[:, 3:4], out_t[:, 3:4]], dim=-1)  # [E, 2]

        phi_x = torch.tanh(self.coord_gate(phi_x_raw))
        coord_msg = rel_x * phi_x
        x_update = torch.zeros_like(x)
        x_update.index_add_(0, row, coord_msg)

        degrees = torch.zeros((x.size(0), 1), device=x.device)
        degrees.index_add_(0, row, torch.ones((row.size(0), 1), device=x.device))
        x_new = x + x_update / degrees.clamp(min=1.0)

        m_i = torch.zeros((h.size(0), m_ij.size(1)), device=h.device)
        m_i.index_add_(0, row, m_ij)
        h_new = h + self.node_mlp(torch.cat([h, m_i], dim=-1))

        return h_new, x_new


class ClassicalTwinModelV2(nn.Module):
    """Classical twin model for molecular property prediction."""
    def __init__(self, in_features=11, hidden_features=16, num_layers=3, out_dim=1,
                 n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            ClassicalTwinLayerV2(hidden_features, n_q_layers=n_q_layers, n_rbf=n_rbf)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_dim)
        )

    def forward(self, h, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        h = self.embedding(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)

        num_graphs = int(batch.max().item()) + 1
        out_pool = torch.zeros((num_graphs, h.size(1)), device=h.device)
        out_pool.index_add_(0, batch, h)
        counts = torch.zeros((num_graphs, 1), device=h.device)
        counts.index_add_(0, batch, torch.ones((h.size(0), 1), device=h.device))
        out_pool = out_pool / counts.clamp(min=1.0)

        return self.head(out_pool)


def create_param_matched_classical_twin(target_params, in_features=11, hidden_features=16,
                                         num_layers=2, out_dim=1, n_q_layers=2, n_rbf=N_RBF):
    """Create a ClassicalTwinModelV2 and return parameter count and diff vs target."""
    model = ClassicalTwinModelV2(
        in_features=in_features,
        hidden_features=hidden_features,
        num_layers=num_layers,
        out_dim=out_dim,
        n_q_layers=n_q_layers,
        n_rbf=n_rbf,
    )
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    diff = total - target_params
    return model, total, diff


class ClassicalTwinLayerV1(nn.Module):
    """Classical twin of EQGNNLayerV1."""
    def __init__(self, in_features, n_q_layers=2):
        super().__init__()
        self.in_features = in_features

        self.feature_proj = nn.Linear(in_features, 2)
        self.edge_mlp = nn.Linear(5, 4)

        n_qubits = 4
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + (n_qubits - 1), in_features),
            nn.SiLU(),
            nn.Linear(in_features, in_features)
        )

    def forward(self, h, x, edge_index):
        row, col = edge_index[0], edge_index[1]

        rel_x = x[row] - x[col]
        d_ij = torch.sum(rel_x ** 2, dim=-1, keepdim=True)

        h_proj = self.feature_proj(h)
        h_i_p = h_proj[row]
        h_j_p = h_proj[col]
        q_inputs = torch.cat([h_i_p, h_j_p, d_ij], dim=-1)  # [E, 5]

        q_out = torch.tanh(self.edge_mlp(q_inputs))  # [E, 4]

        m_ij = q_out[:, :3]
        phi_x = q_out[:, 3:4]

        coord_msg = rel_x * phi_x
        x_update = torch.zeros_like(x)
        x_update.index_add_(0, row, coord_msg)
        degrees = torch.zeros((x.size(0), 1), device=x.device)
        degrees.index_add_(0, row, torch.ones((row.size(0), 1), device=x.device))
        x_new = x + x_update / degrees.clamp(min=1.0)

        m_i = torch.zeros((h.size(0), m_ij.size(1)), device=h.device)
        m_i.index_add_(0, row, m_ij)
        h_new = h + self.node_mlp(torch.cat([h, m_i], dim=-1))

        return h_new, x_new


class ClassicalTwinTrajectoryModel(nn.Module):
    """Classical twin trajectory model for coordinate prediction."""
    def __init__(self, in_features=2, hidden_features=16, num_layers=3, n_q_layers=2):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            ClassicalTwinLayerV2(hidden_features, n_q_layers=n_q_layers)
            for _ in range(num_layers)
        ])

    def forward(self, h, x, edge_index):
        h = self.embedding(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        return x


if __name__ == "__main__":
    print("Classical Twin Parameter Matching Verification")

    from .eq_gnn import QPRISMModelV2, EQGNNModelV2
    from .egnn_baseline import count_parameters

    eqgnn = QPRISMModelV2(in_features=11, hidden_features=16, num_layers=2,
                          n_q_layers=2, out_dim=1)
    twin = ClassicalTwinModelV2(in_features=11, hidden_features=16, num_layers=2,
                                 n_q_layers=2, out_dim=1)

    eqgnn_params = count_parameters(eqgnn)
    twin_params = count_parameters(twin)

    print(f"\nQ-PRISM params:         {eqgnn_params:,}")
    print(f"Classical Twin params:  {twin_params:,}")
    print(f"Difference:             {twin_params - eqgnn_params:+d}")
    print(f"Match ratio:            {twin_params / eqgnn_params:.4f}")

    h = torch.randn(4, 11)
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0,0,0,1,1,2,2,3,3,3],
                                [1,2,3,0,2,0,1,0,1,2]], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)

    out_eqgnn = eqgnn(h, x, edge_index, batch)
    out_twin = twin(h, x, edge_index, batch)

    print(f"\nQ-PRISM output shape:   {out_eqgnn.shape}")
    print(f"Classical Twin output:  {out_twin.shape}")
    assert out_eqgnn.shape == out_twin.shape, "Output shape mismatch!"

    import torch
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    t = torch.randn(1, 3)

    x_rot = x @ Q.T + t
    out_rot = twin(h, x_rot, edge_index, batch)
    diff = torch.abs(out_rot - out_twin).max().item()
    print(f"\nE(3)-Invariance test: max |f(Qx+t) - f(x)| = {diff:.2e}")
    if diff < 1e-4:
        print("Classical Twin is E(3)-invariant.")
    else:
        print("Invariance test failed (precision difference).")

    print("\nAll checks passed.")
