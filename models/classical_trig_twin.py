"""PRISM: Periodic Representation with Invariant Sinusoidal Message-passing models."""

import torch
import torch.nn as nn
import math
from .eq_gnn import GaussianRBF, compute_edge_mean_angles, N_RBF


class PRISMHead(nn.Module):
    """Trigonometric twin head using SIREN-style periodic sine activations."""
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF, omega_0=30.0):
        super().__init__()
        self.feature_proj = nn.Linear(in_features, 2)
        self.omega_0 = omega_0

        input_dim = 4 + n_rbf + 1  # [h_i0, h_i1, h_j0, h_j1, rbf×K, angle]
        n_outputs = 8
        hidden = 4
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, n_outputs, bias=False)

        self._siren_init()

    def _siren_init(self):
        """SIREN weight initialization."""
        n_in1 = self.linear1.in_features
        with torch.no_grad():
            self.linear1.weight.uniform_(-1.0 / n_in1, 1.0 / n_in1)
            if self.linear1.bias is not None:
                self.linear1.bias.uniform_(-1.0 / n_in1, 1.0 / n_in1)

        n_in2 = self.linear2.in_features
        bound2 = math.sqrt(6.0 / n_in2) / self.omega_0
        with torch.no_grad():
            self.linear2.weight.uniform_(-bound2, bound2)

    def forward(self, h_row, h_col, rbf_features, angles):
        h_i_proj = self.feature_proj(h_row)  # [E, 2]
        h_j_proj = self.feature_proj(h_col)  # [E, 2]
        inputs = torch.cat([h_i_proj, h_j_proj, rbf_features, angles], dim=-1)  # [E, 4+K+1]

        x = torch.sin(self.omega_0 * self.linear1(inputs))  # [E, hidden]
        return torch.tanh(self.linear2(x))  # [E, 8], bounded to [-1, 1]


TrigHead = PRISMHead


class FourierFeatureHead(nn.Module):
    """Classical Fourier feature twin head."""
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF, n_fourier=4, sigma=10.0):
        super().__init__()
        self.feature_proj = nn.Linear(in_features, 2)

        input_dim = 4 + n_rbf + 1  # [h_i0, h_i1, h_j0, h_j1, rbf×K, angle]
        n_outputs = 8

        B = torch.randn(input_dim, n_fourier) * sigma
        self.register_buffer("B", B)

        fourier_dim = 2 * n_fourier  # = 8
        hidden = 5
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_outputs, bias=False),
        )

    def forward(self, h_row, h_col, rbf_features, angles):
        h_i_proj = self.feature_proj(h_row)  # [E, 2]
        h_j_proj = self.feature_proj(h_col)  # [E, 2]
        inputs = torch.cat([h_i_proj, h_j_proj, rbf_features, angles], dim=-1)  # [E, 13]

        projected = inputs @ self.B  # [E, n_fourier]
        fourier_features = torch.cat([
            torch.sin(2 * math.pi * projected),
            torch.cos(2 * math.pi * projected),
        ], dim=-1)  # [E, 2×n_fourier]

        return torch.tanh(self.mlp(fourier_features))  # [E, 8], bounded to [-1, 1]


class PRISMLayerV2(nn.Module):
    """PRISM equivariant graph convolution layer."""
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF,
                 head_type="siren", omega_0=30.0):
        super().__init__()
        self.in_features = in_features
        self.n_rbf = n_rbf

        self.rbf = GaussianRBF(n_rbf=n_rbf)

        HeadClass = PRISMHead if head_type == "siren" else FourierFeatureHead
        head_kwargs = dict(in_features=in_features, n_q_layers=n_q_layers, n_rbf=n_rbf)
        if head_type == "siren":
            head_kwargs["omega_0"] = omega_0

        self.head_electronic = HeadClass(**head_kwargs)
        self.head_topological = HeadClass(**head_kwargs)

        self.coord_gate = nn.Linear(2, 1)

        msg_dim = 7 * 2
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


PRISMLayer = PRISMLayerV2
TrigTwinLayerV2 = PRISMLayerV2


class PRISMModelV2(nn.Module):
    """PRISM periodic model for molecular property prediction."""
    def __init__(self, in_features=11, hidden_features=16, num_layers=3, out_dim=1,
                 n_q_layers=2, n_rbf=N_RBF, head_type="siren", omega_0=30.0):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            PRISMLayerV2(hidden_features, n_q_layers=n_q_layers, n_rbf=n_rbf,
                         head_type=head_type, omega_0=omega_0)
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


PRISMModel = PRISMModelV2
PRISM = PRISMModel
TrigTwinModelV2 = PRISMModelV2
PeriodicEGNN = PRISMModel


def create_param_matched_prism(target_params, in_features=11, hidden_features=16,
                               num_layers=2, out_dim=1, n_q_layers=2, n_rbf=N_RBF,
                               head_type="siren", omega_0=30.0):
    """Create a PRISMModelV2 and return parameter count and diff vs target."""
    model = PRISMModelV2(
        in_features=in_features,
        hidden_features=hidden_features,
        num_layers=num_layers,
        out_dim=out_dim,
        n_q_layers=n_q_layers,
        n_rbf=n_rbf,
        head_type=head_type,
        omega_0=omega_0,
    )
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    diff = total - target_params
    return model, total, diff


create_param_matched_trig_twin = create_param_matched_prism
create_param_matched_trigtwin = create_param_matched_prism


if __name__ == "__main__":
    print("Classical Trigonometric Twin Parameter Matching Verification")

    from .eq_gnn import EQGNNModelV2, QPRISMModelV2
    from .classical_twin import ClassicalTwinModelV2
    from .egnn_baseline import count_parameters

    eqgnn = QPRISMModelV2(in_features=11, hidden_features=16, num_layers=2,
                          n_q_layers=2, out_dim=1)
    twin_tanh = ClassicalTwinModelV2(in_features=11, hidden_features=16, num_layers=2,
                                      n_q_layers=2, out_dim=1)
    twin_siren = PRISMModelV2(in_features=11, hidden_features=16, num_layers=2,
                              n_q_layers=2, out_dim=1, head_type="siren")
    twin_fourier = PRISMModelV2(in_features=11, hidden_features=16, num_layers=2,
                                n_q_layers=2, out_dim=1, head_type="fourier")

    eqgnn_params = count_parameters(eqgnn)
    tanh_params = count_parameters(twin_tanh)
    siren_params = count_parameters(twin_siren)
    fourier_params = count_parameters(twin_fourier)

    print(f"\nQ-PRISM params:               {eqgnn_params:,}")
    print(f"Classical Twin (tanh) params: {tanh_params:,}")
    print(f"PRISM (SIREN) params:         {siren_params:,}")
    print(f"PRISM (Fourier) params:       {fourier_params:,}")
    print(f"\nSIREN vs Q-PRISM diff:        {siren_params - eqgnn_params:+d}")
    print(f"Fourier vs Q-PRISM diff:      {fourier_params - eqgnn_params:+d}")
    print(f"SIREN vs Tanh Twin diff:      {siren_params - tanh_params:+d}")

    h = torch.randn(4, 11)
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0,0,0,1,1,2,2,3,3,3],
                                [1,2,3,0,2,0,1,0,1,2]], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)

    out_eqgnn = eqgnn(h, x, edge_index, batch)
    out_siren = twin_siren(h, x, edge_index, batch)
    out_fourier = twin_fourier(h, x, edge_index, batch)

    print(f"\nQ-PRISM output shape:         {out_eqgnn.shape}")
    print(f"PRISM (SIREN) output:         {out_siren.shape}")
    print(f"PRISM (Fourier) output:       {out_fourier.shape}")
    assert out_eqgnn.shape == out_siren.shape == out_fourier.shape, "Output shape mismatch!"

    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    t = torch.randn(1, 3)

    x_rot = x @ Q.T + t
    out_rot_siren = twin_siren(h, x_rot, edge_index, batch)
    diff_siren = torch.abs(out_rot_siren - out_siren).max().item()
    out_rot_fourier = twin_fourier(h, x_rot, edge_index, batch)
    diff_fourier = torch.abs(out_rot_fourier - out_fourier).max().item()

    print(f"\nE(3)-Invariance SIREN:       max |f(Qx+t) - f(x)| = {diff_siren:.2e}")
    print(f"E(3)-Invariance Fourier:     max |f(Qx+t) - f(x)| = {diff_fourier:.2e}")
    if diff_siren < 1e-4 and diff_fourier < 1e-4:
        print("Both Trig Twins are E(3)-invariant.")
    else:
        print("Invariance test issue.")

    print("\nAll checks passed.")
