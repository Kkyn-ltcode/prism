"""Q-PRISM: Quantum Periodic Representation with Invariant Sinusoidal Message-passing."""

import torch
import torch.nn as nn
import pennylane as qml

n_qubits = 4


def _create_v1_qnode():
    """Create a V1 QNode instance."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def quantum_edge_circuit_v1(inputs, weights, dist_weights):
        """V1 4-qubit parameterized quantum edge circuit."""
        if inputs.ndim == 1:
            h_i0, h_i1, h_j0, h_j1, d_ij = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        else:
            h_i0, h_i1, h_j0, h_j1, d_ij = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3], inputs[:, 4]

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

    return quantum_edge_circuit_v1


_v1_qnode = _create_v1_qnode()
quantum_edge_circuit = _v1_qnode


class QPRISMLayerV1(nn.Module):
    """V1 equivariant quantum graph convolution layer."""
    def __init__(self, in_features, n_q_layers=2):
        super().__init__()
        self.in_features = in_features
        self.n_q_layers = n_q_layers

        self.feature_proj = nn.Linear(in_features, 2)

        weight_shapes = {
            "weights": (n_q_layers, n_qubits, 3),
            "dist_weights": (n_q_layers,)
        }
        self.q_edge_layer = qml.qnn.TorchLayer(_create_v1_qnode(), weight_shapes)
        with torch.no_grad():
            self.q_edge_layer.dist_weights.uniform_(0.05, 0.2)

        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + (n_qubits - 1), in_features),
            nn.SiLU(),
            nn.Linear(in_features, in_features)
        )

    def forward(self, h, x, edge_index):
        row, col = edge_index[0], edge_index[1]

        rel_x = x[row] - x[col]                             # [E, 3]
        d_ij = torch.sum(rel_x**2, dim=-1, keepdim=True)   # [E, 1]

        h_proj = self.feature_proj(h)                       # [N, 2]
        h_i_p = h_proj[row]                                 # [E, 2]
        h_j_p = h_proj[col]                                 # [E, 2]
        q_inputs = torch.cat([h_i_p, h_j_p, d_ij], dim=-1) # [E, 5]

        q_out = self.q_edge_layer(q_inputs)                 # [E, 4]

        m_ij = q_out[:, :3]                                 # [E, 3]
        phi_x = q_out[:, 3:4]                               # [E, 1]

        coord_msg = rel_x * phi_x                           # [E, 3]
        x_update = torch.zeros_like(x)
        x_update.index_add_(0, row, coord_msg)

        degrees = torch.zeros((x.size(0), 1), device=x.device)
        degrees.index_add_(0, row, torch.ones((row.size(0), 1), device=x.device))
        x_new = x + x_update / degrees.clamp(min=1.0)

        m_i = torch.zeros((h.size(0), m_ij.size(1)), device=h.device)
        m_i.index_add_(0, row, m_ij)
        h_new = h + self.node_mlp(torch.cat([h, m_i], dim=-1))

        return h_new, x_new


EQGNNLayerV1 = QPRISMLayerV1


class QPRISMModelV1(nn.Module):
    """V1 molecular property prediction model."""
    def __init__(self, in_features=11, hidden_features=16, num_layers=3, out_dim=1, n_q_layers=2):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            QPRISMLayerV1(hidden_features, n_q_layers=n_q_layers) for _ in range(num_layers)
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


EQGNNModelV1 = QPRISMModelV1


N_RBF = 8


class GaussianRBF(nn.Module):
    """Gaussian Radial Basis Function expansion for interatomic distances."""
    def __init__(self, n_rbf=N_RBF, d_min=0.0, d_max=10.0):
        super().__init__()
        self.n_rbf = n_rbf
        centers = torch.linspace(d_min, d_max, n_rbf)
        self.register_buffer("centers", centers)
        initial_width = 1.0 / ((d_max - d_min) / n_rbf) ** 2
        self.widths = nn.Parameter(torch.full((n_rbf,), initial_width))

    def forward(self, d):
        return torch.exp(-self.widths * (d - self.centers) ** 2)


def _create_v2_qnode():
    """Create a V2 QNode instance."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def quantum_edge_circuit_v2(inputs, weights, dist_weights):
        """V2 4-qubit parameterized quantum edge circuit."""
        K = dist_weights.shape[2]
        n_layers = weights.shape[0]

        if inputs.ndim == 1:
            h_i0, h_i1 = inputs[0], inputs[1]
            h_j0, h_j1 = inputs[2], inputs[3]
            rbf = inputs[4:4 + K]
            angle = inputs[4 + K]
        else:
            h_i0, h_i1 = inputs[:, 0], inputs[:, 1]
            h_j0, h_j1 = inputs[:, 2], inputs[:, 3]
            rbf = inputs[:, 4:4 + K]
            angle = inputs[:, 4 + K]

        for l in range(n_layers):
            qml.RY(h_i0, wires=0)
            qml.RY(h_i1, wires=1)
            qml.RY(h_j0, wires=2)
            qml.RY(h_j1, wires=3)

            for q in range(n_qubits):
                if inputs.ndim == 1:
                    phase = sum(dist_weights[l, q, k] * rbf[k] for k in range(K))
                else:
                    phase = sum(dist_weights[l, q, k] * rbf[:, k] for k in range(K))
                qml.RZ(phase, wires=q)

            for q in range(n_qubits):
                qml.RY(angle, wires=q)

            for q in range(n_qubits):
                qml.RX(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)
                qml.RZ(weights[l, q, 2], wires=q)

            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])

        return [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliZ(2)),
            qml.expval(qml.PauliZ(3)),
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
            qml.expval(qml.PauliZ(1) @ qml.PauliZ(2)),
            qml.expval(qml.PauliZ(2) @ qml.PauliZ(3)),
            qml.expval(qml.PauliZ(3) @ qml.PauliZ(0)),
        ]

    return quantum_edge_circuit_v2


class QuantumHead(nn.Module):
    """Quantum processing head with feature projection and parameterized circuit."""
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.feature_proj = nn.Linear(in_features, 2)

        qnode = _create_v2_qnode()
        weight_shapes = {
            "weights": (n_q_layers, n_qubits, 3),
            "dist_weights": (n_q_layers, n_qubits, n_rbf),
        }
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        with torch.no_grad():
            self.q_layer.dist_weights.uniform_(0.05, 0.2)

    def forward(self, h_row, h_col, rbf_features, angles):
        h_i_proj = self.feature_proj(h_row)  # [E, 2]
        h_j_proj = self.feature_proj(h_col)  # [E, 2]
        q_inputs = torch.cat([h_i_proj, h_j_proj, rbf_features, angles], dim=-1)  # [E, 4+K+1]
        return self.q_layer(q_inputs)  # [E, 8]


def compute_edge_mean_angles(x, edge_index):
    """Compute mean 3-body bond angle per edge via vectorized O(E) operations."""
    row, col = edge_index[0], edge_index[1]
    num_nodes = x.size(0)
    num_edges = row.size(0)

    r_ij = x[col] - x[row]  # [E, 3]
    r_ij_len = torch.norm(r_ij, dim=-1, keepdim=True).clamp(min=1e-8)
    r_ij_hat = r_ij / r_ij_len  # [E, 3]

    sum_hat = torch.zeros(num_nodes, 3, device=x.device)
    sum_hat.index_add_(0, row, r_ij_hat)  # [N, 3]

    degrees = torch.zeros(num_nodes, 1, device=x.device)
    degrees.index_add_(0, row, torch.ones(num_edges, 1, device=x.device))

    dot_with_sum = torch.sum(r_ij_hat * sum_hat[row], dim=-1, keepdim=True)  # [E, 1]
    denom = (degrees[row] - 1.0).clamp(min=1.0)
    mean_cos = (dot_with_sum - 1.0) / denom  # [E, 1]
    mean_cos = mean_cos.clamp(-0.9999, 0.9999)
    angles = torch.acos(mean_cos)  # [E, 1]

    return angles


class QPRISMLayerV2(nn.Module):
    """V2 equivariant quantum graph convolution layer."""
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.in_features = in_features
        self.n_rbf = n_rbf

        self.rbf = GaussianRBF(n_rbf=n_rbf)
        self.head_electronic = QuantumHead(in_features, n_q_layers=n_q_layers, n_rbf=n_rbf)
        self.head_topological = QuantumHead(in_features, n_q_layers=n_q_layers, n_rbf=n_rbf)

        self.coord_gate = nn.Linear(2, 1)

        msg_dim = 7 * 2
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + msg_dim, in_features),
            nn.SiLU(),
            nn.Linear(in_features, in_features)
        )

    def forward(self, h, x, edge_index):
        row, col = edge_index[0], edge_index[1]

        rel_x = x[row] - x[col]                              # [E, 3]
        d_ij = torch.sum(rel_x ** 2, dim=-1, keepdim=True)  # [E, 1]

        rbf_features = self.rbf(d_ij)  # [E, K]
        angles = compute_edge_mean_angles(x, edge_index)  # [E, 1]

        out_e = self.head_electronic(h[row], h[col], rbf_features, angles)   # [E, 8]
        out_t = self.head_topological(h[row], h[col], rbf_features, angles)  # [E, 8]

        msg_e = torch.cat([out_e[:, :3], out_e[:, 4:]], dim=-1)  # [E, 7]
        msg_t = torch.cat([out_t[:, :3], out_t[:, 4:]], dim=-1)  # [E, 7]
        m_ij = torch.cat([msg_e, msg_t], dim=-1)                 # [E, 14]

        phi_x_raw = torch.cat([out_e[:, 3:4], out_t[:, 3:4]], dim=-1)  # [E, 2]

        phi_x = torch.tanh(self.coord_gate(phi_x_raw))  # [E, 1]
        coord_msg = rel_x * phi_x                        # [E, 3]
        x_update = torch.zeros_like(x)
        x_update.index_add_(0, row, coord_msg)

        degrees = torch.zeros((x.size(0), 1), device=x.device)
        degrees.index_add_(0, row, torch.ones((row.size(0), 1), device=x.device))
        x_new = x + x_update / degrees.clamp(min=1.0)

        m_i = torch.zeros((h.size(0), m_ij.size(1)), device=h.device)
        m_i.index_add_(0, row, m_ij)
        h_new = h + self.node_mlp(torch.cat([h, m_i], dim=-1))

        return h_new, x_new


class QPRISMModelV2(nn.Module):
    """V2 molecular property prediction model."""
    def __init__(self, in_features=11, hidden_features=16, num_layers=3, out_dim=1,
                 n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList([
            QPRISMLayerV2(hidden_features, n_q_layers=n_q_layers, n_rbf=n_rbf)
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


QPRISMLayer = QPRISMLayerV2
QPRISMModel = QPRISMModelV2
QPRISM = QPRISMModel

EQGNNLayer = QPRISMLayer
EQGNNModel = QPRISMModel
EQGNNLayerV2 = QPRISMLayerV2
EQGNNModelV2 = QPRISMModelV2
EQGNN = QPRISMModel


def _generate_test_geometry():
    """Shared test geometry: 4-node graph with random features and coordinates."""
    h = torch.randn(4, 8)
    x = torch.randn(4, 3)
    edge_index = torch.tensor([
        [0, 1, 0, 2, 0, 3, 1, 2, 2, 3],
        [1, 0, 2, 0, 3, 0, 2, 1, 3, 2]
    ])
    return h, x, edge_index


def _generate_random_se3():
    """Generate a random SO(3) rotation Q and R^3 translation g."""
    random_mat = torch.randn(3, 3)
    q, _ = torch.linalg.qr(random_mat)
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    g = torch.tensor([[1.5, -2.3, 4.1]])
    return q, g


def test_rotational_invariance():
    """Verify rotational equivariance and invariance for V2 architecture."""
    print("Running E(3) Equivariance & Invariance Verification (V2)")

    h, x, edge_index = _generate_test_geometry()
    q, g = _generate_random_se3()
    x_transformed = x @ q.T + g

    layer = EQGNNLayerV2(in_features=8, n_q_layers=2)
    layer.eval()

    h_orig, x_orig = layer(h, x, edge_index)
    h_trans, x_trans = layer(h, x_transformed, edge_index)

    x_expected = x_orig @ q.T + g
    coord_equiv_error = torch.max(torch.abs(x_trans - x_expected)).item()
    feat_inv_error = torch.max(torch.abs(h_trans - h_orig)).item()

    print(f"1. Layer Coordinate Equivariance Error: {coord_equiv_error:.2e}")
    print(f"2. Layer Feature Invariance Error:      {feat_inv_error:.2e}")
    assert coord_equiv_error < 1e-5, f"Layer Coordinate Equivariance failed with error {coord_equiv_error}"
    assert feat_inv_error < 1e-5, f"Layer Feature Invariance failed with error {feat_inv_error}"

    model = EQGNNModelV2(in_features=5, hidden_features=8, num_layers=2)
    model.eval()

    h_m = torch.randn(4, 5)
    batch = torch.tensor([0, 0, 0, 0])
    pred_orig = model(h_m, x, edge_index, batch)
    pred_trans = model(h_m, x_transformed, edge_index, batch)
    pred_inv_error = torch.abs(pred_orig - pred_trans).item()

    print(f"3. End-to-End Global Invariance Error:   {pred_inv_error:.2e}")
    assert pred_inv_error < 1e-5, f"Global Invariance failed with error {pred_inv_error}"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"4. V2 Model Parameter Count (h=8):       {total_params:,}")

    model_16 = EQGNNModelV2(in_features=11, hidden_features=16, num_layers=2)
    params_16 = sum(p.numel() for p in model_16.parameters() if p.requires_grad)
    print(f"5. V2 Model Parameter Count (h=16):      {params_16:,}")

    print("All E(3) Equivariance & Invariance Tests PASSED (V2).")


def test_rotational_invariance_v1():
    """Verify rotational equivariance and invariance for V1 architecture."""
    print("Running V1 E(3) Equivariance & Invariance Verification")

    h, x, edge_index = _generate_test_geometry()
    q, g = _generate_random_se3()
    x_transformed = x @ q.T + g

    layer = EQGNNLayerV1(in_features=8, n_q_layers=2)
    layer.eval()

    h_orig, x_orig = layer(h, x, edge_index)
    h_trans, x_trans = layer(h, x_transformed, edge_index)

    x_expected = x_orig @ q.T + g
    coord_equiv_error = torch.max(torch.abs(x_trans - x_expected)).item()
    feat_inv_error = torch.max(torch.abs(h_trans - h_orig)).item()

    print(f"1. V1 Layer Coord Equivariance Error: {coord_equiv_error:.2e}")
    print(f"2. V1 Layer Feature Invariance Error: {feat_inv_error:.2e}")
    assert coord_equiv_error < 1e-5, f"V1 Coord Equivariance failed: {coord_equiv_error}"
    assert feat_inv_error < 1e-5, f"V1 Feature Invariance failed: {feat_inv_error}"

    model = EQGNNModelV1(in_features=5, hidden_features=8, num_layers=2)
    model.eval()

    h_m = torch.randn(4, 5)
    batch = torch.tensor([0, 0, 0, 0])
    pred_orig = model(h_m, x, edge_index, batch)
    pred_trans = model(h_m, x_transformed, edge_index, batch)
    pred_inv_error = torch.abs(pred_orig - pred_trans).item()

    print(f"3. V1 Global Invariance Error:        {pred_inv_error:.2e}")
    assert pred_inv_error < 1e-5, f"V1 Global Invariance failed: {pred_inv_error}"

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"4. V1 Model Parameter Count (h=8):    {total_params:,}")

    model_16 = EQGNNModelV1(in_features=11, hidden_features=16, num_layers=2)
    params_16 = sum(p.numel() for p in model_16.parameters() if p.requires_grad)
    print(f"5. V1 Model Parameter Count (h=16):   {params_16:,}")

    print("V1 E(3) Tests PASSED.")


if __name__ == "__main__":
    test_rotational_invariance()
    print()
    test_rotational_invariance_v1()
