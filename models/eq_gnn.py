"""
Q-PRISM: Quantum Periodic Representation with Invariant Sinusoidal Message-passing
==================================================================================
Reference Implementation for 3D Molecular Property Prediction.

v1.0: Original 4-Qubit PQC with single-axis distance encoding.
v2.0 (Current Default): Upgraded architecture with:
  - Upgrade 2.1: RBF Multi-Frequency Harmonic Distance Re-Uploading
  - Upgrade 2.2: Multi-Head (2-Head) Quantum Edge Convolution
  - Upgrade 2.3: Entanglement Correlator Readout (⟨Z_i Z_j⟩)
  - Upgrade 2.4: 3-Body SO(3)-Invariant Angular Re-Uploading
  - Upgrade 2.5: Tanh-Gated Equivariant Coordinate Updates
"""

import torch
import torch.nn as nn
import pennylane as qml

n_qubits = 4


# =============================================================================
# V1 ARCHITECTURE (Preserved for backward compatibility & ablation baselines)
# =============================================================================

def _create_v1_qnode():
    """Factory creating a V1 QNode with its own device instance."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def quantum_edge_circuit_v1(inputs, weights, dist_weights):
        """
        V1 4-Qubit PQC with Multi-Layer Multi-Axis Data Re-Uploading for Invariant Edge
        Message Passing and Equivariant Scalar Coordinate Weight generation.

        Args:
            inputs: Tensor of shape [E, 5] containing [h_i[0], h_i[1], h_j[0], h_j[1], d_ij]
                    where d_ij = ||x_i - x_j||^2 (E(3)-invariant).
            weights: Learnable variational parameters of shape [n_layers, n_qubits, 3] (RX, RY, RZ).
            dist_weights: Learnable distance scale parameters of shape [n_layers].

        Returns:
            Expectation values of Pauli-Z across all 4 qubits in [-1, 1].
        """
        if inputs.ndim == 1:
            h_i0, h_i1, h_j0, h_j1, d_ij = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        else:
            h_i0, h_i1, h_j0, h_j1, d_ij = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3], inputs[:, 4]

        n_layers = weights.shape[0]

        # Interleaved Data Re-Uploading across all variational layers
        for l in range(n_layers):
            # Step 1: Multi-Axis Feature Encoding (Non-destructive)
            qml.RY(h_i0, wires=0)
            qml.RY(h_i1, wires=1)
            qml.RY(h_j0, wires=2)
            qml.RY(h_j1, wires=3)

            # Step 2: Distance Data Re-Uploading with layer-specific scaling
            scaled_d = d_ij * dist_weights[l]
            for q in range(n_qubits):
                qml.RZ(scaled_d, wires=q)

            # Step 3: Parameterized Single-Qubit Rotations (RX, RY, RZ)
            for q in range(n_qubits):
                qml.RX(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)
                qml.RZ(weights[l, q, 2], wires=q)

            # Step 4: Circular Ring CNOT Entanglement (Linear Depth O(n))
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])

        # Step 5: Multi-Observable Readout (Pauli-Z Expectations)
        return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

    return quantum_edge_circuit_v1


# Module-level V1 QNode for backward compatibility
_v1_qnode = _create_v1_qnode()
quantum_edge_circuit = _v1_qnode  # Legacy alias


class QPRISMLayerV1(nn.Module):
    """
    V1 Equivariant Quantum Graph Convolution Layer (Q-PRISM V1).
    Guarantees E(3) Equivariance for coordinates and E(3) Invariance for features.
    """
    def __init__(self, in_features, n_q_layers=2):
        super().__init__()
        self.in_features = in_features
        self.n_q_layers = n_q_layers

        # Project high-dimensional atomic features to 2D per node for quantum encoding
        self.feature_proj = nn.Linear(in_features, 2)

        # PennyLane PyTorch Layer with (RX, RY, RZ) per qubit and distance weights
        weight_shapes = {
            "weights": (n_q_layers, n_qubits, 3),
            "dist_weights": (n_q_layers,)
        }
        self.q_edge_layer = qml.qnn.TorchLayer(_create_v1_qnode(), weight_shapes)
        # Initialize distance scaling to keep rotations in calibrated regime [0, 2pi]
        with torch.no_grad():
            self.q_edge_layer.dist_weights.uniform_(0.05, 0.2)

        # Invariant Node Feature Update MLP: phi_h(h_i, m_i) -> h_i_new
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + (n_qubits - 1), in_features),
            nn.SiLU(),
            nn.Linear(in_features, in_features)
        )

    def forward(self, h, x, edge_index):
        """
        Args:
            h: [N, in_features] Node feature matrix (invariant).
            x: [N, 3] 3D coordinates (equivariant).
            edge_index: [2, E] Graph edge index (row, col).

        Returns:
            h_new: [N, in_features] Updated invariant node embeddings.
            x_new: [N, 3] Updated equivariant 3D coordinates.
        """
        row, col = edge_index[0], edge_index[1]

        # 1. Compute Invariant Squared Euclidean Distance
        rel_x = x[row] - x[col]                             # [E, 3]
        d_ij = torch.sum(rel_x**2, dim=-1, keepdim=True)   # [E, 1]

        # 2. Project Features and Prepare Quantum Circuit Input
        h_proj = self.feature_proj(h)                       # [N, 2]
        h_i_p = h_proj[row]                                 # [E, 2]
        h_j_p = h_proj[col]                                 # [E, 2]
        q_inputs = torch.cat([h_i_p, h_j_p, d_ij], dim=-1) # [E, 5]

        # 3. Evaluate Quantum Edge Circuit
        q_out = self.q_edge_layer(q_inputs)                 # [E, 4]

        # Partition outputs: First 3 qubits -> Message vector m_ij, 4th qubit -> Scalar weight phi_x
        m_ij = q_out[:, :3]                                 # [E, 3]
        phi_x = q_out[:, 3:4]                               # [E, 1]

        # 4. Equivariant Coordinate Radial Update (Faithful to Blueprint Eq. 67)
        # x_i^{l+1} = x_i^l + (1 / |N(i)|) * sum_{j in N(i)} (x_i^l - x_j^l) * phi_x^Q(m_ij)
        coord_msg = rel_x * phi_x                           # [E, 3]
        x_update = torch.zeros_like(x)
        x_update.index_add_(0, row, coord_msg)

        # Exact per-node degree normalization: |N(i)|
        degrees = torch.zeros((x.size(0), 1), device=x.device)
        degrees.index_add_(0, row, torch.ones((row.size(0), 1), device=x.device))
        x_new = x + x_update / degrees.clamp(min=1.0)

        # 5. Invariant Node Update (Faithful to Blueprint Eq. 69)
        # m_i = sum_{j in N(i)} m_ij,  h_i^{l+1} = h_i^l + MLP([h_i^l, m_i])
        m_i = torch.zeros((h.size(0), m_ij.size(1)), device=h.device)
        m_i.index_add_(0, row, m_ij)
        h_new = h + self.node_mlp(torch.cat([h, m_i], dim=-1))

        return h_new, x_new


EQGNNLayerV1 = QPRISMLayerV1  # Backward compatibility alias
 
 
class QPRISMModelV1(nn.Module):
    """
    V1 End-to-End Molecular Property Prediction Network (Q-PRISM V1).
    """
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
        """
        Args:
            h: [N, in_features] One-hot or continuous atomic features.
            x: [N, 3] 3D coordinates of atoms.
            edge_index: [2, E] Adjacency list.
            batch: [N] Graph batch assignment vector.

        Returns:
            pred: [num_graphs, out_dim] Molecular property prediction.
        """
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        h = self.embedding(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)

        # Global Invariant Mean Pooling across each molecule
        num_graphs = int(batch.max().item()) + 1
        out_pool = torch.zeros((num_graphs, h.size(1)), device=h.device)
        out_pool.index_add_(0, batch, h)

        # Count atoms per molecule for proper mean aggregation
        counts = torch.zeros((num_graphs, 1), device=h.device)
        counts.index_add_(0, batch, torch.ones((h.size(0), 1), device=h.device))
        out_pool = out_pool / counts.clamp(min=1.0)

        return self.head(out_pool)


EQGNNModelV1 = QPRISMModelV1  # Backward compatibility alias


# =============================================================================
# V2 ARCHITECTURE (Upgraded — Blueprint Sections 2.1–2.5)
# =============================================================================

# --- Upgrade 2.1: Gaussian Radial Basis Function Distance Expansion ---

N_RBF = 8  # Number of learnable Gaussian RBF centers


class GaussianRBF(nn.Module):
    """
    Gaussian Radial Basis Function expansion for interatomic distances.

    φ_k(d) = exp(−γ_k · (d − μ_k)²)

    Centers μ_k are evenly spaced in [0, 10] ², widths γ_k are learnable.
    Under the Fourier analysis of data re-uploading (Pérez-Salinas et al.,
    Schuld et al.), this transforms the PQC into a universal truncated
    Fourier approximator for molecular potential surfaces.

    Equivariance: d_ij = ||x_i − x_j||² is strictly E(3)-invariant;
    any non-linear function φ(d_ij) remains invariant.
    """
    def __init__(self, n_rbf=N_RBF, d_min=0.0, d_max=10.0):
        super().__init__()
        self.n_rbf = n_rbf
        centers = torch.linspace(d_min, d_max, n_rbf)
        self.register_buffer("centers", centers)
        # Learnable widths initialized to spread across the range
        initial_width = 1.0 / ((d_max - d_min) / n_rbf) ** 2
        self.widths = nn.Parameter(torch.full((n_rbf,), initial_width))

    def forward(self, d):
        """
        Args:
            d: [E, 1] Squared Euclidean distances.

        Returns:
            rbf: [E, n_rbf] Gaussian RBF features in (0, 1].
        """
        # d: [E, 1], centers: [K] -> broadcast to [E, K]
        return torch.exp(-self.widths * (d - self.centers) ** 2)


# --- V2 Quantum Circuit with RBF + Angle + Correlator Readout ---

def _create_v2_qnode():
    """Factory creating a V2 QNode with its own device instance."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def quantum_edge_circuit_v2(inputs, weights, dist_weights):
        """
        V2 4-Qubit PQC with:
        - Multi-frequency RBF distance re-uploading (Upgrade 2.1)
        - 3-body angular re-uploading (Upgrade 2.4)
        - Entanglement correlator readout (Upgrade 2.3)

        Args:
            inputs: [E, 4+K+1] = [h_i0, h_i1, h_j0, h_j1, rbf_0..rbf_{K-1}, angle]
            weights: [n_layers, n_qubits, 3] Variational rotation parameters.
            dist_weights: [n_layers, n_qubits, K] Per-qubit RBF distance scaling.

        Returns:
            8 expectation values:
              [⟨Z_0⟩, ⟨Z_1⟩, ⟨Z_2⟩, ⟨Z_3⟩,
               ⟨Z_0 Z_1⟩, ⟨Z_1 Z_2⟩, ⟨Z_2 Z_3⟩, ⟨Z_3 Z_0⟩]
        """
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
            # Step 1: Multi-Axis Feature Encoding (Non-destructive)
            qml.RY(h_i0, wires=0)
            qml.RY(h_i1, wires=1)
            qml.RY(h_j0, wires=2)
            qml.RY(h_j1, wires=3)

            # Step 2: RBF Multi-Frequency Distance Re-Uploading (Upgrade 2.1)
            # Per-qubit phase = Σ_k w_{d,q,k}^{(l)} · φ_k(d_ij)
            for q in range(n_qubits):
                if inputs.ndim == 1:
                    phase = sum(dist_weights[l, q, k] * rbf[k] for k in range(K))
                else:
                    phase = sum(dist_weights[l, q, k] * rbf[:, k] for k in range(K))
                qml.RZ(phase, wires=q)

            # Step 3: SO(3)-Invariant Angular Re-Uploading (Upgrade 2.4)
            # θ_ijk encodes 3-body bond-angle geometry via R_y rotations
            for q in range(n_qubits):
                qml.RY(angle, wires=q)

            # Step 4: Parameterized Single-Qubit Rotations (RX, RY, RZ)
            for q in range(n_qubits):
                qml.RX(weights[l, q, 0], wires=q)
                qml.RY(weights[l, q, 1], wires=q)
                qml.RZ(weights[l, q, 2], wires=q)

            # Step 5: Circular Ring CNOT Entanglement (Linear Depth O(n))
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])

        # Step 6: Multi-Observable Readout (Upgrade 2.3)
        return [
            # Single-qubit Pauli-Z expectations
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1)),
            qml.expval(qml.PauliZ(2)),
            qml.expval(qml.PauliZ(3)),
            # Two-qubit entanglement correlators ⟨Z_i Z_j⟩
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
            qml.expval(qml.PauliZ(1) @ qml.PauliZ(2)),
            qml.expval(qml.PauliZ(2) @ qml.PauliZ(3)),
            qml.expval(qml.PauliZ(3) @ qml.PauliZ(0)),
        ]

    return quantum_edge_circuit_v2


# --- Upgrade 2.2: Quantum Head (Feature Projection + PQC per Subspace) ---

class QuantumHead(nn.Module):
    """
    A single quantum processing head with its own feature projection and PQC.

    Projects node embeddings to 2D via a learned linear map, concatenates with
    shared RBF distance features and 3-body angle, then evaluates the V2 PQC.

    Output: 8 observables = [⟨Z_0⟩, ⟨Z_1⟩, ⟨Z_2⟩, ⟨Z_3⟩,
                              ⟨Z_0 Z_1⟩, ⟨Z_1 Z_2⟩, ⟨Z_2 Z_3⟩, ⟨Z_3 Z_0⟩]
    """
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.feature_proj = nn.Linear(in_features, 2)

        qnode = _create_v2_qnode()
        weight_shapes = {
            "weights": (n_q_layers, n_qubits, 3),
            "dist_weights": (n_q_layers, n_qubits, n_rbf),
        }
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        # Initialize RBF distance weights to calibrated regime
        with torch.no_grad():
            self.q_layer.dist_weights.uniform_(0.05, 0.2)

    def forward(self, h_row, h_col, rbf_features, angles):
        """
        Args:
            h_row: [E, in_features] Source node embeddings.
            h_col: [E, in_features] Target node embeddings.
            rbf_features: [E, K] Gaussian RBF distance features.
            angles: [E, 1] Mean 3-body bond angles per edge.

        Returns:
            q_out: [E, 8] Quantum observable expectations.
        """
        h_i_proj = self.feature_proj(h_row)  # [E, 2]
        h_j_proj = self.feature_proj(h_col)  # [E, 2]
        q_inputs = torch.cat([h_i_proj, h_j_proj, rbf_features, angles], dim=-1)  # [E, 4+K+1]
        return self.q_layer(q_inputs)  # [E, 8]


# --- Upgrade 2.4: 3-Body SO(3)-Invariant Angle Computation ---

def compute_edge_mean_angles(x, edge_index):
    """
    Compute mean 3-body bond angle per edge via vectorized O(E) operations.

    For each edge (i,j), computes the mean cosine angle θ_ijk across all
    third atoms k ∈ N(i), k ≠ j using the identity:

        mean_cos = (r̂_ij · Σ_{k∈N(i)} r̂_ik − 1) / (deg(i) − 1)

    where r̂_ij = (x_j − x_i) / ||x_j − x_i||.

    Equivariance proof: cos θ_ijk = r_ij · r_ik / (||r_ij|| · ||r_ik||)
    depends only on inner products of relative vectors, which are invariant
    under O(3) rotations Q: (Qr_ij)·(Qr_ik) = r_ij · Q^T Q · r_ik = r_ij · r_ik.

    Args:
        x: [N, 3] Node coordinates.
        edge_index: [2, E] Edge index (row=source, col=target).

    Returns:
        angles: [E, 1] Mean bond angle per edge in radians ∈ [0, π].
    """
    row, col = edge_index[0], edge_index[1]
    num_nodes = x.size(0)
    num_edges = row.size(0)

    # Unit direction vectors from source i to target j
    r_ij = x[col] - x[row]  # [E, 3]
    r_ij_len = torch.norm(r_ij, dim=-1, keepdim=True).clamp(min=1e-8)
    r_ij_hat = r_ij / r_ij_len  # [E, 3]

    # Aggregate: sum of outgoing unit vectors per source node
    sum_hat = torch.zeros(num_nodes, 3, device=x.device)
    sum_hat.index_add_(0, row, r_ij_hat)  # [N, 3]

    # Node degree (number of outgoing edges per source)
    degrees = torch.zeros(num_nodes, 1, device=x.device)
    degrees.index_add_(0, row, torch.ones(num_edges, 1, device=x.device))

    # For edge (i,j): mean_cos over k ≠ j = (r̂_ij · sum_hat[i] - 1) / (deg(i) - 1)
    # The "-1" subtracts the self-contribution (r̂_ij · r̂_ij = 1)
    dot_with_sum = torch.sum(r_ij_hat * sum_hat[row], dim=-1, keepdim=True)  # [E, 1]
    denom = (degrees[row] - 1.0).clamp(min=1.0)
    mean_cos = (dot_with_sum - 1.0) / denom  # [E, 1]
    mean_cos = mean_cos.clamp(-0.9999, 0.9999)
    angles = torch.acos(mean_cos)  # [E, 1], in [0, π]

    return angles


# --- V2 Equivariant Quantum Convolution Layer ---

class QPRISMLayerV2(nn.Module):
    """
    V2 Equivariant Quantum Graph Convolution Layer (Q-PRISM V2).

    Implements all Blueprint upgrades:
    - Upgrade 2.1: RBF multi-frequency distance re-uploading
    - Upgrade 2.2: 2-head quantum edge convolution (Electronic + Topological)
    - Upgrade 2.3: Entanglement correlator readout (⟨Z_i Z_j⟩ two-body observables)
    - Upgrade 2.4: 3-body SO(3)-invariant angular re-uploading
    - Upgrade 2.5: Tanh-gated equivariant coordinate updates

    Per-head output decomposition:
        - Message: [⟨Z_0⟩, ⟨Z_1⟩, ⟨Z_2⟩, ⟨Z_0 Z_1⟩, ⟨Z_1 Z_2⟩, ⟨Z_2 Z_3⟩, ⟨Z_3 Z_0⟩] = 7 values
        - Coordinate scalar: ⟨Z_3⟩ = 1 value
    Total message with 2 heads: 14 dimensions
    """
    def __init__(self, in_features, n_q_layers=2, n_rbf=N_RBF):
        super().__init__()
        self.in_features = in_features
        self.n_rbf = n_rbf

        # Shared Gaussian RBF distance expansion (Upgrade 2.1)
        self.rbf = GaussianRBF(n_rbf=n_rbf)

        # Two quantum heads (Upgrade 2.2)
        # Head 1 (Electronic): learns to project electronic features
        # Head 2 (Topological): learns to project topological features
        # Both project from the full hidden embedding — the network learns routing.
        self.head_electronic = QuantumHead(in_features, n_q_layers=n_q_layers, n_rbf=n_rbf)
        self.head_topological = QuantumHead(in_features, n_q_layers=n_q_layers, n_rbf=n_rbf)

        # Tanh-gated coordinate scalar MLP (Upgrade 2.5)
        # Combines the two ⟨Z_3⟩ coordinate scalars from each head
        self.coord_gate = nn.Linear(2, 1)

        # Invariant Node Feature Update MLP
        # Input: in_features + 14 (7 message values per head × 2 heads)
        msg_dim = 7 * 2
        self.node_mlp = nn.Sequential(
            nn.Linear(in_features + msg_dim, in_features),
            nn.SiLU(),
            nn.Linear(in_features, in_features)
        )

    def forward(self, h, x, edge_index):
        """
        Args:
            h: [N, in_features] Node feature matrix (invariant).
            x: [N, 3] 3D coordinates (equivariant).
            edge_index: [2, E] Graph edge index (row, col).

        Returns:
            h_new: [N, in_features] Updated invariant node embeddings.
            x_new: [N, 3] Updated equivariant 3D coordinates.
        """
        row, col = edge_index[0], edge_index[1]

        # 1. Compute Invariant Squared Euclidean Distance
        rel_x = x[row] - x[col]                              # [E, 3]
        d_ij = torch.sum(rel_x ** 2, dim=-1, keepdim=True)  # [E, 1]

        # 2. RBF Distance Expansion (Upgrade 2.1)
        rbf_features = self.rbf(d_ij)  # [E, K]

        # 3. 3-Body Angular Encoding (Upgrade 2.4)
        angles = compute_edge_mean_angles(x, edge_index)  # [E, 1]

        # 4. Multi-Head Quantum Edge Convolution (Upgrade 2.2)
        out_e = self.head_electronic(h[row], h[col], rbf_features, angles)   # [E, 8]
        out_t = self.head_topological(h[row], h[col], rbf_features, angles)  # [E, 8]

        # 5. Parse per-head outputs into message and coordinate scalar
        # Per head output: [⟨Z_0⟩, ⟨Z_1⟩, ⟨Z_2⟩, ⟨Z_3⟩, ⟨Z_0Z_1⟩, ⟨Z_1Z_2⟩, ⟨Z_2Z_3⟩, ⟨Z_3Z_0⟩]
        # Message = indices [0,1,2] (single-qubit) + [4,5,6,7] (correlators) = 7 per head
        # Coordinate scalar = index [3] (⟨Z_3⟩) = 1 per head
        msg_e = torch.cat([out_e[:, :3], out_e[:, 4:]], dim=-1)  # [E, 7]
        msg_t = torch.cat([out_t[:, :3], out_t[:, 4:]], dim=-1)  # [E, 7]
        m_ij = torch.cat([msg_e, msg_t], dim=-1)                 # [E, 14]

        phi_x_raw = torch.cat([out_e[:, 3:4], out_t[:, 3:4]], dim=-1)  # [E, 2]

        # 6. Tanh-Gated Equivariant Coordinate Update (Upgrade 2.5)
        # tanh prevents coordinate explosion during backprop across deep layers (L ≥ 4)
        phi_x = torch.tanh(self.coord_gate(phi_x_raw))  # [E, 1]
        coord_msg = rel_x * phi_x                        # [E, 3]
        x_update = torch.zeros_like(x)
        x_update.index_add_(0, row, coord_msg)

        # Exact per-node degree normalization: |N(i)|
        degrees = torch.zeros((x.size(0), 1), device=x.device)
        degrees.index_add_(0, row, torch.ones((row.size(0), 1), device=x.device))
        x_new = x + x_update / degrees.clamp(min=1.0)

        # 7. Invariant Node Feature Update
        m_i = torch.zeros((h.size(0), m_ij.size(1)), device=h.device)
        m_i.index_add_(0, row, m_ij)
        h_new = h + self.node_mlp(torch.cat([h, m_i], dim=-1))

        return h_new, x_new


# --- V2 Full Molecular Property Prediction Network ---

class QPRISMModelV2(nn.Module):
    """
    V2 End-to-End Molecular Property Prediction Network (Q-PRISM V2).

    Uses QPRISMLayerV2 with all blueprint upgrades:
    - RBF harmonic distance encoding
    - Multi-head (2-head) quantum edge convolution
    - Entanglement correlator readout
    - 3-body angular encoding
    - Tanh-gated equivariant coordinate updates
    """
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
        """
        Args:
            h: [N, in_features] One-hot or continuous atomic features.
            x: [N, 3] 3D coordinates of atoms.
            edge_index: [2, E] Adjacency list.
            batch: [N] Graph batch assignment vector.

        Returns:
            pred: [num_graphs, out_dim] Molecular property prediction.
        """
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        h = self.embedding(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)

        # Global Invariant Mean Pooling across each molecule
        num_graphs = int(batch.max().item()) + 1
        out_pool = torch.zeros((num_graphs, h.size(1)), device=h.device)
        out_pool.index_add_(0, batch, h)

        # Count atoms per molecule for proper mean aggregation
        counts = torch.zeros((num_graphs, 1), device=h.device)
        counts.index_add_(0, batch, torch.ones((h.size(0), 1), device=h.device))
        out_pool = out_pool / counts.clamp(min=1.0)

        return self.head(out_pool)


# =============================================================================
# DEFAULT EXPORTS (V2 is the current default)
# =============================================================================
QPRISMLayer = QPRISMLayerV2
QPRISMModel = QPRISMModelV2
QPRISM = QPRISMModel

# Backward compatibility aliases
EQGNNLayer = QPRISMLayer
EQGNNModel = QPRISMModel
EQGNNLayerV2 = QPRISMLayerV2
EQGNNModelV2 = QPRISMModelV2
EQGNN = QPRISMModel


# =============================================================================
# VERIFICATION TESTS
# =============================================================================

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
    """
    Comprehensive verification for V2 architecture:
    1. Layer Coordinate Equivariance: x'(Qx + g) == Q x'(x) + g
    2. Layer Feature Invariance: h'(Qx + g) == h'(x)
    3. Global Model Prediction Invariance: y_hat(Qx + g) == y_hat(x)
    """
    print("=" * 65)
    print("🧪 Running Rigorous E(3) Equivariance & Invariance Verification (V2)")
    print("=" * 65)

    h, x, edge_index = _generate_test_geometry()
    q, g = _generate_random_se3()
    x_transformed = x @ q.T + g

    # 1. Test Single EQGNNLayerV2 Equivariance
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

    # 2. Test End-to-End Model Prediction Invariance
    model = EQGNNModelV2(in_features=5, hidden_features=8, num_layers=2)
    model.eval()

    h_m = torch.randn(4, 5)
    batch = torch.tensor([0, 0, 0, 0])
    pred_orig = model(h_m, x, edge_index, batch)
    pred_trans = model(h_m, x_transformed, edge_index, batch)
    pred_inv_error = torch.abs(pred_orig - pred_trans).item()

    print(f"3. End-to-End Global Invariance Error:   {pred_inv_error:.2e}")
    assert pred_inv_error < 1e-5, f"Global Invariance failed with error {pred_inv_error}"

    # 3. Parameter count report
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"4. V2 Model Parameter Count (h=8):       {total_params:,}")

    model_16 = EQGNNModelV2(in_features=11, hidden_features=16, num_layers=2)
    params_16 = sum(p.numel() for p in model_16.parameters() if p.requires_grad)
    print(f"5. V2 Model Parameter Count (h=16):      {params_16:,}")

    print("=" * 65)
    print("✅ All E(3) Equivariance & Invariance Tests PASSED (V2)!")
    print("=" * 65)


def test_rotational_invariance_v1():
    """Backward-compatibility verification for V1 architecture."""
    print("=" * 65)
    print("🧪 Running V1 E(3) Equivariance & Invariance Verification")
    print("=" * 65)

    h, x, edge_index = _generate_test_geometry()
    q, g = _generate_random_se3()
    x_transformed = x @ q.T + g

    # 1. Test Single EQGNNLayerV1 Equivariance
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

    # 2. Test End-to-End V1 Model
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

    print("=" * 65)
    print("✅ V1 E(3) Tests PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    test_rotational_invariance()
    print()
    test_rotational_invariance_v1()
