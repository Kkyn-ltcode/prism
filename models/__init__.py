# Q-PRISM (Quantum)
from .eq_gnn import (
    QPRISMLayer, QPRISMModel, QPRISM,
    QPRISMLayerV1, QPRISMModelV1,
    QPRISMLayerV2, QPRISMModelV2,
    GaussianRBF, QuantumHead,
    compute_edge_mean_angles,
    # Backward compatibility aliases
    EQGNNLayer, EQGNNModel, EQGNN,
    EQGNNLayerV1, EQGNNModelV1,
    EQGNNLayerV2, EQGNNModelV2,
)

# PRISM (Classical Periodic Twin)
from .classical_trig_twin import (
    PRISMHead, PRISMLayer, PRISMModel, PRISM,
    PRISMLayerV2, PRISMModelV2,
    FourierFeatureHead,
    create_param_matched_prism,
    # Backward compatibility aliases
    TrigHead, TrigTwinLayerV2, TrigTwinModelV2,
    create_param_matched_trig_twin, create_param_matched_trigtwin, PeriodicEGNN,
)

# Classical Baselines
from .egnn_baseline import (
    EGNNLayer, EGNNModel,
    ParamMatchedEGNNModel,
    count_parameters, create_param_matched_egnn,
)
from .classical_twin import (
    ClassicalHead, ClassicalTwinLayerV2, ClassicalTwinModelV2,
    ClassicalTwinLayerV1, ClassicalTwinTrajectoryModel,
    create_param_matched_classical_twin,
)
from .classical_baselines import (
    SchNetModel, PaiNNModel,
    create_param_matched_schnet, create_param_matched_painn,
)

__all__ = [
    # Q-PRISM
    "QPRISMLayer", "QPRISMModel", "QPRISM",
    "QPRISMLayerV1", "QPRISMModelV1",
    "QPRISMLayerV2", "QPRISMModelV2",
    "EQGNNLayer", "EQGNNModel", "EQGNN",
    "EQGNNLayerV1", "EQGNNModelV1",
    "EQGNNLayerV2", "EQGNNModelV2",
    "GaussianRBF", "QuantumHead",
    "compute_edge_mean_angles",
    # PRISM
    "PRISMHead", "PRISMLayer", "PRISMModel", "PRISM",
    "PRISMLayerV2", "PRISMModelV2",
    "FourierFeatureHead",
    "create_param_matched_prism",
    "TrigHead", "TrigTwinLayerV2", "TrigTwinModelV2",
    "create_param_matched_trig_twin", "create_param_matched_trigtwin", "PeriodicEGNN",
    # EGNN & Classical Baselines
    "EGNNLayer", "EGNNModel",
    "ParamMatchedEGNNModel",
    "count_parameters", "create_param_matched_egnn",
    "ClassicalHead", "ClassicalTwinLayerV2", "ClassicalTwinModelV2",
    "create_param_matched_classical_twin",
    "SchNetModel", "PaiNNModel",
    "create_param_matched_schnet", "create_param_matched_painn",
]
