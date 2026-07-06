from .budget_gates import (
    BudgetGatedConv2d,
    collect_budget_loss,
    collect_budget_stats,
    set_gate_stochastic,
    target_budget_loss,
    update_gate_temperature,
)
from .data import HoustonDataBundle, build_houston_raw_loaders, load_houston_scene, write_houston_data_artifacts
from .compact import CompactBRMNet
from .engine import (
    BRMNetBatch,
    apply_degradation,
    apply_modality_dropout,
    evaluate,
    evaluate_degradation_matrix,
    train_one_epoch,
    unpack_batch,
)
from .encoders import BudgetGatedEncoder
from .fusion import ModalityQualityEstimator, ReliabilityGatedFusion, BudgetGatedFusionHead
from .gated_blocks import HardConcreteConvBlock
from .hard_concrete import (
    HardConcreteGate,
    iter_hard_concrete_gates,
    set_hard_concrete_inference_mode,
    set_hard_concrete_stochastic,
)
from .legacy import MODALITY_CHANNELS, build_houston_args, get_houston_loaders, infer_channels
from .model import BRMNet
from .reporting import write_metrics_csv, write_metrics_json
from .export import export_compact_brmnet
from .resources import (
    BRMNetResourceStats,
    estimate_brmnet_resources,
    estimate_compact_resources,
    find_resource_budget_threshold,
    initialize_uniform_resource_budget,
    resource_budget_loss,
)

__all__ = [
    "BudgetGatedConv2d",
    "HoustonDataBundle",
    "CompactBRMNet",
    "build_houston_raw_loaders",
    "load_houston_scene",
    "write_houston_data_artifacts",
    "collect_budget_loss",
    "collect_budget_stats",
    "target_budget_loss",
    "set_gate_stochastic",
    "update_gate_temperature",
    "BudgetGatedEncoder",
    "ModalityQualityEstimator",
    "ReliabilityGatedFusion",
    "BudgetGatedFusionHead",
    "HardConcreteGate",
    "HardConcreteConvBlock",
    "iter_hard_concrete_gates",
    "set_hard_concrete_inference_mode",
    "set_hard_concrete_stochastic",
    "BRMNet",
    "BRMNetBatch",
    "apply_degradation",
    "apply_modality_dropout",
    "evaluate",
    "evaluate_degradation_matrix",
    "train_one_epoch",
    "unpack_batch",
    "MODALITY_CHANNELS",
    "build_houston_args",
    "get_houston_loaders",
    "infer_channels",
    "write_metrics_csv",
    "write_metrics_json",
    "BRMNetResourceStats",
    "estimate_brmnet_resources",
    "resource_budget_loss",
    "estimate_compact_resources",
    "initialize_uniform_resource_budget",
    "find_resource_budget_threshold",
    "export_compact_brmnet",
]
