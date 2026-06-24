from .budget_gates import (
    BudgetGatedConv2d,
    collect_budget_loss,
    collect_budget_stats,
    set_gate_stochastic,
    target_budget_loss,
    update_gate_temperature,
)
from .data import HoustonDataBundle, build_houston_raw_loaders, load_houston_scene, write_houston_data_artifacts
from .engine import BRMNetBatch, apply_degradation, evaluate, evaluate_degradation_matrix, train_one_epoch, unpack_batch
from .encoders import BudgetGatedEncoder
from .fusion import ModalityQualityEstimator, ReliabilityGatedFusion, BudgetGatedFusionHead
from .hard_concrete import HardConcreteGate
from .legacy import MODALITY_CHANNELS, build_houston_args, get_houston_loaders, infer_channels
from .model import BRMNet
from .reporting import write_metrics_csv, write_metrics_json

__all__ = [
    "BudgetGatedConv2d",
    "HoustonDataBundle",
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
    "BRMNet",
    "BRMNetBatch",
    "apply_degradation",
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
]
