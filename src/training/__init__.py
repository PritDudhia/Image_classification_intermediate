"""Training module initialization."""

from .trainer import Trainer
from .losses import (
    LabelSmoothingCrossEntropy,
    FocalLoss,
    DistillationLoss,
    get_criterion
)
from .metrics import (
    MetricTracker,
    top_k_accuracy,
    calibration_error,
    AverageMeter
)
from .callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    GradualUnfreezing
)

__all__ = [
    'Trainer',
    'LabelSmoothingCrossEntropy',
    'FocalLoss',
    'DistillationLoss',
    'get_criterion',
    'MetricTracker',
    'top_k_accuracy',
    'calibration_error',
    'AverageMeter',
    'EarlyStopping',
    'ModelCheckpoint',
    'LearningRateScheduler',
    'GradualUnfreezing',
]
