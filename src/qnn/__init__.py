"""Quantum Neural Network from scratch.

This package intentionally avoids quantum SDK dependencies. It provides a
small but production-structured statevector simulator, a data-reuploading QNN,
parameter-shift gradients, optimizers, metrics, plotting, tests, and a CLI.
"""

from qnn.datasets import make_classification_dataset, train_test_split
from qnn.metrics import accuracy_score, binary_cross_entropy
from qnn.qnn import DataReuploadingQNN
from qnn.trainer import Trainer, TrainingConfig

__all__ = [
    "DataReuploadingQNN",
    "Trainer",
    "TrainingConfig",
    "accuracy_score",
    "binary_cross_entropy",
    "make_classification_dataset",
    "train_test_split",
]
