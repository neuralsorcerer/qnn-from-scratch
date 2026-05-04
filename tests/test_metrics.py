import numpy as np

from qnn.metrics import accuracy_score, binary_cross_entropy, confusion_matrix_binary


def test_metrics_reject_non_binary_labels():
    """Ensure metric helpers reject labels outside {0, 1}."""
    proba = np.array([0.2, 0.8, 0.6])
    y = np.array([0, 2, 1])
    for fn in (binary_cross_entropy, accuracy_score, confusion_matrix_binary):
        try:
            fn(proba, y)
        except ValueError as exc:
            assert "binary labels" in str(exc)
        else:
            raise AssertionError("Expected ValueError for non-binary labels")


def test_metrics_reject_length_mismatch():
    """Ensure metric helpers reject inconsistent prediction/label lengths."""
    proba = np.array([0.2, 0.8, 0.6])
    y = np.array([0, 1])
    try:
        binary_cross_entropy(proba, y)
    except ValueError as exc:
        assert "must contain 3 elements" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched lengths")
