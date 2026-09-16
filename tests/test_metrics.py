import numpy as np
import pytest

from qnn.metrics import (
    accuracy_score,
    binary_cross_entropy,
    binary_cross_entropy_derivative,
    confusion_matrix_binary,
)


def test_bce_known_value():
    p = np.array([0.25, 0.75])
    y = np.array([0, 1])
    expected = -np.log(0.75)
    assert np.isclose(binary_cross_entropy(p, y), expected, atol=1e-8)


def test_bce_derivative_matches_central_difference_interior():
    p = np.array([0.2, 0.8, 0.55])
    y = np.array([0, 1, 0])
    analytic = binary_cross_entropy_derivative(p, y)
    h = 1e-7
    numeric = np.zeros_like(p)
    for i in range(len(p)):
        plus = p.copy(); plus[i] += h
        minus = p.copy(); minus[i] -= h
        numeric[i] = (binary_cross_entropy(plus, y) - binary_cross_entropy(minus, y)) / (2*h)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-6)


def test_bce_boundary_derivative_matches_one_sided_difference():
    p = np.array([0.0])
    y = np.array([1])
    analytic = binary_cross_entropy_derivative(p, y)[0]
    h = 1e-10
    numeric = (binary_cross_entropy(np.array([h]), y) - binary_cross_entropy(p, y)) / h
    np.testing.assert_allclose(analytic, numeric, rtol=5e-2)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 1.1])
def test_metrics_reject_invalid_probabilities(bad):
    for fn in (binary_cross_entropy, accuracy_score, confusion_matrix_binary):
        with pytest.raises(ValueError):
            fn(np.array([0.2, bad]), np.array([0, 1]))


def test_metrics_reject_non_binary_labels():
    for fn in (binary_cross_entropy, accuracy_score, confusion_matrix_binary):
        with pytest.raises(ValueError, match="binary labels"):
            fn(np.array([0.2, 0.8]), np.array([0, 2]))


def test_confusion_matrix_counts():
    result = confusion_matrix_binary(
        np.array([0.1, 0.9, 0.7, 0.2]),
        np.array([0, 1, 0, 1]),
    )
    assert result == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_metrics_reject_complex_probabilities_instead_of_dropping_imaginary_part():
    import pytest

    with pytest.raises(TypeError, match="real-valued"):
        binary_cross_entropy(np.array([0.2 + 0.1j, 0.8]), np.array([0, 1]))


def test_metrics_reject_boolean_and_string_probability_arrays():
    import pytest

    y = np.array([0, 1])
    for bad in (np.array([True, False]), np.array(["0.2", "0.8"])):
        with pytest.raises(TypeError, match="real numeric"):
            binary_cross_entropy(bad, y)
