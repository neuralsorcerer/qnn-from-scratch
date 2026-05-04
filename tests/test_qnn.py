import numpy as np

from qnn.qnn import DataReuploadingQNN


def test_qnn_probability_is_valid():
    """Check predicted probabilities are bounded and have expected shape."""
    model = DataReuploadingQNN(seed=123)
    X = np.array([[0.1, -0.2], [0.4, 0.7]])
    p = model.predict_proba(X)
    assert p.shape == (2,)
    assert np.all((0.0 <= p) & (p <= 1.0))


def test_parameter_shift_gradient_shape_and_finite_values():
    """Check parameter-shift gradients match parameter shape and stay finite."""
    model = DataReuploadingQNN(seed=123)
    X = np.array([[0.1, -0.2], [0.4, 0.7], [-0.3, 0.2]])
    y = np.array([0, 1, 0])
    grad = model.parameter_shift_gradient(X, y)
    assert grad.shape == model.params.shape
    assert np.all(np.isfinite(grad))


def test_parameter_shift_gradient_matches_finite_difference_bce_loss():
    """Validate parameter-shift gradient against finite-difference loss gradients."""
    model = DataReuploadingQNN(num_qubits=2, num_layers=1, seed=11)
    X = np.array([[0.15, -0.1], [0.35, 0.55]])
    y = np.array([0, 1])

    analytic_grad = model.parameter_shift_gradient(X, y)

    def loss_for_params(params: np.ndarray) -> float:
        proba = model.predict_proba(X, params=params)
        proba = np.clip(proba, 1e-9, 1.0 - 1e-9)
        return float(np.mean(-(y * np.log(proba) + (1.0 - y) * np.log(1.0 - proba))))

    eps = 1e-6
    numeric_grad = np.zeros_like(model.params)
    for idx in np.ndindex(model.params.shape):
        plus = model.params.copy()
        minus = model.params.copy()
        plus[idx] += eps
        minus[idx] -= eps
        numeric_grad[idx] = (loss_for_params(plus) - loss_for_params(minus)) / (2.0 * eps)

    assert np.allclose(analytic_grad, numeric_grad, atol=1e-4, rtol=1e-4)
