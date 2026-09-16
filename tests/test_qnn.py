import numpy as np
import pytest

from qnn.metrics import binary_cross_entropy
from qnn.qnn import DataReuploadingQNN


def test_one_qubit_constructor_chooses_valid_default_observable():
    model = DataReuploadingQNN(num_qubits=1, num_features=2)
    assert model.observable_wire == 0


def test_feature_mapping_rejects_silently_ignored_features():
    with pytest.raises(ValueError, match="never be uploaded"):
        DataReuploadingQNN(num_qubits=2, num_features=4)


def test_qnn_probability_is_valid_and_matches_direct_marginal():
    model = DataReuploadingQNN(num_qubits=2, num_layers=2, seed=123)
    X = np.array([[0.1, -0.2], [0.4, 0.7]])
    p = model.predict_proba(X)
    assert p.shape == (2,)
    assert np.all((0.0 <= p) & (p <= 1.0))
    for row, probability in zip(X, p):
        state = model.forward_state(row)
        direct = model.sim.probability_one(state, int(model.observable_wire))
        np.testing.assert_allclose(probability, direct, atol=1e-12)


def test_set_parameters_copies_input():
    model = DataReuploadingQNN(num_layers=1, seed=2)
    params = np.ones_like(model.params)
    model.set_parameters(params)
    params[:] = 9.0
    assert np.all(model.params == 1.0)


def test_parameter_shift_gradient_matches_finite_difference_bce_loss_multilayer():
    model = DataReuploadingQNN(num_qubits=2, num_layers=2, seed=11)
    X = np.array([[0.15, -0.1], [0.35, 0.55], [-0.7, 0.2]])
    y = np.array([0, 1, 0])
    analytic = model.parameter_shift_gradient(X, y)

    h = 1e-6
    numeric = np.zeros_like(model.params)
    base = model.params.copy()
    for idx in np.ndindex(base.shape):
        plus = base.copy(); plus[idx] += h
        minus = base.copy(); minus[idx] -= h
        numeric[idx] = (
            binary_cross_entropy(model.predict_proba(X, plus), y)
            - binary_cross_entropy(model.predict_proba(X, minus), y)
        ) / (2*h)
    np.testing.assert_allclose(analytic, numeric, atol=2e-5, rtol=2e-5)


def test_forward_rejects_nonfinite_features():
    model = DataReuploadingQNN()
    with pytest.raises(ValueError, match="finite"):
        model.forward_state(np.array([np.nan, 0.0]))


def test_predict_rejects_bad_threshold():
    model = DataReuploadingQNN()
    with pytest.raises(ValueError):
        model.predict(np.array([[0.0, 0.0]]), threshold=1.2)


def test_qnn_rejects_complex_features_and_parameters():
    model = DataReuploadingQNN(num_layers=1)
    with pytest.raises(TypeError, match="real-valued"):
        model.forward_state(np.array([0.1 + 0.2j, 0.3]))
    with pytest.raises(TypeError, match="real-valued"):
        model.set_parameters(model.params.astype(np.complex128) + 1j)


def test_parameter_shift_rejects_complex_labels():
    model = DataReuploadingQNN(num_qubits=1, num_layers=1, num_features=1, seed=3)
    X = np.array([[0.1], [0.2]])
    with pytest.raises(TypeError, match="y must be real-valued"):
        model.parameter_shift_gradient(X, np.array([0 + 0j, 1 + 0.2j]))


def test_qnn_rejects_boolean_and_string_features_and_parameters():
    model = DataReuploadingQNN(num_qubits=1, num_layers=1, num_features=2, seed=3)
    for bad_x in (np.array([[True, False]]), np.array([["0.1", "0.2"]])):
        with pytest.raises(TypeError, match="real numeric"):
            model.predict_proba(bad_x)
    for bad_params in (
        np.ones(model.params.shape, dtype=bool),
        np.full(model.params.shape, "0.1"),
    ):
        with pytest.raises(TypeError, match="real numeric"):
            model.set_parameters(bad_params)
