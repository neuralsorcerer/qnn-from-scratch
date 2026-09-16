import numpy as np
import pytest

from qnn.optim import Adam, SGD


def test_sgd_step():
    opt = SGD(0.1)
    np.testing.assert_allclose(opt.step(np.array([1.0, 2.0]), np.array([0.5, -1.0])), [0.95, 2.1])


def test_adam_first_step_matches_expected_sign_normalization():
    opt = Adam(0.1)
    actual = opt.step(np.zeros(2), np.array([1.0, -2.0]))
    np.testing.assert_allclose(actual, [-0.1, 0.1], atol=1e-7)


def test_adam_two_steps_matches_reference_equations():
    opt = Adam(learning_rate=0.05, beta1=0.9, beta2=0.999, eps=1e-8)
    params = np.array([0.3, -0.4])
    grads = [np.array([0.2, -0.1]), np.array([0.4, 0.3])]
    ref_m = np.zeros(2); ref_v = np.zeros(2); ref_params = params.copy()
    for t, grad in enumerate(grads, start=1):
        ref_m = 0.9*ref_m + 0.1*grad
        ref_v = 0.999*ref_v + 0.001*(grad*grad)
        m_hat = ref_m/(1-0.9**t)
        v_hat = ref_v/(1-0.999**t)
        ref_params = ref_params - 0.05*m_hat/(np.sqrt(v_hat)+1e-8)
        params = opt.step(params, grad)
    np.testing.assert_allclose(params, ref_params, atol=1e-12)


def test_optimizer_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        Adam().step(np.zeros(2), np.zeros(3))


def test_adam_rejects_boolean_and_complex_beta_values():
    for name in ("beta1", "beta2"):
        with pytest.raises(TypeError):
            Adam(**{name: False})
        with pytest.raises(TypeError):
            Adam(**{name: 0.9 + 0.1j})


def test_adam_accepts_numpy_real_beta_scalars():
    opt = Adam(beta1=np.float64(0.8), beta2=np.float32(0.95))
    assert opt.beta1 == 0.8
    assert np.isclose(opt.beta2, 0.95)


def test_optimizers_reject_boolean_and_string_arrays():
    for optimizer in (SGD(), Adam()):
        with pytest.raises(TypeError, match="real numeric"):
            optimizer.step(np.array([True, False]), np.array([0.1, 0.2]))
        with pytest.raises(TypeError, match="real numeric"):
            optimizer.step(np.array([1.0, 2.0]), np.array(["0.1", "0.2"]))


def test_adam_time_step_is_internal_state_only():
    with pytest.raises(TypeError):
        Adam(t=5)  # type: ignore[call-arg]
