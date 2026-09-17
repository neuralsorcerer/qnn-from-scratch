import numpy as np
import pytest

from qnn.datasets import make_classification_dataset, train_test_split


def test_vertical_horizontal_semantics_without_noise():
    _, yv, raw_v = make_classification_dataset(64, seed=4, kind="vertical", noise=0.0)
    _, yh, raw_h = make_classification_dataset(64, seed=4, kind="horizontal", noise=0.0)
    np.testing.assert_array_equal(yv, (raw_v[:, 0] > 0).astype(np.int64))
    np.testing.assert_array_equal(yh, (raw_h[:, 1] > 0).astype(np.int64))


def test_dataset_is_deterministic_for_same_seed():
    a = make_classification_dataset(32, seed=9, kind="sine", noise=0.1)
    b = make_classification_dataset(32, seed=9, kind="sine", noise=0.1)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_angle_features_are_raw_times_pi():
    X, _, raw = make_classification_dataset(20, seed=1)
    np.testing.assert_allclose(X, raw * np.pi)


def test_split_guarantees_nonempty_partitions():
    X = np.arange(4).reshape(2, 2)
    y = np.array([0, 1])
    split = train_test_split(X, y, X, test_size=0.99, shuffle=False)
    assert len(split.X_train) == 1
    assert len(split.X_test) == 1


def test_split_preserves_alignment():
    X = np.column_stack([np.arange(10), -np.arange(10)])
    y = np.arange(10)
    raw = X + 100
    split = train_test_split(X, y, raw, test_size=0.3, seed=3)
    for block_x, block_y, block_raw in [
        (split.X_train, split.y_train, split.raw_train),
        (split.X_test, split.y_test, split.raw_test),
    ]:
        for row, label, raw_row in zip(block_x, block_y, block_raw):
            assert row[0] == label
            np.testing.assert_array_equal(raw_row, row + 100)


@pytest.mark.parametrize("bad_noise", [-1.0, np.nan, np.inf])
def test_bad_noise_rejected(bad_noise):
    with pytest.raises(ValueError):
        make_classification_dataset(10, noise=bad_noise)


def test_circle_without_noise_uses_radial_score():
    X, y, raw = make_classification_dataset(num_samples=64, seed=21, kind="circle", noise=0.0)
    del X
    expected = (np.sqrt(raw[:, 0] ** 2 + raw[:, 1] ** 2) > 0.68).astype(np.int64)
    np.testing.assert_array_equal(y, expected)


def test_split_rejects_nonfinite_and_complex_arrays():
    X = np.array([[0.0, 0.0], [1.0, np.nan]])
    y = np.array([0, 1])
    raw = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        train_test_split(X, y, raw)

    Xc = np.array([[0.0 + 1j, 0.0], [1.0, 1.0]])
    with pytest.raises(TypeError, match="real-valued"):
        train_test_split(Xc, y, raw)


def test_split_rejects_string_boolean_and_array_test_size():
    X = np.zeros((4, 2), dtype=np.float64)
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    raw = X.copy()
    for bad in ("0.25", True, np.array(0.25)):
        with pytest.raises(TypeError):
            train_test_split(X, y, raw, test_size=bad)  # type: ignore[arg-type]


def test_split_rejects_boolean_and_string_arrays():
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    raw = np.zeros((4, 2), dtype=np.float64)
    with pytest.raises(TypeError, match="real numeric"):
        train_test_split(np.zeros((4, 2), dtype=bool), y, raw)
    with pytest.raises(TypeError, match="real numeric"):
        train_test_split(np.full((4, 2), "0.1"), y, raw)


def test_split_preserves_numeric_input_dtypes():
    X = np.zeros((4, 2), dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.int16)
    raw = np.ones((4, 2), dtype=np.float32)
    split = train_test_split(X, y, raw, test_size=0.25, seed=1)
    assert split.X_train.dtype == np.float32
    assert split.X_test.dtype == np.float32
    assert split.y_train.dtype == np.int16
    assert split.y_test.dtype == np.int16
    assert split.raw_train.dtype == np.float32
    assert split.raw_test.dtype == np.float32
