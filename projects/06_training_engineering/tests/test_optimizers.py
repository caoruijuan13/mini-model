import numpy as np

from optimizers import Adam, SGD


def test_sgd_applies_gradient_and_tracks_step():
    params = {"weight": np.array([1.0, -1.0])}
    optimizer = SGD(learning_rate=0.1)
    optimizer.step(params, {"weight": np.array([0.5, -0.25])})
    assert np.allclose(params["weight"], [0.95, -0.975])
    assert optimizer.step_count == 1


def test_adam_first_step_bias_correction_and_state():
    params = {"weight": np.array([1.0, -1.0])}
    gradients = {"weight": np.array([0.5, -0.25])}
    optimizer = Adam(learning_rate=0.1)
    optimizer.step(params, gradients)
    assert np.allclose(params["weight"], [0.9, -0.9], atol=1e-8)
    assert np.allclose(optimizer.first_moment["weight"], [0.05, -0.025])
    assert np.allclose(optimizer.second_moment["weight"], [0.00025, 0.0000625])
    assert optimizer.step_count == 1


def test_adam_state_round_trip():
    params = {"weight": np.array([1.0, -1.0])}
    optimizer = Adam(learning_rate=0.01)
    optimizer.step(params, {"weight": np.array([0.5, -0.25])})
    restored = Adam(learning_rate=0.01)
    restored.load_state_dict(optimizer.state_dict(), params)
    assert restored.step_count == optimizer.step_count
    assert np.array_equal(
        restored.first_moment["weight"], optimizer.first_moment["weight"]
    )
    assert np.array_equal(
        restored.second_moment["weight"], optimizer.second_moment["weight"]
    )
