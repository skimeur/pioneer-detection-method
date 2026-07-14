"""Sanity tests for the pioneer-detection package."""

import numpy as np
import pandas as pd
import pytest

from pioneer_detection import (
    compute_pioneer_weights_angles,
    compute_pioneer_weights_distance,
    compute_granger_weights,
    compute_lagged_correlation_weights,
    compute_multivariate_regression_weights,
    compute_transfer_entropy_weights,
    compute_linear_pooling_weights,
    compute_median_pooling,
    pooled_forecast,
)


@pytest.fixture
def forecasts():
    """Expert 1 moves first; experts 2-3 converge toward it."""
    rng = np.random.default_rng(42)
    t = 30
    pioneer = np.linspace(3.0, 1.5, t)
    follower1 = np.concatenate([np.full(5, 3.0), np.linspace(3.0, 1.5, t - 5)])
    follower2 = np.concatenate([np.full(8, 3.0), np.linspace(3.0, 1.5, t - 8)])
    data = np.column_stack([pioneer, follower1, follower2])
    data += rng.normal(0, 0.01, data.shape)
    return pd.DataFrame(data, columns=["E1", "E2", "E3"])


def test_angle_weights_sum_to_one(forecasts):
    w = compute_pioneer_weights_angles(forecasts)
    sums = w.sum(axis=1, min_count=1).dropna()
    assert np.allclose(sums, 1.0)


def test_distance_weights_sum_to_one(forecasts):
    w = compute_pioneer_weights_distance(forecasts)
    sums = w.sum(axis=1, min_count=1).dropna()
    assert np.allclose(sums, 1.0)


def test_pdm_identifies_pioneer(forecasts):
    w = compute_pioneer_weights_angles(forecasts)
    avg = w.mean()
    assert avg["E1"] == avg.max()


def test_pooled_forecast_fallback_is_mean(forecasts):
    w = pd.DataFrame(np.nan, index=forecasts.index, columns=forecasts.columns)
    pooled = pooled_forecast(forecasts, w)
    assert np.allclose(pooled, forecasts.mean(axis=1))


def test_linear_pooling_weights(forecasts):
    w = compute_linear_pooling_weights(forecasts)
    assert np.allclose(w.values, 1.0 / 3.0)


def test_median_pooling(forecasts):
    pooled = compute_median_pooling(forecasts)
    assert len(pooled) == len(forecasts)


@pytest.mark.parametrize(
    "method",
    [
        compute_granger_weights,
        compute_lagged_correlation_weights,
        compute_multivariate_regression_weights,
        compute_transfer_entropy_weights,
    ],
)
def test_alternative_methods_produce_valid_weights(method, forecasts):
    w = method(forecasts)
    assert w.shape == forecasts.shape
    sums = w.sum(axis=1).dropna()
    assert np.allclose(sums, 1.0)


def test_backwards_compatible_pdm_shim():
    from pdm import compute_pioneer_weights_angles as legacy
    assert legacy is compute_pioneer_weights_angles
