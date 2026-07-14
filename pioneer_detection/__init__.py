"""
pioneer-detection
=================

Official Python implementation of the Pioneer Detection Method (PDM),
a convergence-based expert-aggregation algorithm for environments with
structural change, introduced in:

    Vansteenberghe, Eric (2026),
    "Insurance supervision under climate change: a pioneer detection method,"
    The Geneva Papers on Risk and Insurance - Issues and Practice, 51(1), 176-207,
    https://doi.org/10.1057/s41288-025-00367-y

Quick start
-----------
>>> import pandas as pd
>>> from pioneer_detection import compute_pioneer_weights_angles, pooled_forecast
>>> forecasts = pd.DataFrame({
...     "E1": [1.0, 1.1, 1.2, 1.3],
...     "E2": [0.5, 0.5, 0.9, 1.2],
...     "E3": [0.4, 0.4, 0.8, 1.1],
... })
>>> weights = compute_pioneer_weights_angles(forecasts)
>>> pooled = pooled_forecast(forecasts, weights)
"""

from pioneer_detection.core import (
    # PDM variants
    compute_pioneer_weights_angles,
    compute_pioneer_weights_distance,
    # Alternative inter-temporal pioneer detection methods
    compute_granger_weights,
    compute_lagged_correlation_weights,
    compute_multivariate_regression_weights,
    compute_transfer_entropy_weights,
    # Traditional benchmarks
    compute_linear_pooling_weights,
    compute_median_pooling,
    # Shared utility
    pooled_forecast,
    # Backward-compatible aliases
    compute_pioneer_weights_simple,
    pooled_forecast_simple,
)

__version__ = "1.0.0"

__all__ = [
    "compute_pioneer_weights_angles",
    "compute_pioneer_weights_distance",
    "compute_granger_weights",
    "compute_lagged_correlation_weights",
    "compute_multivariate_regression_weights",
    "compute_transfer_entropy_weights",
    "compute_linear_pooling_weights",
    "compute_median_pooling",
    "pooled_forecast",
    "compute_pioneer_weights_simple",
    "pooled_forecast_simple",
]
