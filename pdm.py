"""
Pioneer Detection Method (PDM) and Alternative Approaches
==========================================================

Python implementations of the Pioneer Detection Method (PDM) and alternative
inter-temporal pioneer detection methods, as introduced and compared in:

    Vansteenberghe, Eric (2025),
    "Insurance Supervision under Climate Change: A Pioneer Detection Method,"
    The Geneva Papers on Risk and Insurance - Issues and Practice,
    https://doi.org/10.1057/s41288-025-00367-y

Methods implemented
-------------------
PDM variants:
  1. PDM with distances  (compute_pioneer_weights_distance)
  2. PDM with angles      (compute_pioneer_weights_angles)

Alternative inter-temporal pioneer detection methods:
  3. Granger Causality        (compute_granger_weights)
  4. Lagged Correlation        (compute_lagged_correlation_weights)
  5. Multivariate Linear Regressions (compute_multivariate_regression_weights)
  6. Transfer Entropy           (compute_transfer_entropy_weights)

Traditional benchmarks:
  7. Linear Opinion Pooling    (compute_linear_pooling_weights)
  8. Median Pooling             (compute_median_pooling)

Shared utility:
  - pooled_forecast  (weighted combination with mean fallback)
"""

import pandas as pd
import numpy as np

__all__ = [
    # PDM variants
    "compute_pioneer_weights_angles",
    "compute_pioneer_weights_distance",
    # Alternative methods
    "compute_granger_weights",
    "compute_lagged_correlation_weights",
    "compute_multivariate_regression_weights",
    "compute_transfer_entropy_weights",
    # Traditional benchmarks
    "compute_linear_pooling_weights",
    "compute_median_pooling",
    # Shared utility
    "pooled_forecast",
    # Backward-compatible aliases
    "compute_pioneer_weights_simple",
    "pooled_forecast_simple",
]


# ---------------------------------------------------------------------------
# Helper: leave-one-out mean for each expert
# ---------------------------------------------------------------------------

def _leave_one_out_mean(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the leave-one-out mean for each expert.

    For expert i, this is the mean of all other experts at each time step.
    Used internally by all methods as the cross-sectional benchmark m_{-i}.

    Parameters
    ----------
    X : pd.DataFrame
        (T x N) matrix of expert estimates, already cast to float.

    Returns
    -------
    m_minus : pd.DataFrame
        Same shape as X. Column i contains the mean of all columns except i.
    """
    m_minus = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for col in X.columns:
        others = X.drop(columns=col)
        m_minus[col] = others.mean(axis=1)
    return m_minus


# ---------------------------------------------------------------------------
# 1. PDM with distances (original simple version)
# ---------------------------------------------------------------------------

def compute_pioneer_weights_distance(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    PDM with distance-based weighting (Appendix A.2 of the paper).

    Steps 1-2 are identical to the angle-based PDM. Step 3 uses y-axis
    distances instead of angles:
        w_i^t = delta_distance * delta_orientation * |Delta_{-i}| / (|Delta_{-i}| + |Delta_i|)

    This variant is found to be non-robust (see Table 2 in the paper).

    Parameters
    ----------
    forecasts : pd.DataFrame
        (T x N) DataFrame where rows are time periods and columns are experts.
        Values must be numeric (int or float).

    Returns
    -------
    weights : pd.DataFrame
        Same shape as ``forecasts``. Contains normalised pioneer weights in
        [0, 1] that sum to 1 across experts at each time step where at least
        one pioneer is detected.  Rows with no pioneer contain NaN.

    Examples
    --------
    >>> import pandas as pd
    >>> forecasts = pd.DataFrame({
    ...     "E1": [1.0, 1.1, 1.2, 1.3],
    ...     "E2": [0.5, 0.5, 0.9, 1.2],
    ...     "E3": [0.4, 0.4, 0.8, 1.1],
    ... })
    >>> w = compute_pioneer_weights_distance(forecasts)
    >>> pooled = pooled_forecast(forecasts, w)
    """
    X = forecasts.astype(float)
    m_minus = _leave_one_out_mean(X)

    delta_X = X.diff()
    delta_m = m_minus.diff()

    # Step 1: distance reduction
    distance = (X - m_minus).abs()
    distance_prev = distance.shift(1)
    cond_distance = distance < distance_prev

    # Step 2: orientation (peers move more than expert)
    cond_orientation = delta_m.abs() > delta_X.abs()

    # Step 3: proportion (distance-based)
    denom = delta_m.abs() + delta_X.abs()
    proportion = delta_m.abs() / denom

    mask = cond_distance & cond_orientation & (denom > 0)
    raw = proportion.where(mask, 0.0)

    row_sums = raw.sum(axis=1)
    weights = raw.div(row_sums.replace(0.0, np.nan), axis=0)
    return weights


# Keep backward-compatible alias
compute_pioneer_weights_simple = compute_pioneer_weights_distance


# ---------------------------------------------------------------------------
# 2. PDM with angles (preferred method)
# ---------------------------------------------------------------------------

def compute_pioneer_weights_angles(
    forecasts: pd.DataFrame,
    step: float = 1.0,
) -> pd.DataFrame:
    """
    PDM with angle-based weighting (Equation 4-5 of the paper).

    The angle theta between the movement vector and the horizontal captures
    the *speed* of convergence. The weight attributed to expert i is:
        w_i^t = delta_distance * delta_orientation * |theta_{-i}| / (|theta_{-i}| + |theta_i|)

    where theta = arccos((s^2 + u_y * v_y) / (sqrt(s^2 + u_y^2) * sqrt(s^2 + v_y^2)))
    and s is the time step between observations.

    This is the preferred approach in the paper.

    Parameters
    ----------
    forecasts : pd.DataFrame
        (T x N) DataFrame where rows are time periods and columns are experts.
        Values must be numeric (int or float).
    step : float
        Time step between observations (the x-component of both vectors).
        Default is 1.0 for unit-spaced observations. Set to the actual
        inter-observation interval when observations are not unit-spaced
        (e.g., 12 for annual data indexed monthly).

    Returns
    -------
    weights : pd.DataFrame
        Same shape as ``forecasts``. Contains normalised pioneer weights in
        [0, 1] that sum to 1 across experts at each time step where at least
        one pioneer is detected.  Rows with no pioneer contain NaN.

    Examples
    --------
    >>> import pandas as pd
    >>> forecasts = pd.DataFrame({
    ...     "E1": [1.0, 1.1, 1.2, 1.3],
    ...     "E2": [0.5, 0.5, 0.9, 1.2],
    ...     "E3": [0.4, 0.4, 0.8, 1.1],
    ... })
    >>> w = compute_pioneer_weights_angles(forecasts)
    >>> pooled = pooled_forecast(forecasts, w)
    """
    X = forecasts.astype(float)
    m_minus = _leave_one_out_mean(X)

    delta_X = X.diff()
    delta_m = m_minus.diff()

    # Step 1: distance reduction
    distance = (X - m_minus).abs()
    distance_prev = distance.shift(1)
    cond_distance = distance < distance_prev

    # Step 2: orientation (peers move more — checked via angles)
    s2 = step ** 2

    def _angle(dy):
        """Angle between the movement vector (step, dy) and horizontal (step, 0)."""
        # theta = arccos(s^2 / (sqrt(s^2 + dy^2) * s))  = arctan(|dy| / s)
        return np.arctan2(dy.abs(), step)

    theta_i = _angle(delta_X)    # expert's own movement angle
    theta_mi = _angle(delta_m)   # peers' movement angle

    cond_orientation = theta_mi > theta_i

    # Step 3: proportion (angle-based)
    denom = theta_mi + theta_i
    proportion = theta_mi / denom

    mask = cond_distance & cond_orientation & (denom > 0)
    raw = proportion.where(mask, 0.0)

    row_sums = raw.sum(axis=1)
    weights = raw.div(row_sums.replace(0.0, np.nan), axis=0)
    return weights


# ---------------------------------------------------------------------------
# 3. Granger Causality weights (Appendix A.3)
# ---------------------------------------------------------------------------

def compute_granger_weights(
    forecasts: pd.DataFrame,
    maxlag: int = 1,
) -> pd.DataFrame:
    """
    Granger Causality-based pioneer weights (Appendix A.3).

    For each expert i, test whether i Granger-causes the leave-one-out mean
    of the other experts. Experts whose past values significantly predict
    the group's future values receive higher weights.

    The weight is proportional to (1 - p-value) of the F-test, normalised
    across experts at each t (rolling window not used; one weight per expert
    over the full sample, broadcast to all t).

    Requires statsmodels.

    Parameters
    ----------
    forecasts : pd.DataFrame
        (T x N) expert forecasts.
    maxlag : int
        Maximum lag for the Granger test (default 1).

    Returns
    -------
    weights : pd.DataFrame
        Constant weights broadcast across time.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    X = forecasts.astype(float).dropna()
    m_minus = _leave_one_out_mean(X)

    scores = {}
    for col in X.columns:
        data = pd.concat([m_minus[col], X[col]], axis=1).dropna()
        if len(data) < maxlag + 3:
            scores[col] = 0.0
            continue
        try:
            res = grangercausalitytests(data.values, maxlag=maxlag, verbose=False)
            min_p = min(res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1))
            scores[col] = max(1.0 - min_p, 0.0)
        except Exception:
            scores[col] = 0.0

    total = sum(scores.values())
    if total == 0:
        w = {col: 1.0 / len(X.columns) for col in X.columns}
    else:
        w = {col: v / total for col, v in scores.items()}

    weights = pd.DataFrame(
        {col: [w[col]] * len(forecasts) for col in forecasts.columns},
        index=forecasts.index,
    )
    return weights


# ---------------------------------------------------------------------------
# 4. Lagged Correlation weights (Appendix A.4)
# ---------------------------------------------------------------------------

def compute_lagged_correlation_weights(
    forecasts: pd.DataFrame,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Lagged-correlation pioneer weights (Appendix A.4).

    Measures Pearson correlation between lagged expert i and the current
    leave-one-out mean. A high correlation means expert i's past values
    predict where the group is heading.

    Parameters
    ----------
    forecasts : pd.DataFrame
        (T x N) expert forecasts.
    lag : int
        Number of periods to lag expert i (default 1).

    Returns
    -------
    weights : pd.DataFrame
        Constant weights broadcast across time.
    """
    X = forecasts.astype(float).dropna()
    m_minus = _leave_one_out_mean(X)

    scores = {}
    for col in X.columns:
        lagged_expert = X[col].shift(lag)
        valid = pd.concat([lagged_expert, m_minus[col]], axis=1).dropna()
        if len(valid) < 3:
            scores[col] = 0.0
            continue
        corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
        scores[col] = max(corr, 0.0)

    total = sum(scores.values())
    if total == 0:
        w = {col: 1.0 / len(X.columns) for col in X.columns}
    else:
        w = {col: v / total for col, v in scores.items()}

    weights = pd.DataFrame(
        {col: [w[col]] * len(forecasts) for col in forecasts.columns},
        index=forecasts.index,
    )
    return weights


# ---------------------------------------------------------------------------
# 5. Multivariate Linear Regression weights (Appendix A.6)
# ---------------------------------------------------------------------------

def compute_multivariate_regression_weights(
    forecasts: pd.DataFrame,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Multivariate linear regression pioneer weights (Appendix A.6).

    For each expert i, regress the leave-one-out mean (at time t) on
    expert i's lagged estimate (at time t-lag). The regression coefficient
    serves as the voting weight when significant.

    Following Yi et al. (2000), significant coefficients are interpreted
    as evidence of pioneership.

    Parameters
    ----------
    forecasts : pd.DataFrame
        (T x N) expert forecasts.
    lag : int
        Lag of the regressor (default 1).

    Returns
    -------
    weights : pd.DataFrame
        Constant weights broadcast across time.
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    X = forecasts.astype(float).dropna()
    m_minus = _leave_one_out_mean(X)

    scores = {}
    for col in X.columns:
        y = m_minus[col].iloc[lag:]
        x = X[col].iloc[:-lag].values if lag > 0 else X[col].values
        if len(y) < 4:
            scores[col] = 0.0
            continue
        x_const = add_constant(x)
        try:
            res = OLS(y.values, x_const).fit()
            coef = res.params[1]
            pval = res.pvalues[1]
            scores[col] = max(coef, 0.0) if pval < 0.10 else 0.0
        except Exception:
            scores[col] = 0.0

    total = sum(scores.values())
    if total == 0:
        w = {col: 1.0 / len(X.columns) for col in X.columns}
    else:
        w = {col: v / total for col, v in scores.items()}

    weights = pd.DataFrame(
        {col: [w[col]] * len(forecasts) for col in forecasts.columns},
        index=forecasts.index,
    )
    return weights


# ---------------------------------------------------------------------------
# 6. Transfer Entropy weights (Appendix A.7)
# ---------------------------------------------------------------------------

def compute_transfer_entropy_weights(
    forecasts: pd.DataFrame,
    n_bins: int = 3,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Transfer entropy pioneer weights (Appendix A.7, Schreiber 2000).

    Measures information transfer from expert i to the leave-one-out mean.
    Continuous time series are discretized into bins (Dimpfl & Peter 2014
    recommend 3 bins along the 5% and 95% quantiles).

    Transfer entropy from X to Y is:
        TE_{X->Y} = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})

    estimated from empirical joint histograms.

    Parameters
    ----------
    forecasts : pd.DataFrame
        (T x N) expert forecasts.
    n_bins : int
        Number of bins for discretization (default 3).
    lag : int
        Lag for transfer entropy estimation (default 1).

    Returns
    -------
    weights : pd.DataFrame
        Constant weights broadcast across time.
    """
    X = forecasts.astype(float).dropna()
    m_minus = _leave_one_out_mean(X)

    def _discretize(series, n_bins):
        """Discretize a series into n_bins using quantile boundaries."""
        quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
        boundaries = np.quantile(series.dropna(), quantiles)
        return np.digitize(series.values, boundaries)

    def _transfer_entropy(source, target, lag):
        """Estimate transfer entropy from source to target."""
        src_d = _discretize(source, n_bins)
        tgt_d = _discretize(target, n_bins)

        n = len(src_d)
        if n <= lag + 1:
            return 0.0

        tgt_t = tgt_d[lag:]
        tgt_past = tgt_d[:-lag] if lag > 0 else tgt_d
        src_past = src_d[:-lag] if lag > 0 else src_d
        length = min(len(tgt_t), len(tgt_past), len(src_past))
        tgt_t = tgt_t[:length]
        tgt_past = tgt_past[:length]
        src_past = src_past[:length]

        # Joint and conditional entropies via counting
        def _entropy_from_counts(counts):
            p = counts / counts.sum()
            p = p[p > 0]
            return -np.sum(p * np.log2(p))

        # H(Y_t | Y_{t-1})
        joint_yy = np.zeros((n_bins, n_bins))
        for i in range(length):
            joint_yy[tgt_t[i], tgt_past[i]] += 1
        h_yy = _entropy_from_counts(joint_yy.flatten())
        h_ypast = _entropy_from_counts(joint_yy.sum(axis=0))
        h_cond_yy = h_yy - h_ypast

        # H(Y_t | Y_{t-1}, X_{t-1})
        joint_yyx = np.zeros((n_bins, n_bins, n_bins))
        for i in range(length):
            joint_yyx[tgt_t[i], tgt_past[i], src_past[i]] += 1
        h_yyx = _entropy_from_counts(joint_yyx.flatten())
        h_yx_past = _entropy_from_counts(joint_yyx.sum(axis=0).flatten())
        h_cond_yyx = h_yyx - h_yx_past

        te = h_cond_yy - h_cond_yyx
        return max(te, 0.0)

    scores = {}
    for col in X.columns:
        scores[col] = _transfer_entropy(X[col], m_minus[col], lag)

    total = sum(scores.values())
    if total == 0:
        w = {col: 1.0 / len(X.columns) for col in X.columns}
    else:
        w = {col: v / total for col, v in scores.items()}

    weights = pd.DataFrame(
        {col: [w[col]] * len(forecasts) for col in forecasts.columns},
        index=forecasts.index,
    )
    return weights


# ---------------------------------------------------------------------------
# 7. Linear Opinion Pooling (simple mean)
# ---------------------------------------------------------------------------

def compute_linear_pooling_weights(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Linear opinion pooling: equal weights 1/N for all experts.

    Parameters
    ----------
    forecasts : pd.DataFrame
        (T x N) expert forecasts.

    Returns
    -------
    weights : pd.DataFrame
        Uniform weights.
    """
    n = len(forecasts.columns)
    weights = pd.DataFrame(
        1.0 / n,
        index=forecasts.index,
        columns=forecasts.columns,
    )
    return weights


# ---------------------------------------------------------------------------
# 8. Median Pooling
# ---------------------------------------------------------------------------

def compute_median_pooling(forecasts: pd.DataFrame) -> pd.Series:
    """
    Median pooling: the pooled estimate is the cross-sectional median.

    Unlike other methods, this returns a Series directly (not weights),
    because the median is not expressible as a fixed linear combination.

    Parameters
    ----------
    forecasts : pd.DataFrame
        (T x N) expert forecasts.

    Returns
    -------
    pooled : pd.Series
        Median forecast at each time period.
    """
    return forecasts.astype(float).median(axis=1)


# ---------------------------------------------------------------------------
# Shared: pooled forecast from weights
# ---------------------------------------------------------------------------

def pooled_forecast(
    forecasts: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    """
    Compute the supervisor's pooled estimate: S_t = sum_i w_i^t * x_i^t.

    At time steps where no pioneer is detected (all weights are NaN or sum
    to zero), the pooled estimate falls back to the simple cross-sectional
    mean.  This fallback corresponds to the initialisation rule w_i^0 = 1/m
    described in the paper.

    Parameters
    ----------
    forecasts : pd.DataFrame
        (T x N) expert forecasts.
    weights : pd.DataFrame
        (T x N) weights produced by any of the weight-computation functions.

    Returns
    -------
    pooled : pd.Series
        Length-T pooled estimate.

    Examples
    --------
    >>> w = compute_pioneer_weights_angles(forecasts)
    >>> pooled = pooled_forecast(forecasts, w)
    """
    forecasts = forecasts.astype(float)
    weights = weights.astype(float)

    weighted_sum = (forecasts * weights).sum(axis=1, min_count=1)

    weight_sums = weights.sum(axis=1, min_count=1)
    no_pioneer = weight_sums.isna() | (weight_sums == 0)

    fallback_mean = forecasts.mean(axis=1)

    pooled = weighted_sum.copy()
    pooled[no_pioneer] = fallback_mean[no_pioneer]
    return pooled


# Backward-compatible alias
pooled_forecast_simple = pooled_forecast