# %% CODE 0 - PDM

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

# %% CODE 1

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECB Inflation Panel — HICP (ECB) with ADF, Granger, and VAR
===========================================================

Overview
--------
Single-file, reproducible script that builds a monthly inflation panel from the
ECB Data Portal (SDMX 2.1 REST) — HICP inflation (y/y, %) for multiple countries.

It then runs a compact time-series workflow suitable for teaching and quick diagnostics:
- ADF unit-root tests on inflation levels
- Bivariate Granger causality screening (predictors → target)
- Small VAR in levels with lag order selected by BIC

Key features
------------
- Uses official ECB SDMX API (no scraping)
- Explicit SDMX keys and dimensions documented in code
- Month indexing standardized to month-start timestamps for safe merges

Data source
-----------
ECB:  ECB Data Portal, dataset "ICP" (HICP).
      SDMX 2.1 REST pattern:
      https://data-api.ecb.europa.eu/service/data/ICP/{key}?format=csvdata&startPeriod=...&endPeriod=...

Econometric workflow
--------------------
- ADF test (H0: unit root) on each inflation series (levels)
- Granger causality tests (bivariate): does X help predict the target series?
  Ranking uses the minimum p-value across lags 1..maxlag
- VAR: target + top 2 Granger predictors; lag order chosen by BIC; VAR in levels

Outputs
-------
- Multi-line plot of the panel (incl. 0-line)
- Console tables:
  * ADF stats/p-values
  * Granger ranking
  * VAR lag selection (BIC) and estimation summary

Dependencies
------------
requests, pandas, numpy, matplotlib, statsmodels

Author / License
----------------
Eric Vansteenberghe (Banque de France)
Adapted: 2026-03-16 by Amin Tarbouch
License: MIT (recommended for teaching code)
"""

import requests
import pandas as pd
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR


# ============================================================
# 1. Fetch ECB HICP inflation panel
# ============================================================

def fetch_ecb_hicp_inflation_panel(
    countries,
    start="1997-01-01",
    end=None,
    item="000000",   # headline all-items HICP
    sa="N",          # neither seasonally nor working-day adjusted
    measure="4",     # percentage change (as used in ICP keys)
    variation="ANR", # annual rate of change
    freq="M",
    timeout=60
):
    """
    Fetch a monthly cross-country panel of HICP inflation (annual rate of change)
    from the ECB Data Portal (ICP dataflow).

    Returns
    -------
    panel_wide : pd.DataFrame
        Index: pandas datetime (monthly)
        Columns: country codes (e.g., DE, FR, IT)
        Values: inflation rate (float)
    raw_long : pd.DataFrame
        Long format with series dimensions, TIME_PERIOD and OBS_VALUE.
    """
    base = "https://data-api.ecb.europa.eu/service/data"
    key = f"{freq}.{'+'.join(countries)}.{sa}.{item}.{measure}.{variation}"

    params = {"format": "csvdata", "startPeriod": start}
    if end is not None:
        params["endPeriod"] = end

    url = f"{base}/ICP/{key}"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()

    raw = pd.read_csv(StringIO(r.text))

    if "TIME_PERIOD" not in raw.columns or "OBS_VALUE" not in raw.columns:
        raise ValueError(f"Unexpected response format. Columns: {list(raw.columns)}")

    country_col = "REF_AREA" if "REF_AREA" in raw.columns else None
    if country_col is None:
        for cand in ["GEO", "LOCATION", "COUNTRY", "REF_AREA"]:
            if cand in raw.columns:
                country_col = cand
                break
    if country_col is None:
        standard = {"TIME_PERIOD", "OBS_VALUE", "OBS_STATUS", "OBS_CONF", "UNIT_MULT", "DECIMALS"}
        nonstandard = [c for c in raw.columns if c not in standard]
        if not nonstandard:
            raise ValueError("Could not infer the country column from the response.")
        country_col = nonstandard[0]

    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"])
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")

    panel = (
        raw.pivot_table(index="TIME_PERIOD", columns=country_col, values="OBS_VALUE", aggfunc="last")
        .sort_index()
    )

    return panel, raw


# ============================================================
# 2. Example usage — EU-11 panel
# ============================================================

countries = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]
infl_panel, infl_long = fetch_ecb_hicp_inflation_panel(
    countries=countries,
    start="2000-01",
    end="2025-12"
)

# Ensure month-start index
infl_panel.index = pd.to_datetime(infl_panel.index).to_period("M").to_timestamp(how="start")


# ============================================================
# 3. Plot the inflation panel
# ============================================================

plt.figure(figsize=(12, 6))
for country in infl_panel.columns:
    plt.plot(infl_panel.index, infl_panel[country], label=country, linewidth=1)

plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.xlabel("Time")
plt.ylabel("Inflation rate (y/y, %)")
plt.title("HICP Inflation Panel (ECB Data Portal)")
plt.legend(ncol=3, fontsize=9, frameon=False)
plt.tight_layout()
plt.show()


# ============================================================
# 4. Prepare data
# ============================================================

df = infl_panel.copy().sort_index().dropna()


# ============================================================
# 5. ADF unit-root tests
# ============================================================

print("\n=== ADF unit-root tests (levels) ===")
adf_results = []
for c in df.columns:
    stat, pval, _, _, _, _ = adfuller(df[c], autolag="AIC")
    adf_results.append({"country": c, "ADF_stat": stat, "pvalue": pval})

adf_table = pd.DataFrame(adf_results).sort_values("pvalue")
print(adf_table.to_string(index=False))


# ============================================================
# 6. Granger causality: X → FR
# ============================================================

target = "FR"
maxlag = 6

print(f"\n=== Granger causality tests: X → {target} ===")
granger_out = []

for c in df.columns:
    if c == target:
        continue

    data_gc = df[[target, c]]

    try:
        res = grangercausalitytests(data_gc, maxlag=maxlag, verbose=False)
        min_p = min(res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1))
        granger_out.append({"country": c, "min_pvalue": min_p})
    except Exception as e:
        print(f"Granger test failed for {c}: {e}")

granger_rank = (
    pd.DataFrame(granger_out)
    .sort_values("min_pvalue")
    .reset_index(drop=True)
)

print(f"\n=== Ranking of countries by Granger causality for {target} ===")
print(granger_rank.to_string(index=False))


# ============================================================
# 7. Simple VAR with BIC (FR + top 2 predictors)
# ============================================================

top_countries = granger_rank["country"].iloc[:2].tolist()
var_vars = [target] + top_countries

print("\nVAR variables:", var_vars)

X_var = df[var_vars]
model = VAR(X_var)
lag_selection = model.select_order(maxlags=6)
p = lag_selection.selected_orders["bic"]
p = max(1, p)

print("\n=== VAR lag selection (BIC) ===")
print(lag_selection.summary())
print(f"Selected lag order p = {p}")

var_res = model.fit(p)
print("\n=== VAR estimation results ===")
print(var_res.summary())


# %% CODE FINAL

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Part A - PDM on European inflation dynamics
EU-11 version only (Ukraine excluded because of SSL fetch failure)

This script:
1. Loads infl_panel from ecb_hicp_panel_var_granger.py
2. Keeps only the 11 EU countries:
   DE, FR, IT, ES, NL, BE, AT, PT, IE, FI, GR
3. Restricts to complete-case sample with .dropna()
4. Computes PDM pioneer weights using compute_pioneer_weights_angles
5. Plots:
   - one line per country over time
   - one heatmap
6. Builds average pioneer weights by subperiod
7. Ranks countries in each subperiod
8. Saves all outputs to an "outputs_partA_no_ukraine" folder
"""

from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdm import compute_pioneer_weights_angles


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = Path("outputs_partA_no_ukraine")
OUTPUT_DIR.mkdir(exist_ok=True)

EU_COUNTRIES = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]

COUNTRY_NAMES = {
    "DE": "Germany",
    "FR": "France",
    "IT": "Italy",
    "ES": "Spain",
    "NL": "Netherlands",
    "BE": "Belgium",
    "AT": "Austria",
    "PT": "Portugal",
    "IE": "Ireland",
    "FI": "Finland",
    "GR": "Greece",
}

PERIODS = {
    "I (2002-2007)": ("2002-01", "2007-12"),
    "II (2008-2012)": ("2008-01", "2012-12"),
    "III (2013-2019)": ("2013-01", "2019-12"),
    "IV (2020-2021)": ("2020-01", "2021-12"),
    "V (2022-2023)": ("2022-01", "2023-12"),
    "VI (2024-2025)": ("2024-01", "2025-12"),
}


# ============================================================
# Loading data
# ============================================================

def load_inflation_panel() -> pd.DataFrame:
    """
    Imports ecb_hicp_panel_var_granger.py and retrieves infl_panel.

    Important:
    This assumes that importing the module succeeds on your machine.
    If your original file still tries to fetch Ukraine on import, and crashes before
    infl_panel is created, then you will need to comment out the Ukraine-fetching part
    inside ecb_hicp_panel_var_granger.py itself.

    Once infl_panel exists, this script keeps only the EU-11 columns.
    """
    module = importlib.import_module("ecb_hicp_panel_var_granger")

    if not hasattr(module, "infl_panel"):
        raise AttributeError(
            "No variable named 'infl_panel' was found in "
            "ecb_hicp_panel_var_granger.py."
        )

    infl_panel = getattr(module, "infl_panel")

    if not isinstance(infl_panel, pd.DataFrame):
        raise TypeError("'infl_panel' exists but is not a pandas DataFrame.")

    panel = infl_panel.copy()

    if not isinstance(panel.index, pd.DatetimeIndex):
        panel.index = pd.to_datetime(panel.index)

    missing_cols = [c for c in EU_COUNTRIES if c not in panel.columns]
    if missing_cols:
        raise ValueError(
            f"These required EU country columns are missing from infl_panel: {missing_cols}"
        )

    panel = panel[EU_COUNTRIES].copy()
    return panel


# ============================================================
# Analysis helpers
# ============================================================

def average_weights_by_period(weights: pd.DataFrame, periods: dict) -> pd.DataFrame:
    """
    Returns a DataFrame:
    rows = countries
    columns = subperiods
    values = mean pioneer weight in that period
    """
    out = {}
    for period_name, (start, end) in periods.items():
        sub = weights.loc[start:end]
        out[period_name] = sub.mean(axis=0)
    return pd.DataFrame(out)


def rank_weights_by_period(avg_table: pd.DataFrame) -> pd.DataFrame:
    """
    1 = highest average pioneer weight in that subperiod
    """
    return avg_table.rank(axis=0, ascending=False, method="dense").astype(int)


def build_nonzero_summary(weights: pd.DataFrame) -> pd.DataFrame:
    """
    For each country:
    - number of months with non-zero pioneer weight
    - share of months with non-zero pioneer weight
    - first and last non-zero date
    - average weight when non-zero
    - average weight full sample
    """
    rows = []

    for col in weights.columns:
        s = weights[col]
        mask = s > 0
        nonzero_dates = s.index[mask]

        rows.append(
            {
                "country": col,
                "country_name": COUNTRY_NAMES.get(col, col),
                "months_nonzero": int(mask.sum()),
                "share_nonzero": float(mask.mean()),
                "first_nonzero": nonzero_dates.min().strftime("%Y-%m") if len(nonzero_dates) else None,
                "last_nonzero": nonzero_dates.max().strftime("%Y-%m") if len(nonzero_dates) else None,
                "avg_weight_when_nonzero": float(s[mask].mean()) if mask.any() else 0.0,
                "avg_weight_full_sample": float(s.mean()),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["months_nonzero", "avg_weight_full_sample"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return out


# ============================================================
# Plotting
# ============================================================

def plot_line_chart(weights: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(14, 7))

    for col in weights.columns:
        plt.plot(weights.index, weights[col], linewidth=1.4, label=col)

    plt.title("PDM pioneer weights over time (angles) - EU-11 only")
    plt.xlabel("Date")
    plt.ylabel("Pioneer weight")
    plt.legend(ncol=4, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(weights: pd.DataFrame, output_path: Path) -> None:
    data = weights.T.values

    fig, ax = plt.subplots(figsize=(15, 6))
    im = ax.imshow(data, aspect="auto", interpolation="nearest")

    ax.set_title("PDM pioneer weights heatmap (angles) - EU-11 only")
    ax.set_xlabel("Date")
    ax.set_ylabel("Country")
    ax.set_yticks(np.arange(len(weights.columns)))
    ax.set_yticklabels(weights.columns)

    n_ticks = min(12, len(weights.index))
    tick_positions = np.linspace(0, len(weights.index) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [weights.index[i].strftime("%Y-%m") for i in tick_positions],
        rotation=45,
        ha="right"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pioneer weight")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Text discussion templates
# ============================================================

def print_a1d_comment(weights: pd.DataFrame) -> None:
    """
    A short template for A.1(d)
    """
    stable = weights.loc["2000-01":"2007-12"]
    max_by_month = stable.max(axis=1)

    print("\n" + "=" * 80)
    print("A.1(d) DISCUSSION")
    print("=" * 80)
    print(
        "During 2000-2007, inflation was relatively low and stable. "
        "In the PDM framework, strong pioneers are less likely in such periods if countries move "
        "more synchronously and there is less directional divergence followed by convergence."
    )
    print()
    print(
        f"Average monthly maximum pioneer weight in 2000-2007: {max_by_month.mean():.6f}"
    )
    print(
        f"Maximum observed monthly pioneer weight in 2000-2007: {max_by_month.max():.6f}"
    )
    print()
    print(
        "Interpretation: if these values are lower than in crisis periods, this suggests that "
        "clear pioneers were less common during the Great Moderation. That is consistent with "
        "PDM theory: when panel members evolve in a more stable and homogeneous way, fewer countries "
        "stand out as early movers toward which the rest subsequently converges."
    )


def print_a2c_comment(avg_table: pd.DataFrame) -> None:
    """
    A short template for A.2(c)
    """
    print("\n" + "=" * 80)
    print("A.2(c) ECONOMIC INTERPRETATION")
    print("=" * 80)
    print(
        "Possible explanations for countries with high pioneer weights include differences in:"
    )
    print("- energy mix and exposure to imported energy shocks")
    print("- trade openness and speed of pass-through from foreign prices")
    print("- financial structure and credit transmission")
    print("- geographic position and logistics exposure")
    print("- sectoral composition, such as manufacturing, tourism, or shipping")
    print()
    print("Top 3 countries by subperiod:")
    for period in avg_table.columns:
        leaders = avg_table[period].sort_values(ascending=False).head(3)
        summary = ", ".join([f"{idx} ({val:.6f})" for idx, val in leaders.items()])
        print(f"{period}: {summary}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Loading inflation panel...")
    panel = load_inflation_panel()

    print("Restricting to complete-case sample with dropna()...")
    panel = panel.dropna().copy()

    print(f"Final sample shape: {panel.shape}")
    print(f"Start date: {panel.index.min().strftime('%Y-%m')}")
    print(f"End date:   {panel.index.max().strftime('%Y-%m')}")
    print(f"Countries:  {list(panel.columns)}")

    print("\nComputing angle-based pioneer weights...")
    w_angles = compute_pioneer_weights_angles(panel)

    if not isinstance(w_angles, pd.DataFrame):
        raise TypeError("compute_pioneer_weights_angles did not return a DataFrame.")

    if not isinstance(w_angles.index, pd.DatetimeIndex):
        w_angles.index = panel.index

    if list(w_angles.columns) != list(panel.columns):
        w_angles.columns = panel.columns

    # Save raw weights
    weights_csv = OUTPUT_DIR / "partA_pdm_angles_weights_eu11.csv"
    w_angles.to_csv(weights_csv)

    # Save plots
    print("Saving plots...")
    plot_line_chart(w_angles, OUTPUT_DIR / "partA_pioneer_weights_lines_eu11.png")
    plot_heatmap(w_angles, OUTPUT_DIR / "partA_pioneer_weights_heatmap_eu11.png")

    # Non-zero summary
    print("Building non-zero weight summary...")
    nonzero_summary = build_nonzero_summary(w_angles)
    nonzero_summary.to_csv(
        OUTPUT_DIR / "partA_nonzero_weight_summary_eu11.csv",
        index=False
    )

    print("\nA.1(c) Countries receiving non-zero pioneer weight:")
    print(
        nonzero_summary[
            ["country", "months_nonzero", "share_nonzero", "first_nonzero", "last_nonzero"]
        ].to_string(index=False)
    )

    # Averages by subperiod
    print("\nComputing average weights by subperiod...")
    avg_table = average_weights_by_period(w_angles, PERIODS)
    avg_table.to_csv(OUTPUT_DIR / "partA_average_weights_by_subperiod_eu11.csv")

    print("\nA.2(a) Average pioneer weights by subperiod:")
    print(avg_table.round(6).to_string())

    # Rankings
    ranking_table = rank_weights_by_period(avg_table)
    ranking_table.to_csv(OUTPUT_DIR / "partA_rankings_by_subperiod_eu11.csv")

    print("\nA.2(b) Rankings by subperiod (1 = highest average pioneer weight):")
    print(ranking_table.to_string())

    # Save top rankings as text
    with open(OUTPUT_DIR / "partA_top_rankings_eu11.txt", "w", encoding="utf-8") as f:
        for period in avg_table.columns:
            f.write(f"{period}\n")
            f.write(avg_table[period].sort_values(ascending=False).to_string())
            f.write("\n\n")

    # Print discussion templates
    print_a1d_comment(w_angles)
    print_a2c_comment(avg_table)

    print("\nDone.")
    print(f"All outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

# %%
