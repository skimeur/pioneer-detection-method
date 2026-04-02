#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
exercise_pdm_inflation.py

DETAILED OVERVIEW

This script solves the PDM inflation exercise in a single file.

The original assignment combines:
- a panel of ECB HICP inflation series for several European countries
- a target-tracking exercise using Ukraine

Because the Ukraine data source is not working here, France is used as the
target instead.

The script is split into two parts.

PART A
Goal:
    Apply the Pioneer Detection Method (PDM) to a panel of European inflation
    series and identify which countries behave as pioneers over time.

Pipeline:
    1. Download ECB monthly year-on-year HICP inflation
    2. Build a balanced panel by removing missing values
    3. Compute dynamic PDM weights using the angle-based version
    4. Display the inflation panel
    5. Display the full PDM weights
    6. Display the top pioneers only
    7. Compute average PDM weights by subperiod
    8. Rank countries within each subperiod
    9. Display a heatmap of average weights

Interpretation:
    Countries with larger pioneer weights are interpreted as moving earlier
    than the others in the common inflation dynamics.

PART B
Goal:
    Use the same cross-country information in a target-tracking /
    pseudo-forecasting exercise.

Adaptation:
    France replaces Ukraine as the target country.

Setup:
    - Target: France
    - Experts: all other countries in the ECB panel

Pipeline:
    1. Remove France from the expert panel
    2. Compute the dominant pioneer in rolling windows
    3. Compare several methods from pdm.py
    4. Compute RMSE against actual French inflation
    5. Display the dominant pioneer over time
    6. Display RMSE by method
    7. Display the best pooled series against France
    8. Print a short discussion

Printed outputs:
    - panel shape
    - average pioneer weights by subperiod
    - ranks by subperiod
    - dominant pioneer counts
    - RMSE comparison
    - short discussion

Displayed figures:
    - inflation panel
    - PDM weights over time
    - top pioneer weights
    - heatmap of average weights by subperiod
    - dominant pioneer over time
    - full-sample RMSE by method
    - best pooled series vs France

Important:
    This script assumes that pdm.py is in the same folder and provides the
    imported functions below.
"""

from __future__ import annotations

from io import StringIO
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from pdm import (
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

# Global configuration

COUNTRIES = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]
TARGET = "FR"
START = "2000-01-01"
END = "2025-12-31"
ROLLING_WINDOW = 36

SUBPERIODS = {
    "2002-2007": ("2002-01-01", "2007-12-31"),
    "2008-2012": ("2008-01-01", "2012-12-31"),
    "2013-2019": ("2013-01-01", "2019-12-31"),
    "2020-2021": ("2020-01-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2025": ("2024-01-01", "2025-12-31"),
}

METHODS = [
    "PDM (angles)",
    "PDM (distances)",
    "Granger Causality",
    "Lagged Correlation",
    "Multivar. Regression",
    "Transfer Entropy",
    "Linear Pooling",
    "Median Pooling",
]

# Data loading

def fetch_ecb_hicp_inflation_panel(
    countries: list[str],
    start: str = START,
    end: str | None = END,
    item: str = "000000",
    sa: str = "N",
    measure: str = "4",
    variation: str = "ANR",
    freq: str = "M",
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Download monthly year-on-year HICP inflation from the ECB.

    Returns a wide DataFrame:
    - index = dates
    - columns = countries
    - values = inflation rates
    """
    base = "https://data-api.ecb.europa.eu/service/data"
    key = f"{freq}.{'+'.join(countries)}.{sa}.{item}.{measure}.{variation}"

    params = {"format": "csvdata", "startPeriod": start}
    if end is not None:
        params["endPeriod"] = end

    url = f"{base}/ICP/{key}"
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()

    raw = pd.read_csv(StringIO(response.text))

    if "TIME_PERIOD" not in raw.columns or "OBS_VALUE" not in raw.columns:
        raise ValueError(f"Unexpected ECB response format: {list(raw.columns)}")

    country_col = "REF_AREA" if "REF_AREA" in raw.columns else None
    if country_col is None:
        for candidate in ["GEO", "LOCATION", "COUNTRY"]:
            if candidate in raw.columns:
                country_col = candidate
                break
    if country_col is None:
        raise ValueError("Country column not found in ECB response.")

    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"]).dt.to_period("M").dt.to_timestamp()
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")

    panel = (
        raw.pivot_table(index="TIME_PERIOD", columns=country_col, values="OBS_VALUE", aggfunc="last")
        .sort_index()
        .reindex(columns=countries)
    )

    return panel

# Utility functions

def rmse(yhat: pd.Series, y: pd.Series) -> float:
    """Compute RMSE after aligning both series."""
    z = pd.concat([yhat.rename("yhat"), y.rename("y")], axis=1).dropna()
    if z.empty:
        return np.nan
    return float(np.sqrt(np.mean((z["yhat"] - z["y"]) ** 2)))


def average_weights_by_subperiod(
    weights: pd.DataFrame,
    subperiods: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Average dynamic weights over each predefined subperiod."""
    out = {}
    for label, (start, end) in subperiods.items():
        out[label] = weights.loc[start:end].mean(axis=0, skipna=True)
    return pd.DataFrame(out)


def rank_weights(avg_weights: pd.DataFrame) -> pd.DataFrame:
    """Rank countries by average weight within each subperiod."""
    return avg_weights.rank(ascending=False, method="dense").astype("Int64")


def encode_categories(series: pd.Series) -> tuple[pd.Series, dict[int, str]]:
    """Convert country labels to integers for plotting."""
    labels = pd.Index(sorted(x for x in series.dropna().unique()))
    mapping = {i: label for i, label in enumerate(labels)}
    reverse = {label: i for i, label in mapping.items()}
    return series.map(reverse), mapping


def top_n_series(weights: pd.DataFrame, n: int = 4) -> list[str]:
    """Select countries with the highest average weights."""
    return weights.mean().sort_values(ascending=False).head(n).index.tolist()


def pooled_series_for_method(panel: pd.DataFrame, method: str) -> pd.Series:
    """Build the pooled series implied by one method."""
    if method == "PDM (angles)":
        w = compute_pioneer_weights_angles(panel)
        return pooled_forecast(panel, w)

    if method == "PDM (distances)":
        w = compute_pioneer_weights_distance(panel)
        return pooled_forecast(panel, w)

    if method == "Granger Causality":
        w = compute_granger_weights(panel, maxlag=1)
        return pooled_forecast(panel, w)

    if method == "Lagged Correlation":
        w = compute_lagged_correlation_weights(panel, lag=1)
        return pooled_forecast(panel, w)

    if method == "Multivar. Regression":
        w = compute_multivariate_regression_weights(panel, lag=1)
        return pooled_forecast(panel, w)

    if method == "Transfer Entropy":
        w = compute_transfer_entropy_weights(panel, n_bins=3, lag=1)
        return pooled_forecast(panel, w)

    if method == "Linear Pooling":
        w = compute_linear_pooling_weights(panel)
        return pooled_forecast(panel, w)

    if method == "Median Pooling":
        return compute_median_pooling(panel)

    raise ValueError(f"Unknown method: {method}")


def rmse_table(expert_panel: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Compare all methods by RMSE on the full sample and by subperiod."""
    rows = []

    for method in METHODS:
        try:
            forecast = pooled_series_for_method(expert_panel, method)
        except Exception as e:
            print(f"Method failed: {method} -> {e}")
            forecast = pd.Series(index=expert_panel.index, dtype=float)

        row = {"Method": method, "Full sample": rmse(forecast, target)}
        for label, (start, end) in SUBPERIODS.items():
            row[label] = rmse(forecast.loc[start:end], target.loc[start:end])
        rows.append(row)

    return pd.DataFrame(rows).set_index("Method").sort_values("Full sample")


def rolling_dominant_pioneer(expert_panel: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    """Find the dominant pioneer in each rolling window."""
    out = []

    for end_idx in range(window - 1, len(expert_panel)):
        block = expert_panel.iloc[end_idx - window + 1 : end_idx + 1]
        w = compute_pioneer_weights_angles(block).iloc[-1].dropna()

        if w.empty:
            dominant = None
            dominant_weight = np.nan
        else:
            dominant = str(w.idxmax())
            dominant_weight = float(w.max())

        out.append(
            {
                "date": block.index[-1],
                "dominant_pioneer": dominant,
                "dominant_weight": dominant_weight,
            }
        )

    return pd.DataFrame(out).set_index("date")

# Plotting functions

def plot_inflation_panel(panel: pd.DataFrame) -> None:
    """Display the full inflation panel."""
    plt.figure(figsize=(12, 6))
    for country in panel.columns:
        plt.plot(panel.index, panel[country], label=country, linewidth=1.1)
    plt.axhline(0, linestyle="--", linewidth=0.8)
    plt.title("ECB HICP Inflation Panel")
    plt.xlabel("Time")
    plt.ylabel("Year-on-year inflation (%)")
    plt.legend(ncol=4, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.show()


def plot_pioneer_weights(weights: pd.DataFrame) -> None:
    """Display the full set of PDM weights."""
    plt.figure(figsize=(12, 6))
    for country in weights.columns:
        plt.plot(weights.index, weights[country], label=country, linewidth=1.2)
    plt.title("Pioneer Weights Over Time")
    plt.xlabel("Time")
    plt.ylabel("Weight")
    plt.legend(ncol=4, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.show()


def plot_top_pioneers(weights: pd.DataFrame, n: int = 4) -> None:
    """Display only the countries with the strongest average pioneer weights."""
    leaders = top_n_series(weights, n=n)
    plt.figure(figsize=(12, 5))
    for country in leaders:
        plt.plot(weights.index, weights[country], label=country, linewidth=1.8)
    plt.title(f"Top {n} Pioneer Weights")
    plt.xlabel("Time")
    plt.ylabel("Weight")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def plot_average_weights_heatmap(avg_weights: pd.DataFrame) -> None:
    """Display a heatmap of average weights by country and subperiod."""
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(avg_weights.values, aspect="auto")
    ax.set_xticks(np.arange(len(avg_weights.columns)))
    ax.set_xticklabels(avg_weights.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(avg_weights.index)))
    ax.set_yticklabels(avg_weights.index)
    ax.set_title("Average Pioneer Weights by Subperiod")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_dominant_pioneer(dominant: pd.DataFrame) -> None:
    """Display the dominant pioneer over time."""
    coded, mapping = encode_categories(dominant["dominant_pioneer"])

    plt.figure(figsize=(12, 5))
    plt.step(dominant.index, coded, where="post")
    plt.yticks(list(mapping.keys()), list(mapping.values()))
    plt.title("Dominant Pioneer Over Time (Target = France)")
    plt.xlabel("Time")
    plt.ylabel("Dominant pioneer")
    plt.tight_layout()
    plt.show()


def plot_forecast_vs_target(expert_panel: pd.DataFrame, target: pd.Series, method: str) -> None:
    """Display one pooled series against actual French inflation."""
    forecast = pooled_series_for_method(expert_panel, method)
    z = pd.concat([target.rename("France"), forecast.rename(method)], axis=1).dropna()

    plt.figure(figsize=(12, 5))
    plt.plot(z.index, z["France"], label="France", linewidth=2)
    plt.plot(z.index, z[method], label=method, linewidth=1.6)
    plt.title(f"France vs pooled series: {method}")
    plt.xlabel("Time")
    plt.ylabel("Inflation (%)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def plot_rmse_bars(rmse_tbl: pd.DataFrame) -> None:
    """Display a bar chart of full-sample RMSE by method."""
    s = rmse_tbl["Full sample"].dropna().sort_values()
    plt.figure(figsize=(10, 5))
    plt.bar(s.index, s.values)
    plt.title("Full-sample RMSE by Method")
    plt.ylabel("RMSE")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# Discussion

def print_discussion(avg_weights: pd.DataFrame, rmse_tbl: pd.DataFrame, dominant: pd.DataFrame) -> None:
    """Print a short interpretation of the main results."""
    print("\nSHORT DISCUSSION\n")

    top_by_period = avg_weights.idxmax(axis=0)
    best_method = rmse_tbl["Full sample"].idxmin()

    print("Part A")
    print(
        "Pioneer rankings are more informative during periods of inflation dispersion "
        "or structural shocks than during calm and synchronized periods."
    )
    print(
        "If pioneer weights are weak before 2008, this is economically plausible: "
        "inflation was more stable and there was less visible cross-country leadership."
    )
    print()

    print("Top pioneer by subperiod:")
    for period, country in top_by_period.items():
        print(f"- {period}: {country}")
    print()

    print(
        "Shifts in the ranking can reflect differences in exposure to energy shocks, "
        "trade patterns, financial stress, and transmission speed across countries."
    )
    print()

    print("Part B")
    print(f"The best method in full-sample RMSE is: {best_method}.")
    print(
        "Still, this should not be interpreted as pure forecasting performance. "
        "PDM is mainly a tool for detecting pioneers and directional convergence."
    )
    print(
        "A low RMSE means the weighted panel tracks French inflation well, but that may "
        "reflect common shocks and co-movement rather than true predictive causality."
    )
    print()

    if not dominant.empty and dominant["dominant_pioneer"].dropna().any():
        counts = dominant["dominant_pioneer"].value_counts()
        print("Most frequent dominant pioneers in the rolling analysis:")
        for country, n in counts.head(5).items():
            print(f"- {country}: {n} windows")
    else:
        print("No stable dominant pioneer was identified in the rolling analysis.")

# Main pipeline

def main() -> None:
    print("Loading ECB inflation data...")
    panel = fetch_ecb_hicp_inflation_panel(COUNTRIES, START, END).dropna()

    print("\nBalanced panel shape:", panel.shape)
    print(panel.tail())

    print("\nPART A — Full ECB panel")

    plot_inflation_panel(panel)

    w_angles = compute_pioneer_weights_angles(panel)
    avg_weights = average_weights_by_subperiod(w_angles, SUBPERIODS)
    ranks = rank_weights(avg_weights)

    print("\nAverage pioneer weights by subperiod:")
    print(avg_weights.round(4))

    print("\nRanks by subperiod (1 = highest average weight):")
    print(ranks)

    plot_pioneer_weights(w_angles)
    plot_top_pioneers(w_angles, n=4)
    plot_average_weights_heatmap(avg_weights)

    print("\nPART B — France as target")

    target = panel[TARGET].rename("France")
    expert_panel = panel.drop(columns=[TARGET])

    print("\nExperts used:")
    print(list(expert_panel.columns))

    dominant = rolling_dominant_pioneer(expert_panel, ROLLING_WINDOW)

    print("\nDominant pioneer counts:")
    print(dominant["dominant_pioneer"].value_counts())

    plot_dominant_pioneer(dominant)

    print("\nRMSE comparison across methods:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rmse_tbl = rmse_table(expert_panel, target)

    print(rmse_tbl.round(4))
    plot_rmse_bars(rmse_tbl)

    best_method = rmse_tbl["Full sample"].dropna().idxmin()
    plot_forecast_vs_target(expert_panel, target, best_method)

    print_discussion(avg_weights, rmse_tbl, dominant)


if __name__ == "__main__":
    main()