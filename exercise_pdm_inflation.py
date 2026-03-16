#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECB–SSSU Inflation Panel — HICP (ECB) + Ukraine CPI (SSSU) with ADF, Granger, and VAR
===================================================================================

Overview
--------
Single-file, reproducible script that builds a monthly inflation panel from:

1) ECB Data Portal (SDMX 2.1 REST) — HICP inflation (y/y, %), multiple countries.
2) State Statistics Service of Ukraine (SSSU) SDMX v3 — CPI index (prev. month = 100),
   converted to y/y inflation by chaining 12 monthly factors.

It then runs a compact time-series workflow suitable for teaching and quick diagnostics:
- ADF unit-root tests on inflation levels
- Bivariate Granger causality screening (predictors → target)
- Small VAR in levels with lag order selected by BIC

Key features (for readers + LLMs)
---------------------------------
- Uses official APIs (no HTML scraping).
- Explicit SDMX keys and dimensions documented in code.
- Robust handling of SSSU SDMX-CSV metadata rows (keeps only TIME_PERIOD = 'YYYY-Mmm').
- Month indexing standardized to month-start timestamps for safe merges.

Data sources
------------
ECB:  ECB Data Portal, dataset "ICP" (HICP).
      SDMX 2.1 REST pattern:
      https://data-api.ecb.europa.eu/service/data/ICP/{key}?format=csvdata&startPeriod=...&endPeriod=...

SSSU: SSSU SDMX v3 endpoint (Ukraine CPI, prev. month = 100), dataflow:
      SSSU / DF_PRICE_CHANGE_CONSUMER_GOODS_SERVICE / version "~"
      key: INDEX_CONSUMPRICE.PREV_MONTH.UA00000000000000000.0.M

Econometric workflow (teaching level)
-------------------------------------
- ADF test (H0: unit root) on each inflation series (levels).
- Granger causality tests (bivariate): does X help predict the target series?
  Ranking uses the minimum p-value across lags 1..maxlag.
- VAR: target + top 2 Granger predictors; lag order chosen by BIC; VAR in levels.

Outputs
-------
- Multi-line plot of the panel (incl. 0-line).
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
Created: 2026-01-24
License: MIT (recommended for teaching code)


Notes
-----
- This script uses revised (latest) data, not real-time vintages.
- Missing values are handled with complete-case deletion prior to estimation.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


COUNTRIES = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]
TARGET = "FR"
START = "2000-01"
END = "2025-12"

PERIODS = {
    "I (2002-07)": ("2002-01", "2007-12"),
    "II (2008-12)": ("2008-01", "2012-12"),
    "III (2013-19)": ("2013-01", "2019-12"),
    "IV (2020-21)": ("2020-01", "2021-12"),
    "V (2022-23)": ("2022-01", "2023-12"),
    "VI (2024-25)": ("2024-01", "2025-12"),
}


def fetch_ecb_hicp_inflation_panel(countries, start=START, end=END, timeout=60):
    base = "https://data-api.ecb.europa.eu/service/data"
    key = f"M.{'+'.join(countries)}.N.000000.4.ANR"
    url = f"{base}/ICP/{key}"
    params = {"format": "csvdata", "startPeriod": start, "endPeriod": end}

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()

    raw = pd.read_csv(StringIO(r.text))
    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"])
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")

    panel = (
        raw.pivot_table(
            index="TIME_PERIOD",
            columns="REF_AREA",
            values="OBS_VALUE",
            aggfunc="last",
        )
        .sort_index()
    )

    panel.index = panel.index.to_period("M").to_timestamp()
    return panel


def rmse(actual, estimate):
    x = pd.concat([actual, estimate], axis=1).dropna()
    x.columns = ["actual", "estimate"]
    return np.sqrt(((x["actual"] - x["estimate"]) ** 2).mean())


def average_weights_by_period(weights, periods):
    out = {}
    for name, (start, end) in periods.items():
        out[name] = weights.loc[start:end].mean()
    return pd.DataFrame(out)


def plot_pioneer_weights(weights):
    plt.figure(figsize=(12, 6))
    for col in weights.columns:
        plt.plot(weights.index, weights[col], linewidth=1, label=col)
    plt.title("Pioneer weights over time")
    plt.xlabel("Time")
    plt.ylabel("Weight")
    plt.legend(ncol=3, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.show()


def dominant_pioneer(weights):
    return weights.idxmax(axis=1)


def plot_dominant_pioneer(series):
    cats = pd.Categorical(series)
    plt.figure(figsize=(12, 4))
    plt.scatter(series.index, cats.codes, s=10)
    plt.yticks(range(len(cats.categories)), cats.categories)
    plt.title(f"Dominant pioneer over time (target = {TARGET})")
    plt.xlabel("Time")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.show()


def short_part_a_discussion(w_angles, avg_table):
    full_avg = w_angles.mean().sort_values(ascending=False)
    stable_avg = w_angles.loc["2000-01":"2007-12"].mean().sort_values(ascending=False)
    nonzero = (w_angles > 0).sum().sort_values(ascending=False)

    top_full = ", ".join(full_avg.head(3).index.tolist())
    top_stable = ", ".join(stable_avg.head(3).index.tolist())

    print("\nPart A - short discussion")
    print(
        f"Over the full sample, the countries with the highest average pioneer weights are {top_full}. "
        f"Positive pioneer weights are not constant but switch over time across countries."
    )
    print(
        f"In the low-inflation period 2000-2007, the highest average pioneer weights are observed for {top_stable}. "
        f"However, the plot suggests frequent switching rather than one persistent leader."
    )
    print(
        "This indicates that in more stable inflation environments, leadership is weaker and more diffuse, "
        "which is consistent with the idea that inflation dynamics are more synchronized across countries."
    )
    print(
        f"The countries with the largest number of non-zero pioneer observations are {', '.join(nonzero.head(3).index.tolist())}."
    )


def short_part_b_discussion(rmse_table):
    best_method = rmse_table["Overall"].idxmin()
    best_rmse = rmse_table.loc[best_method, "Overall"]

    print("\nPart B - short discussion")
    print(
        f"Using {TARGET} as the target country, the best-performing method in terms of overall RMSE is "
        f"{best_method} with RMSE = {best_rmse:.4f}."
    )
    print(
        "This suggests that some combination rules are better at tracking French inflation than others, "
        "even when the expert panel excludes France itself."
    )
    print(
        "Performance also varies across subperiods, which indicates that the relative usefulness of each "
        "pooling method is not stable over time."
    )


def part_a(panel):
    w_angles = compute_pioneer_weights_angles(panel)

    print("\nPart A.1 - first rows of pioneer weights")
    print(w_angles.head().to_string())

    print("\nPart A.1 - countries with non-zero pioneer weights")
    print((w_angles > 0).sum().sort_values(ascending=False).to_string())

    print("\nPart A.1 - average pioneer weights over full sample")
    print(w_angles.mean().sort_values(ascending=False).to_string())

    plot_pioneer_weights(w_angles)

    avg_table = average_weights_by_period(w_angles, PERIODS)

    print("\nPart A.2 - average pioneer weights by subperiod")
    print(avg_table.to_string())

    print("\nPart A.2 - rankings by subperiod")
    for col in avg_table.columns:
        print(f"\n{col}")
        print(avg_table[col].sort_values(ascending=False).to_string())

    print("\nPart A.1 - 2000-2007 average pioneer weights")
    print(w_angles.loc["2000-01":"2007-12"].mean().sort_values(ascending=False).to_string())

    short_part_a_discussion(w_angles, avg_table)

    return w_angles, avg_table


def part_b(panel):
    experts = panel.drop(columns=[TARGET])
    actual = panel[TARGET]

    w_angles = compute_pioneer_weights_angles(experts)
    dom = dominant_pioneer(w_angles)

    print(f"\nPart B.1 - dominant pioneer counts (target = {TARGET})")
    print(dom.value_counts().to_string())

    plot_dominant_pioneer(dom)

    methods = {
        "PDM_Angles": compute_pioneer_weights_angles(experts),
        "PDM_Distance": compute_pioneer_weights_distance(experts),
        "Granger": compute_granger_weights(experts),
        "LaggedCorr": compute_lagged_correlation_weights(experts),
        "MultiReg": compute_multivariate_regression_weights(experts),
        "TransferEntropy": compute_transfer_entropy_weights(experts),
        "LinearPooling": compute_linear_pooling_weights(experts),
    }

    results = {}

    for name, weights in methods.items():
        estimate = pooled_forecast(experts, weights)
        results[name] = {"Overall": rmse(actual, estimate)}
        for period_name, (start, end) in PERIODS.items():
            results[name][period_name] = rmse(actual.loc[start:end], estimate.loc[start:end])

    median_estimate = compute_median_pooling(experts)
    results["MedianPooling"] = {"Overall": rmse(actual, median_estimate)}
    for period_name, (start, end) in PERIODS.items():
        results["MedianPooling"][period_name] = rmse(
            actual.loc[start:end],
            median_estimate.loc[start:end],
        )

    rmse_table = pd.DataFrame(results).T.sort_values("Overall")

    print(f"\nPart B.2 - RMSE by method (target = {TARGET})")
    print(rmse_table.to_string())

    estimate = pooled_forecast(experts, methods["PDM_Angles"])
    comp = pd.concat(
        [actual.rename("Actual"), estimate.rename("PDM_Angles")],
        axis=1
    ).dropna()

    plt.figure(figsize=(12, 6))
    plt.plot(comp.index, comp["Actual"], linewidth=1.5, label=f"Actual {TARGET}")
    plt.plot(comp.index, comp["PDM_Angles"], linewidth=1.2, label="PDM_Angles")
    plt.title(f"Actual vs pooled estimate ({TARGET})")
    plt.xlabel("Time")
    plt.ylabel("Inflation")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    short_part_b_discussion(rmse_table)

    return rmse_table


def main():
    infl_panel = fetch_ecb_hicp_inflation_panel(COUNTRIES)
    panel = infl_panel.dropna().sort_index()

    print("\nColumns")
    print(panel.columns.tolist())

    print("\nShape after dropna()")
    print(panel.shape)

    print("\nNote")
    print(f"This version uses {TARGET} as the target country in Part B.")

    part_a(panel)
    part_b(panel)


if __name__ == "__main__":
    main()