#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exercise: Applying the Pioneer Detection Method to European Inflation Dynamics

This script:
1. Reuses infl_panel from ecb_hicp_panel_var_granger.py
2. Imports PDM methods from pdm.py
3. Produces:
   - Part A figures and tables
   - Part B figures and RMSE tables
4. Saves outputs to ./outputs/figures and ./outputs/tables
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# PATHS / OUTPUTS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SUBPERIODS
# =========================================================

PERIODS = {
    "I (2002-07)": ("2002-01", "2007-12"),
    "II (2008-12)": ("2008-01", "2012-12"),
    "III (2013-19)": ("2013-01", "2019-12"),
    "IV (2020-21)": ("2020-01", "2021-12"),
    "V (2022-23)": ("2022-01", "2023-12"),
    "VI (2024-25)": ("2024-01", "2025-12"),
}

UA_COL = "UA"


# =========================================================
# UTILITIES
# =========================================================

def import_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def save_table(df: pd.DataFrame | pd.Series, name: str) -> None:
    path_csv = TAB_DIR / f"{name}.csv"
    if isinstance(df, pd.Series):
        df.to_frame().to_csv(path_csv, index=True)
    else:
        df.to_csv(path_csv, index=True)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    z = pd.concat([y_true.rename("true"), y_pred.rename("pred")], axis=1).dropna()
    if z.empty:
        return np.nan
    return float(np.sqrt(((z["pred"] - z["true"]) ** 2).mean()))


def average_weights_by_period(weights: pd.DataFrame, periods: dict) -> pd.DataFrame:
    out = {}
    for name, (start, end) in periods.items():
        sub = weights.loc[start:end]
        out[name] = sub.mean()
    return pd.DataFrame(out)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def normalize_if_needed(weights: pd.DataFrame) -> pd.DataFrame:
    w = weights.copy()
    row_sums = w.sum(axis=1)
    mask = row_sums.notna() & (row_sums != 0)
    w.loc[mask] = w.loc[mask].div(row_sums[mask], axis=0)
    return w


# =========================================================
# LOAD MODULES
# =========================================================

print("\nLoading pdm.py ...")
pdm = import_module_from_file("pdm", BASE_DIR / "pdm.py")

print("Loading ecb_hicp_panel_var_granger.py ...")
print("Note: importing this file also runs its top-level diagnostics and plotting.")
ecb_mod = import_module_from_file("ecb_hicp_panel_var_granger", BASE_DIR / "ecb_hicp_panel_var_granger.py")

if not hasattr(ecb_mod, "infl_panel"):
    raise AttributeError("infl_panel was not found in ecb_hicp_panel_var_granger.py")

infl_panel = ecb_mod.infl_panel.copy()
infl_panel = ensure_datetime_index(infl_panel)

print("\nInflation panel loaded.")
print("Columns:", list(infl_panel.columns))
print("Sample:", infl_panel.index.min(), "to", infl_panel.index.max())


# =========================================================
# PREPARE PANEL
# =========================================================

panel = infl_panel.dropna().copy()

if UA_COL not in panel.columns:
    raise KeyError("Column 'UA' not found in infl_panel")

eu_cols = [c for c in panel.columns if c != UA_COL]
eu_panel = panel[eu_cols].copy()
actual_ua = panel[UA_COL].copy()


# =========================================================
# PART A
# =========================================================

print("\n" + "=" * 70)
print("PART A — WHO PIONEERED EUROPEAN INFLATION DYNAMICS?")
print("=" * 70)

w_angles_full = pdm.compute_pioneer_weights_angles(panel)
w_angles_full = ensure_datetime_index(w_angles_full)

# Figure A1: line plot of weights
plt.figure(figsize=(14, 7))
for c in w_angles_full.columns:
    plt.plot(w_angles_full.index, w_angles_full[c], linewidth=1.2, label=c)
plt.title("Part A — PDM (angles) pioneer weights over time")
plt.xlabel("Time")
plt.ylabel("Pioneer weight")
plt.legend(ncol=3, fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "partA_pioneer_weights_lines.png", dpi=220)
plt.close()

# Figure A1 alternative: heatmap
plt.figure(figsize=(14, 7))
plt.imshow(w_angles_full.T.fillna(0.0), aspect="auto", interpolation="nearest")
plt.yticks(range(len(w_angles_full.columns)), w_angles_full.columns)
xticks = np.linspace(0, len(w_angles_full.index) - 1, 8, dtype=int)
plt.xticks(xticks, [w_angles_full.index[i].strftime("%Y-%m") for i in xticks], rotation=45)
plt.colorbar(label="Weight")
plt.title("Part A — PDM (angles) pioneer weights heatmap")
plt.tight_layout()
plt.savefig(FIG_DIR / "partA_pioneer_weights_heatmap.png", dpi=220)
plt.close()

# Which countries had non-zero pionner weight and when: 

nonzero_events = (
    w_angles_full
    .stack()
    .reset_index()
)

nonzero_events.columns = ["date", "country", "weight"]

# only strictly possitive events
nonzero_events = nonzero_events[nonzero_events["weight"] > 0].copy()
nonzero_events["date"] = pd.to_datetime(nonzero_events["date"]).dt.strftime("%Y-%m-%d")

# group by country and save dates
dates_by_country = (
    nonzero_events
    .sort_values(["country", "date"])
    .groupby("country")["date"]
    .apply(list)
    .reset_index(name="dates_with_nonzero_pioneer_weight")
)

print(dates_by_country)

'''
Discussion:
During the 2000-2007 period, while there are non-zero pioneer weights, 
they are quite sparse and non persistent, particularlly when compared to 
later periods of crisis like 2020-2023 seem to display a more frequent 
and concentrated pioneer-weight spikes. This is consistent with PDM, 
meaning that there are fewer episodes of early divergence followed by
convergence of the rest of the panel.

'''

# Table A2: average weights by subperiod
avg_weights = average_weights_by_period(w_angles_full, PERIODS)
save_table(avg_weights, "partA_average_weights_by_subperiod")

print("\nAverage pioneer weights by subperiod:")
print(avg_weights.round(4))

# Rankings by period
rankings = pd.DataFrame({
    col: avg_weights[col].sort_values(ascending=False).index
    for col in avg_weights.columns
})
save_table(rankings, "partA_rankings_by_subperiod")

print("\nTop countries by subperiod:")
for col in avg_weights.columns:
    print(f"\n{col}")
    print(avg_weights[col].sort_values(ascending=False).round(4))

# Notice that the ranking does change in each period

# A2.C
'''
The rankings do change substantially across subperiods, withouth a single country
acting as a permanent inflation pioneer. Instead, pioneership seems to depend on 
the particular macroeconomic shock that's affecting Europe at the time. Thus, 
countries tend to move earlier due to their differences in energy exposure, trade
openess, sectoral structure, financial vulnerability or geographic position. This 
is more visible in crisis periods where pioneer weights not only change across 
countries, but also seem to become more concentrated sometimes, like the case of 
Belgium in 2022-2023. This suggests that pioneership is more shock dependent rather
than structural.
'''

# Non-zero share
nonzero_share = (w_angles_full.fillna(0) > 0).mean().sort_values(ascending=False)
save_table(nonzero_share, "partA_nonzero_weight_share")

print("\nShare of periods with non-zero pioneer weight:")
print(nonzero_share.round(4))


# =========================================================
# PART B.1 — ROLLING DOMINANT PIONEER RELATIVE TO UKRAINE
# =========================================================

print("\n" + "=" * 70)
print("PART B.1 — ROLLING PIONEER DETECTION RELATIVE TO UKRAINE")
print("=" * 70)

def pairwise_pioneer_score_vs_ua(country_series: pd.Series, ua_series: pd.Series) -> pd.Series:
    """
    Pairwise PDM(country, UA). We keep the country's last-window pioneer weight.
    This is a pragmatic way to interpret 'relative to Ukraine' using the available pdm.py API.
    """
    pair = pd.concat([country_series.rename("country"), ua_series.rename("UA")], axis=1).dropna()
    w = pdm.compute_pioneer_weights_angles(pair)
    # Country's weight in each date
    return w["country"]

def rolling_dominant_pioneer_vs_ua(panel_df: pd.DataFrame, window: int = 24) -> pd.Series:
    winners = {}
    for end in range(window, len(panel_df) + 1):
        sub = panel_df.iloc[end - window:end]
        scores = {}
        for c in eu_cols:
            w_pair = pairwise_pioneer_score_vs_ua(sub[c], sub[UA_COL])
            last_score = w_pair.iloc[-1]
            scores[c] = 0.0 if pd.isna(last_score) else float(last_score)

        if all(v == 0.0 for v in scores.values()):
            winners[sub.index[-1]] = np.nan
        else:
            winners[sub.index[-1]] = max(scores, key=scores.get)

    return pd.Series(winners, name="dominant_pioneer")

dom_pioneer = rolling_dominant_pioneer_vs_ua(panel, window=24)

# Save counts
dom_counts = dom_pioneer.value_counts(dropna=True).rename("count")
save_table(dom_counts, "partB1_dominant_pioneer_counts")

# Plot dominant pioneer over time
labels = sorted(dom_pioneer.dropna().unique())
mapping = {lab: i for i, lab in enumerate(labels)}
yvals = dom_pioneer.map(mapping)

plt.figure(figsize=(14, 5))
plt.scatter(dom_pioneer.index, yvals, s=15)
plt.yticks(list(mapping.values()), list(mapping.keys()))
plt.title("Part B.1 — Dominant EU pioneer relative to Ukraine (24-month rolling window)")
plt.xlabel("Time")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(FIG_DIR / "partB1_dominant_pioneer_vs_ua.png", dpi=220)
plt.close()

print("\nDominant pioneer counts:")
print(dom_counts)

'''
Even though the pioneer does change through time, is it strongly persistent as Austria is by far
the most frequent EU pioneer relative to Ukraine. 
'''

# =========================================================
# PART B.2 — FORECASTING EVALUATION
# =========================================================

print("\n" + "=" * 70)
print("PART B.2 — FORECASTING EVALUATION")
print("=" * 70)

methods = {
    "PDM_Angles": lambda X: pdm.compute_pioneer_weights_angles(X),
    "PDM_Distance": lambda X: pdm.compute_pioneer_weights_distance(X),
    "Granger": lambda X: pdm.compute_granger_weights(X, maxlag=1),
    "Lagged_Correlation": lambda X: pdm.compute_lagged_correlation_weights(X, lag=1),
    "Multivariate_Regression": lambda X: pdm.compute_multivariate_regression_weights(X, lag=1),
    "Transfer_Entropy": lambda X: pdm.compute_transfer_entropy_weights(X, n_bins=3, lag=1),
    "Linear_Pooling": lambda X: pdm.compute_linear_pooling_weights(X),
}

forecasts = {}
rmse_total = {}
rmse_subperiod = {}

for method_name, weight_fn in methods.items():
    try:
        w = weight_fn(eu_panel)
        w = ensure_datetime_index(w)
        w = normalize_if_needed(w)
        fcast = pdm.pooled_forecast(eu_panel, w).rename(method_name)

        forecasts[method_name] = fcast
        rmse_total[method_name] = rmse(actual_ua, fcast)

        per_scores = {}
        for pname, (start, end) in PERIODS.items():
            per_scores[pname] = rmse(actual_ua.loc[start:end], fcast.loc[start:end])
        rmse_subperiod[method_name] = per_scores

    except Exception as e:
        print(f"{method_name} failed: {e}")
        rmse_total[method_name] = np.nan
        rmse_subperiod[method_name] = {p: np.nan for p in PERIODS}

# Median pooling returns a Series directly
try:
    median_fcast = pdm.compute_median_pooling(eu_panel).rename("Median_Pooling")
    forecasts["Median_Pooling"] = median_fcast
    rmse_total["Median_Pooling"] = rmse(actual_ua, median_fcast)

    per_scores = {}
    for pname, (start, end) in PERIODS.items():
        per_scores[pname] = rmse(actual_ua.loc[start:end], median_fcast.loc[start:end])
    rmse_subperiod["Median_Pooling"] = per_scores

except Exception as e:
    print(f"Median_Pooling failed: {e}")
    rmse_total["Median_Pooling"] = np.nan
    rmse_subperiod["Median_Pooling"] = {p: np.nan for p in PERIODS}

# Save RMSE tables
rmse_total_df = pd.DataFrame.from_dict(rmse_total, orient="index", columns=["RMSE_Total"]).sort_values("RMSE_Total")
rmse_subperiod_df = pd.DataFrame.from_dict(rmse_subperiod, orient="index")

save_table(rmse_total_df, "partB2_rmse_total")
save_table(rmse_subperiod_df, "partB2_rmse_by_subperiod")

print("\nRMSE total by method:")
print(rmse_total_df.round(4))

print("\nRMSE by subperiod:")
print(rmse_subperiod_df.round(4))

# Save forecast panel
forecast_df = pd.concat([actual_ua.rename("UA_Actual")] + list(forecasts.values()), axis=1)
save_table(forecast_df, "partB2_forecasts_vs_actual")

# Plot forecasts
plt.figure(figsize=(14, 7))
plt.plot(forecast_df.index, forecast_df["UA_Actual"], linewidth=2.5, label="UA Actual")
for col in forecast_df.columns:
    if col != "UA_Actual":
        plt.plot(forecast_df.index, forecast_df[col], linewidth=1.1, alpha=0.9, label=col)
plt.title("Part B.2 — Ukraine inflation: actual vs pooled estimates")
plt.xlabel("Time")
plt.ylabel("Inflation (y/y, %)")
plt.legend(ncol=3, fontsize=8, frameon=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "partB2_forecasts_vs_actual.png", dpi=220)
plt.close()

print("\nDone.")
print(f"Figures saved to: {FIG_DIR}")
print(f"Tables saved to:  {TAB_DIR}")

''' 
B2.c

The most accurate forecast of Ukraine's inflation is given by PDM_Distance, with a RMSE of
14.218698, followed closely (almost identical) by PDM_Angles with a RMSE of 14.218754.

B2.d

The PDM was designed to identify pioneers and accelerate collective learning, not to forecast 
a target variable. In this exercise, a good RMSE only means that a weighted combination of EU 
inflation series happens to track Ukraine’s inflation well. This may reflect common shocks 
rather than true predictive structure. The interpretation is further limited because the pooled 
estimate uses contemporaneous information and falls back to the cross-sectional mean when no
pioneer is detected, so it is not a genuine ex ante forecast. In addition, some methods use 
time-varying weights while others rely on effectively constant weights, and the underlying
data are revised rather than real-time vintages. Therefore, the RMSE comparison is useful as 
a tracking exercise, but not as proof of forecasting ability.
'''