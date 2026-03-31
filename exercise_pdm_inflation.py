"""Exercise A/B: Pioneer Detection on European inflation panel.

This script implements the core tasks from the exercise in a single, self-contained
script:

  * Part A: compute PDM pioneer weights for the full EU panel, plot their time
    evolution, and save mean weights by subperiod.
  * Part B: treat FRANCE as the "target" country (in place of Ukraine), and
    forecast France using the remaining EU countries as experts. This produces:
      - a rolling "dominant pioneer" plot (which expert is currently strongest)
      - RMSE comparisons of several pooling methods over the full sample + subperiods.

Run:
  python exercise_pdm_inflation.py

Outputs saved (in current dir):
  - pioneer_weights_heatmap.png
  - pioneer_weights_lines_top5.png
  - pioneer_weights_subperiod_table.csv
  - dominant_pioneer_fr_rolling.png
  - fr_forecast_rmse.csv

After running, inspect the PNGs to interpret the time-series patterns and inspect
+ write-up based on the CSVs (subperiod averages and RMSE)."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecb_hicp_panel_var_granger import fetch_ecb_hicp_inflation_panel
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

# --- Configuration ----------------------------------------------------------------
START = "2000-01"
END = "2025-12"
EU_COUNTRIES = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]
PERIODS = {
    "I_2002_2007": ("2002-01", "2007-12"),
    "II_2008_2012": ("2008-01", "2012-12"),
    "III_2013_2019": ("2013-01", "2019-12"),
    "IV_2020_2021": ("2020-01", "2021-12"),
    "V_2022_2023": ("2022-01", "2023-12"),
    "VI_2024_2025": ("2024-01", "2025-12"),
}

OUTPUT_DIR = os.path.dirname(__file__)

# --- Part A: Weights over time and subperiod averages ---------------------------

panel, _ = fetch_ecb_hicp_inflation_panel(countries=EU_COUNTRIES, start=START, end=END)
panel_cc = panel.dropna(how="any")
print("Panel shape:", panel.shape, "-> complete-case:", panel_cc.shape)

weights = compute_pioneer_weights_angles(panel_cc)

nan_rows = weights.isna().any(axis=1).sum()
print(f"Rows with no pioneer (NaN weights): {nan_rows}")

weights_plot = weights.fillna(0.0)
nonzero_counts = (weights_plot != 0).sum()
print("Non-zero counts per country:\n", nonzero_counts)

# heatmap (full period)
plt.figure(figsize=(12, 6))
plt.imshow(weights_plot.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
plt.yticks(range(weights_plot.shape[1]), weights_plot.columns)
plt.xticks(
    range(0, weights_plot.shape[0], max(1, weights_plot.shape[0] // 10)),
    [weights_plot.index[i].strftime("%Y-%m") for i in range(0, weights_plot.shape[0], max(1, weights_plot.shape[0] // 10))],
)
plt.title("Pioneer weights (angles) — full panel")
plt.xlabel("Time")
plt.colorbar(label="Weight")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pioneer_weights_heatmap.png"), dpi=150)
plt.close()

# top-5 line plot
mean_abs = weights_plot.abs().mean().sort_values(ascending=False)
top5 = mean_abs.index[:5].tolist()
plt.figure(figsize=(12, 5))
for c in top5:
    plt.plot(weights_plot.index, weights_plot[c], label=c, linewidth=1.5)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("Pioneer weights (top 5 countries by mean |weight|)")
plt.xlabel("Time")
plt.ylabel("Weight")
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pioneer_weights_lines_top5.png"), dpi=150)
plt.close()

# average weights by subperiod
period_avg = pd.DataFrame(index=weights_plot.columns)
for name, (start, end) in PERIODS.items():
    window = weights_plot.loc[start:end]
    period_avg[name] = window.mean()

period_avg.to_csv(os.path.join(OUTPUT_DIR, "pioneer_weights_subperiod_table.csv"))
print("Average weights by subperiod saved to pioneer_weights_subperiod_table.csv")
print(period_avg)

# --- Part B: FRANCE as target ----------------------------------------------------

TARGET = "FR"
experts = [c for c in EU_COUNTRIES if c != TARGET]

panel_cc = panel_cc[experts + [TARGET]]  # ensure order
expert_panel = panel_cc[experts]
target_series = panel_cc[TARGET]

# B1: Rolling pioneer (dominant expert over time)
window = 24

# Map countries to integers for plotting
country_to_code = {c: i for i, c in enumerate(experts)}
code_to_country = {i: c for c, i in country_to_code.items()}

dominant_codes = []
dates = []
missing = 0
for end_idx in range(window, len(panel_cc) + 1):
    window_data = panel_cc.iloc[end_idx - window : end_idx]
    w = compute_pioneer_weights_angles(window_data[experts])
    last = w.iloc[-1][experts]

    if last.isna().all():
        # No pioneer detected in this window
        dominant_codes.append(np.nan)
        missing += 1
    else:
        dominant = last.idxmax()
        dominant_codes.append(country_to_code[dominant])

    dates.append(window_data.index[-1])

print(f"Rolling pioneer windows with no detected pioneer: {missing} / {len(dates)}")

plt.figure(figsize=(12, 3))
plt.plot(dates, dominant_codes, marker="o", linestyle="-", markersize=3)
plt.yticks(list(code_to_country.keys()), list(code_to_country.values()))
plt.title(f"Dominant pioneer (expert) for {TARGET} — rolling {window}-month window")
plt.xlabel("Time")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dominant_pioneer_fr_rolling.png"), dpi=150)
plt.close()

# B2: Forecast RMSE for target using various pooling methods
methods = {
    "PDM_angles": compute_pioneer_weights_angles,
    "PDM_distance": compute_pioneer_weights_distance,
    "Granger": compute_granger_weights,
    "LagCorr": compute_lagged_correlation_weights,
    "LinReg": compute_multivariate_regression_weights,
    "TransEnt": compute_transfer_entropy_weights,
    "LinPool": compute_linear_pooling_weights,
    "Median": compute_median_pooling,
}

rmse_rows = []
for name, func in methods.items():
    if name == "Median":
        pred = func(expert_panel)
    else:
        weights_experts = func(expert_panel)
        pred = pooled_forecast(expert_panel, weights_experts)

    pred = pred.reindex(target_series.index)
    mask = target_series.notna() & pred.notna()

    rmse_all = np.sqrt(((pred[mask] - target_series[mask]) ** 2).mean())

    row = {"method": name, "RMSE_all": rmse_all}
    for pname, (s, e) in PERIODS.items():
        mask_p = (target_series.index >= s) & (target_series.index <= e) & mask
        row[f"RMSE_{pname}"] = (
            np.sqrt(((pred[mask_p] - target_series[mask_p]) ** 2).mean())
            if mask_p.any()
            else np.nan
        )
    rmse_rows.append(row)

rmse_df = pd.DataFrame(rmse_rows).set_index("method")
rmse_df.to_csv(os.path.join(OUTPUT_DIR, "fr_forecast_rmse.csv"))
print("Saved FR forecast RMSE results to fr_forecast_rmse.csv")

print("\nDONE — outputs saved.\n")
print("Key files to interpret and use for write-up:")
print("  - pioneer_weights_heatmap.png: time-series panel heatmap of pioneer weights")
print("  - pioneer_weights_lines_top5.png: top 5 countries by mean weight over time")
print("  - pioneer_weights_subperiod_table.csv: average weights in defined subperiods")
print("  - dominant_pioneer_fr_rolling.png: which expert country is dominant over rolling windows")
print("  - fr_forecast_rmse.csv: RMSE of pooled forecasts for France (full sample + subperiods)")
