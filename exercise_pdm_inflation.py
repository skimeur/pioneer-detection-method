#!/usr/bin/env python3
# PDM Exercise — inflation dynamics, target = France

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings

from ecb_hicp_panel_var_granger import build_inflation_panel

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

warnings.filterwarnings("ignore")

# load data
infl_panel = build_inflation_panel()
panel = infl_panel.dropna()
print(f"Panel: {panel.shape}, countries: {list(panel.columns)}")

PERIODS = {
    "I (2002-07)":   ("2002-01", "2007-12"),
    "II (2008-12)":  ("2008-01", "2012-12"),
    "III (2013-19)": ("2013-01", "2019-12"),
    "IV (2020-21)":  ("2020-01", "2021-12"),
    "V (2022-23)":   ("2022-01", "2023-12"),
    "VI (2024-25)":  ("2024-01", "2025-12"),
}

TARGET = "FR"

# ===================== PART A =====================
print("\n=== PART A ===")

# A.1 — PDM weights over time
w_angles = compute_pioneer_weights_angles(panel)

fig, ax = plt.subplots(figsize=(14, 6))
colors = cm.tab20(np.linspace(0, 1, len(panel.columns)))
for i, country in enumerate(panel.columns):
    ax.plot(w_angles.index, w_angles[country], label=country, linewidth=1.0, color=colors[i], alpha=0.8)
ax.set_xlabel("Time")
ax.set_ylabel("Pioneer weight")
ax.set_title("A.1 — PDM (angles) pioneer weights over time")
ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left")
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig("fig_a1_pioneer_weights_over_time.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(14, 5))
w_fill = w_angles.fillna(0)
im = ax.pcolormesh(w_fill.index, range(len(w_fill.columns)), w_fill.T.values, cmap="YlOrRd", shading="auto")
ax.set_yticks(range(len(w_fill.columns)))
ax.set_yticklabels(w_fill.columns)
ax.set_xlabel("Time")
ax.set_title("A.1 — Pioneer weights heatmap")
plt.colorbar(im, ax=ax, label="Pioneer weight")
plt.tight_layout()
plt.savefig("fig_a1_pioneer_weights_heatmap.png", dpi=150)
plt.show()

print("\nA.1(c) Average pioneer weight (full sample):")
print(w_angles.mean().sort_values(ascending=False).to_string())

# A.2 — Average weights by subperiod
subperiod_table = pd.DataFrame(index=panel.columns, dtype=float)
for name, (s, e) in PERIODS.items():
    sub = w_angles.loc[s:e]
    if len(sub) > 0:
        subperiod_table[name] = sub.mean()

print("\nA.2(a) Average pioneer weight by country and subperiod:")
print(subperiod_table.round(4).to_string())

print("\nA.2(b) Ranking per subperiod:")
for name in subperiod_table.columns:
    ranked = subperiod_table[name].sort_values(ascending=False)
    print(f"  {name}: {', '.join(f'{c}={v:.3f}' for c, v in ranked.items() if not np.isnan(v))}")

# A.1(d) + A.2(c) — Discussion
print("""
A.1(d): During stable inflation (2002-07), pioneer signals are weak because
there's no divergence-convergence pattern. During breaks (GFC, COVID, energy
crisis), countries hit first show up as pioneers.

A.2(c): NL and BE rank high during 2022-23 energy crisis — they are small open
economies heavily dependent on gas imports, so energy shocks hit them first.
FI's high weight in 2020-21 reflects early COVID supply chain disruption via
Nordic trade channels. AT and IT lead in calmer periods due to tight trade links
with Central/Eastern Europe. GR spikes during debt crisis periods. Countries
with larger, more diversified economies (DE, FR) tend to be followers rather
than pioneers — shocks are absorbed more gradually.
""")

# ===================== PART B =====================
print(f"\n=== PART B (target = {TARGET}) ===")

other_cols = [c for c in panel.columns if c != TARGET]
other_panel = panel[other_cols]
actual_target = panel[TARGET]

# B.1 — Rolling pioneer detection
WINDOW = 36

dominant_pioneers = []
for end_idx in range(WINDOW, len(panel)):
    window_slice = panel.iloc[end_idx - WINDOW:end_idx]
    date = panel.index[end_idx - 1]

    w_window = compute_pioneer_weights_angles(window_slice)
    avg_w = w_window.mean()
    avg_w_others = avg_w.drop(TARGET, errors="ignore")
    if avg_w_others.notna().any() and avg_w_others.sum() > 0:
        dominant = avg_w_others.idxmax()
        dominant_weight = avg_w_others.max()
    else:
        dominant = "None"
        dominant_weight = 0.0

    dominant_pioneers.append({
        "date": date,
        "dominant_pioneer": dominant,
        "weight": dominant_weight,
    })

dom_df = pd.DataFrame(dominant_pioneers).set_index("date")

fig, ax = plt.subplots(figsize=(14, 4))
country_codes = sorted(dom_df["dominant_pioneer"].unique())
country_to_num = {c: i for i, c in enumerate(country_codes)}
dom_df["pioneer_num"] = dom_df["dominant_pioneer"].map(country_to_num)

ax.scatter(dom_df.index, dom_df["pioneer_num"], c=dom_df["pioneer_num"],
           cmap="tab20", s=10, alpha=0.8)
ax.set_yticks(range(len(country_codes)))
ax.set_yticklabels(country_codes)
ax.set_xlabel("Time")
ax.set_ylabel("Dominant pioneer")
ax.set_title(f"B.1 — Dominant pioneer for {TARGET} ({WINDOW}-month rolling window)")
plt.tight_layout()
plt.savefig("fig_b1_dominant_pioneer.png", dpi=150)
plt.show()

print("B.1 Dominant pioneer counts by subperiod:")
for name, (s, e) in PERIODS.items():
    sub = dom_df.loc[s:e]
    if len(sub) > 0:
        counts = sub["dominant_pioneer"].value_counts().head(3)
        print(f"  {name}: {', '.join(f'{c}({n})' for c, n in counts.items())}")

# B.2 — Forecasting evaluation

METHODS = {
    "PDM (angles)": lambda df: compute_pioneer_weights_angles(df),
    "PDM (distances)": lambda df: compute_pioneer_weights_distance(df),
    "Granger causality": lambda df: compute_granger_weights(df),
    "Lagged correlation": lambda df: compute_lagged_correlation_weights(df),
    "Multivariate regression": lambda df: compute_multivariate_regression_weights(df),
    "Transfer entropy": lambda df: compute_transfer_entropy_weights(df),
    "Linear pooling": lambda df: compute_linear_pooling_weights(df),
}


def compute_rmse(pred, actual):
    c = pred.index.intersection(actual.index)
    p, a = pred.loc[c], actual.loc[c]
    v = p.notna() & a.notna()
    return np.sqrt(((p[v] - a[v]) ** 2).mean()) if v.sum() > 0 else np.nan


rmse_overall = {}
rmse_by_period = {name: {} for name in PERIODS}
forecasts_dict = {}

for method_name, weight_fn in METHODS.items():
    w = weight_fn(other_panel)
    fc = pooled_forecast(other_panel, w)
    forecasts_dict[method_name] = fc
    rmse_overall[method_name] = compute_rmse(fc, actual_target)

    for period_name, (s, e) in PERIODS.items():
        fc_sub = fc.loc[s:e]
        act_sub = actual_target.loc[s:e]
        rmse_by_period[period_name][method_name] = compute_rmse(fc_sub, act_sub)

median_fc = compute_median_pooling(other_panel)
forecasts_dict["Median pooling"] = median_fc
rmse_overall["Median pooling"] = compute_rmse(median_fc, actual_target)
for period_name, (s, e) in PERIODS.items():
    fc_sub = median_fc.loc[s:e]
    act_sub = actual_target.loc[s:e]
    rmse_by_period[period_name]["Median pooling"] = compute_rmse(fc_sub, act_sub)

print(f"\nB.2(b) Overall RMSE (target = {TARGET}):")
print(pd.Series(rmse_overall).sort_values().round(4).to_string())

print(f"\nB.2(c) RMSE by method and subperiod:")
print(pd.DataFrame(rmse_by_period).round(4).to_string())

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(actual_target.index, actual_target, label=f"{TARGET} (actual)",
        color="black", linewidth=2)

top_methods = ["PDM (angles)", "Linear pooling", "Median pooling"]
method_colors = ["tab:blue", "tab:orange", "tab:green"]
for method_name, color in zip(top_methods, method_colors):
    fc = forecasts_dict[method_name]
    common = fc.index.intersection(actual_target.index)
    ax.plot(common, fc.loc[common], label=method_name,
            linewidth=1, alpha=0.8, color=color)

ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.set_xlabel("Time")
ax.set_ylabel("Inflation rate (y/y, %)")
ax.set_title(f"B.2 — Pooled forecasts vs actual {TARGET} inflation")
ax.legend(fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig("fig_b2_forecasts_vs_actual.png", dpi=150)
plt.show()

print(f"""
B.2(d): PDM is not a forecasting tool. Low RMSE just means the weighted average
of EU countries tracks {TARGET} well in-sample, not that it predicts future
inflation. The "forecast" uses time-t values (no lag), and weights are fitted
on the same data. Since {TARGET} shares monetary policy with the eurozone,
simple averages already track it well. PDM's value is identifying who moves
first, not minimizing RMSE.
""")

print("Done.")
