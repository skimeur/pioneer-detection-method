#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise: Applying the Pioneer Detection Method to European Inflation Dynamics
Andrey Zalizniak

Uses functions from:
  - ecb_hicp_panel_var_granger.py  (data fetching via ECB SDMX)
  - pdm.py                        (all PDM / alternative methods)

The script first attempts to fetch Ukrainian CPI data from the SSSU API.
If that fails, it falls back to GB fetched from the ECB.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR

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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EU_COUNTRIES = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]

FALLBACK_COUNTRY = "GB"
FALLBACK_LABEL = "United Kingdom"

START = "2000-01"
END = "2025-12"

SUBPERIODS = {
    "I (2002-07)":   ("2002-01", "2007-12"),
    "II (2008-12)":  ("2008-01", "2012-12"),
    "III (2013-19)": ("2013-01", "2019-12"),
    "IV (2020-21)":  ("2020-01", "2021-12"),
    "V (2022-23)":   ("2022-01", "2023-12"),
    "VI (2024-25)":  ("2024-01", "2025-12"),
}

# ---------------------------------------------------------------------------
# Step 1: Fetch EU inflation panel
# ---------------------------------------------------------------------------

print("Fetching EU HICP inflation panel...")
infl_panel, _ = fetch_ecb_hicp_inflation_panel(
    countries=EU_COUNTRIES,
    start=START,
    end=END,
)
infl_panel.index = pd.to_datetime(infl_panel.index).to_period("M").to_timestamp(how="start")

# ---------------------------------------------------------------------------
# Step 2: Try Ukraine, fall back to GB
# ---------------------------------------------------------------------------

try:
    import requests
    from io import StringIO

    print("Attempting to fetch Ukraine CPI from SSSU...")
    base = "https://stat.gov.ua/sdmx/workspaces/default:integration/registry/sdmx/3.0/data"
    key = "INDEX_CONSUMPRICE.PREV_MONTH.UA00000000000000000.0.M"
    url = f"{base}/dataflow/SSSU/DF_PRICE_CHANGE_CONSUMER_GOODS_SERVICE/~/{key}"
    params = {"c[TIME_PERIOD]": f"ge:{START}+le:{END}"}
    headers = {
        "Accept": "application/vnd.sdmx.data+csv;version=2.0.0;labels=id;timeFormat=normalized;keys=both",
        "User-Agent": "Mozilla/5.0",
    }
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    ua_raw = pd.read_csv(StringIO(r.text), dtype=str)
    ua_raw = ua_raw.loc[
        ua_raw["TIME_PERIOD"].astype(str).str.match(r"^\d{4}-M\d{2}$", na=False)
        & ua_raw["OBS_VALUE"].notna()
    ].copy()

    if ua_raw.empty:
        raise ValueError("SSSU response contained no usable observations")

    s = ua_raw[["TIME_PERIOD", "OBS_VALUE"]].copy()
    s["TIME_PERIOD"] = pd.to_datetime(
        s["TIME_PERIOD"].str.replace(r"^(\d{4})-M(\d{2})$", r"\1-\2-01", regex=True)
    )
    s["OBS_VALUE"] = pd.to_numeric(
        s["OBS_VALUE"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    s = s.dropna().sort_values("TIME_PERIOD").set_index("TIME_PERIOD")
    monthly_factor = s["OBS_VALUE"] / 100.0
    yoy_factor = monthly_factor.rolling(12).apply(np.prod, raw=True)
    target_series = ((yoy_factor - 1.0) * 100.0).rename("UA")
    target_series.index = target_series.index.to_period("M").to_timestamp(how="start")

    TARGET_COUNTRY = "UA"
    TARGET_LABEL = "Ukraine"
    print("  -> Ukraine data fetched successfully.")

except Exception as e:
    print(f"  -> Ukraine data unavailable ({e})")
    print(f"  -> Falling back to {FALLBACK_COUNTRY} ({FALLBACK_LABEL}) from ECB.")

    TARGET_COUNTRY = FALLBACK_COUNTRY
    TARGET_LABEL = FALLBACK_LABEL

    target_panel, _ = fetch_ecb_hicp_inflation_panel(
        countries=[FALLBACK_COUNTRY],
        start=START,
        end=END,
    )
    target_panel.index = pd.to_datetime(target_panel.index).to_period("M").to_timestamp(how="start")
    target_series = target_panel[FALLBACK_COUNTRY].rename(FALLBACK_COUNTRY)

# ---------------------------------------------------------------------------
# Step 3: Merge into a single panel
# ---------------------------------------------------------------------------

infl_panel = infl_panel.join(target_series, how="left")
ALL_COUNTRIES = EU_COUNTRIES + [TARGET_COUNTRY]

print(f"\nTarget country : {TARGET_COUNTRY} ({TARGET_LABEL})")
print(f"Panel shape    : {infl_panel.shape}")
print(f"Date range     : {infl_panel.index.min()} to {infl_panel.index.max()}")
print(f"Columns        : {list(infl_panel.columns)}")
print(infl_panel.head())

# Plot the inflation panel
fig, ax = plt.subplots(figsize=(12, 6))
cmap = plt.get_cmap("tab20", len(infl_panel.columns))
for i, country in enumerate(infl_panel.columns):
    ax.plot(infl_panel.index, infl_panel[country], label=country,
            linewidth=1, color=cmap(i))
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Time")
ax.set_ylabel("Inflation rate (y/y, %)")
ax.set_title("HICP Inflation Panel (ECB Data Portal)")
ax.legend(ncol=4, fontsize=8, frameon=False)
fig.tight_layout()
plt.savefig("fig_inflation_panel.png", dpi=150)
plt.close(fig)
print("Saved: fig_inflation_panel.png")

# Prepare complete-case data
df = infl_panel.copy().sort_index().dropna()
print(f"\nComplete-case panel: {df.shape[0]} months, {df.shape[1]} countries")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# ADF unit-root tests
print(f"\nADF unit-root tests (levels)\n")
adf_results = []
for c in df.columns:
    stat, pval, _, _, _, _ = adfuller(df[c], autolag="AIC")
    adf_results.append({"country": c, "ADF_stat": round(stat, 4), "pvalue": round(pval, 4)})
adf_table = pd.DataFrame(adf_results).sort_values("pvalue")
print(adf_table.to_string(index=False))

# Granger causality: X -> TARGET_COUNTRY
maxlag = 6
print(f"\nGranger causality tests: X -> {TARGET_COUNTRY}\n")
granger_out = []
for c in df.columns:
    if c == TARGET_COUNTRY:
        continue
    data_gc = df[[TARGET_COUNTRY, c]]
    try:
        res = grangercausalitytests(data_gc, maxlag=maxlag, verbose=False)
        min_p = min(res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1))
        granger_out.append({"country": c, "min_pvalue": round(min_p, 6)})
    except Exception as exc:
        print(f"  Granger test failed for {c}: {exc}")

granger_rank = pd.DataFrame(granger_out).sort_values("min_pvalue").reset_index(drop=True)
print(f"Ranking of countries by Granger causality for {TARGET_COUNTRY}\n")
print(granger_rank.to_string(index=False))

# VAR: TARGET_COUNTRY + top 2 Granger predictors
top_countries = granger_rank["country"].iloc[:2].tolist()
var_vars = [TARGET_COUNTRY] + top_countries
print(f"\nVAR variables: {var_vars}")

X_var = df[var_vars]
model = VAR(X_var)
lag_selection = model.select_order(maxlags=6)
p = max(1, lag_selection.selected_orders["bic"])

print(f"\nVAR lag selection (BIC)\n")
print(lag_selection.summary())
print(f"Selected lag order p = {p}")

var_res = model.fit(p)
print(f"\nVAR estimation results\n")
print(var_res.summary())


# ---------------------------------------------------------------------------
# PART A: Who pioneered European inflation dynamics?
# ---------------------------------------------------------------------------
# All 12 series (11 EU + TARGET_COUNTRY) treated as "experts".

panel = df.copy()
print(f"\nPart A")
print(f"Panel: {panel.shape[0]} months, {panel.shape[1]} countries\n")

# ---------------------------------------------------------------------------
# A.1 - PDM weights over time
# ---------------------------------------------------------------------------

w_angles = compute_pioneer_weights_angles(panel)

# A.1(c) - Line plot of pioneer weights
fig, ax = plt.subplots(figsize=(14, 6))
cmap = plt.get_cmap("tab20", len(panel.columns))
for i, country in enumerate(panel.columns):
    ax.plot(w_angles.index, w_angles[country], label=country,
            linewidth=1, color=cmap(i))
ax.set_xlabel("Time")
ax.set_ylabel("Pioneer weight (PDM angles)")
ax.set_title("A.1 - Pioneer weights over time (all 12 countries)")
ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left")
ax.set_ylim(bottom=0)
fig.tight_layout()
plt.savefig("fig_A1_pioneer_weights_time.png", dpi=150)
plt.close(fig)
print("Saved: fig_A1_pioneer_weights_time.png")

# A.1(c) - Heatmap
fig, ax = plt.subplots(figsize=(14, 4))
w_plot = w_angles.fillna(0).T
im = ax.pcolormesh(w_plot.columns, range(len(w_plot.index)), w_plot.values,
                   cmap="YlOrRd", shading="auto")
ax.set_yticks(range(len(w_plot.index)))
ax.set_yticklabels(w_plot.index)
ax.set_xlabel("Time")
ax.set_title("A.1 - Pioneer weights heatmap")
fig.colorbar(im, ax=ax, label="Weight")
fig.tight_layout()
plt.savefig("fig_A1_pioneer_weights_heatmap.png", dpi=150)
plt.close(fig)
print("Saved: fig_A1_pioneer_weights_heatmap.png")

# A.1(c) - Fraction of time each country has non-zero weight
nonzero_frac = (w_angles.fillna(0) > 0).mean().sort_values(ascending=False)
print("\nA.1(c) Fraction of months with non-zero pioneer weight:\n")
print(nonzero_frac.round(4).to_string())

# ---------------------------------------------------------------------------
# A.2 - Average pioneer weights by subperiod
# ---------------------------------------------------------------------------

avg_weights = pd.DataFrame(index=panel.columns, columns=list(SUBPERIODS.keys()))
for name, (s, e) in SUBPERIODS.items():
    sub = w_angles.loc[s:e]
    if not sub.empty:
        avg_weights[name] = sub.mean()

avg_weights = avg_weights.astype(float).dropna(axis=1, how="all")

print(f"\nA.2(a) Average PDM (angles) weights by subperiod\n")
print(avg_weights.round(4).to_string())

# A.2(b) - Rankings per subperiod
print(f"\nA.2(b) Rankings (1 = highest weight)\n")
rankings = avg_weights.rank(ascending=False, method="min").astype(int)
print(rankings.to_string())


# ---------------------------------------------------------------------------
# Insight check: persistent inflation level vs PDM overweighting
# ---------------------------------------------------------------------------
# Hypothesis: if a country's inflation is persistently above (or below) the
# cross-sectional mean, the PDM might overweight it because any move by
# others "toward" that country registers as convergence.

print(f"\nInsight check: does a persistent inflation level drive PDM weight?\n")

# 1) Overall correlation: mean inflation level vs mean PDM weight
mean_inflation = panel.mean()
mean_weight = w_angles.mean()
insight_df = pd.DataFrame({
    "mean_inflation": mean_inflation,
    "mean_pdm_weight": mean_weight,
}).sort_values("mean_pdm_weight", ascending=False)
print(insight_df.round(4).to_string())

corr_level_weight = insight_df["mean_inflation"].corr(insight_df["mean_pdm_weight"])
print(f"\nCorrelation(mean inflation level, mean PDM weight) = {corr_level_weight:.4f}")

# 2) Per-subperiod Spearman rank correlation
print(f"\nPer-subperiod Spearman rank correlation (inflation level vs PDM weight):\n")
for name, (s, e) in SUBPERIODS.items():
    sub_panel = panel.loc[s:e]
    sub_w = w_angles.loc[s:e]
    if sub_panel.empty or sub_w.empty:
        continue
    level_mean = sub_panel.mean()
    weight_mean = sub_w.mean()
    rho = level_mean.corr(weight_mean, method="spearman")
    print(f"  {name}: rho = {rho:.4f}")

# 3) Time-varying: at each t, correlate inflation rank with weight rank
rolling_corr = []
for t in w_angles.index:
    w_t = w_angles.loc[t].dropna()
    x_t = panel.loc[t, w_t.index]
    if len(w_t) >= 3 and w_t.sum() > 0:
        rho = x_t.corr(w_t, method="spearman")
        rolling_corr.append({"date": t, "spearman_rho": rho})

if rolling_corr:
    rc_df = pd.DataFrame(rolling_corr).set_index("date")
    avg_rho = rc_df["spearman_rho"].mean()
    print(f"\nTime-varying rank correlation (inflation level vs PDM weight at each t):")
    print(f"  Average Spearman rho = {avg_rho:.4f}")
    print(f"  Fraction positive    = {(rc_df['spearman_rho'] > 0).mean():.2%}")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(rc_df.index, rc_df["spearman_rho"], linewidth=0.8, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Rank correlation: inflation level vs PDM weight at each month")
    fig.tight_layout()
    plt.savefig("fig_A_insight_level_vs_weight.png", dpi=150)
    plt.close(fig)
    print("Saved: fig_A_insight_level_vs_weight.png")

    if abs(avg_rho) > 0.3:
        print("\n  -> Evidence of level effect: countries persistently far from consensus")
        print("     tend to attract higher PDM weight. This could indicate overweighting")
        print("     of outlier series that others converge toward mechanically.")
    else:
        print("\n  -> Weak level effect: PDM weights mainly reflect directional moves,")
        print("     not persistent level differences.")

# 4) Mean absolute deviation from panel mean vs PDM weight
print(f"\nMean absolute deviation from panel mean vs mean PDM weight:\n")
panel_mean = panel.mean(axis=1)
abs_dev = panel.sub(panel_mean, axis=0).abs().mean()
dev_df = pd.DataFrame({
    "mean_abs_deviation": abs_dev,
    "mean_pdm_weight": mean_weight,
}).sort_values("mean_pdm_weight", ascending=False)
print(dev_df.round(4).to_string())
corr_dev = dev_df["mean_abs_deviation"].corr(dev_df["mean_pdm_weight"])
print(f"\nCorrelation(mean abs deviation, mean PDM weight) = {corr_dev:.4f}")
if abs(corr_dev) > 0.5:
    print("  -> Countries persistently far from the group mean do receive higher")
    print("     PDM weight on average. Your overweighting insight is supported.")
else:
    print("  -> Deviation from consensus does not strongly predict PDM weight.")


# ---------------------------------------------------------------------------
# PART B: Predicting target country's inflation trajectory
# ---------------------------------------------------------------------------
# TARGET_COUNTRY = "true parameter to predict", 11 EU countries = "experts"

expert_cols = [c for c in panel.columns if c != TARGET_COUNTRY]
eu_panel = panel[expert_cols]
actual_target = panel[TARGET_COUNTRY]

print(f"\nPart B: target = {TARGET_COUNTRY} ({TARGET_LABEL})")
print(f"  Experts: {list(expert_cols)}")
print(f"  Observations: {len(panel)}")


# B.1 - Rolling pioneer detection
ROLLING_WINDOW = 36  # months

dominant_pioneer = pd.Series(index=panel.index, dtype=str)

for end_idx in range(ROLLING_WINDOW, len(panel)):
    window = eu_panel.iloc[end_idx - ROLLING_WINDOW : end_idx]
    w_roll = compute_pioneer_weights_angles(window)
    avg_w = w_roll.mean()
    if avg_w.notna().any() and avg_w.sum() > 0:
        dominant_pioneer.iloc[end_idx] = avg_w.idxmax()

dominant_pioneer = dominant_pioneer.dropna()

# B.1(b) - Plot dominant pioneer over time
fig, ax = plt.subplots(figsize=(14, 4))
pioneer_codes = sorted(dominant_pioneer.unique())
code_to_num = {c: i for i, c in enumerate(pioneer_codes)}
y_vals = dominant_pioneer.map(code_to_num)
cmap_b1 = plt.get_cmap("tab20", len(pioneer_codes))

ax.scatter(dominant_pioneer.index, y_vals, c=[cmap_b1(v) for v in y_vals],
           s=12, edgecolors="none")
ax.set_yticks(range(len(pioneer_codes)))
ax.set_yticklabels(pioneer_codes)
ax.set_xlabel("Time")
ax.set_ylabel("Dominant pioneer")
ax.set_title(f"B.1 - Dominant pioneer for {TARGET_COUNTRY} ({TARGET_LABEL}) "
             f"over time (rolling {ROLLING_WINDOW}m window)")
fig.tight_layout()
plt.savefig("fig_B1_dominant_pioneer.png", dpi=150)
plt.close(fig)
print("Saved: fig_B1_dominant_pioneer.png")

# Frequency table of dominant pioneer by subperiod
print(f"\nB.1 Dominant pioneer frequency by subperiod\n")
for name, (s, e) in SUBPERIODS.items():
    sub = dominant_pioneer.loc[s:e]
    if sub.empty:
        continue
    counts = sub.value_counts()
    top = counts.index[0]
    print(f"  {name}: top = {top} ({counts.iloc[0]}/{len(sub)} months)")


# ---------------------------------------------------------------------------
# B.2 - Forecasting evaluation using all methods
# ---------------------------------------------------------------------------

methods = {
    "PDM angles":       lambda p: compute_pioneer_weights_angles(p),
    "PDM distance":     lambda p: compute_pioneer_weights_distance(p),
    "Granger":          lambda p: compute_granger_weights(p),
    "Lagged corr.":     lambda p: compute_lagged_correlation_weights(p),
    "Multivar. reg.":   lambda p: compute_multivariate_regression_weights(p),
    "Transfer entropy":  lambda p: compute_transfer_entropy_weights(p),
    "Linear pooling":   lambda p: compute_linear_pooling_weights(p),
}

forecasts_by_method = {}
for method_name, weight_fn in methods.items():
    w = weight_fn(eu_panel)
    forecasts_by_method[method_name] = pooled_forecast(eu_panel, w)

# Median pooling returns a Series directly
forecasts_by_method["Median pooling"] = compute_median_pooling(eu_panel)

# B.2(b-c) - RMSE by method, overall and by subperiod
rmse_table = pd.DataFrame(index=list(forecasts_by_method.keys()),
                          columns=["Overall"] + list(SUBPERIODS.keys()))

for method_name, fc in forecasts_by_method.items():
    errors = (fc - actual_target).dropna()
    if len(errors) > 0:
        rmse_table.loc[method_name, "Overall"] = np.sqrt((errors ** 2).mean())

    for period_name, (s, e) in SUBPERIODS.items():
        err_sub = errors.loc[s:e]
        if len(err_sub) > 0:
            rmse_table.loc[method_name, period_name] = np.sqrt((err_sub ** 2).mean())

rmse_table = rmse_table.astype(float).dropna(axis=1, how="all")

print(f"\nB.2 RMSE: pooled estimate vs actual {TARGET_COUNTRY} ({TARGET_LABEL}) inflation\n")
print(rmse_table.round(4).to_string())

# Best method per column
print(f"\nBest method per period (lowest RMSE):\n")
for col in rmse_table.columns:
    valid = rmse_table[col].dropna()
    if not valid.empty:
        best = valid.idxmin()
        print(f"  {col}: {best} (RMSE = {valid[best]:.4f})")

# B.2(a) - Plot pooled estimates vs actual
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(actual_target.index, actual_target,
        label=f"Actual {TARGET_COUNTRY}", linewidth=2, color="black")
cmap_b2 = plt.get_cmap("tab10", len(forecasts_by_method))
for i, (method_name, fc) in enumerate(forecasts_by_method.items()):
    ax.plot(fc.index, fc, label=method_name, linewidth=1, alpha=0.7,
            color=cmap_b2(i))
ax.set_xlabel("Time")
ax.set_ylabel("Inflation rate (y/y, %)")
ax.set_title(f"B.2 - Pooled estimates vs actual {TARGET_COUNTRY} ({TARGET_LABEL}) inflation")
ax.legend(ncol=3, fontsize=7, frameon=False)
fig.tight_layout()
plt.savefig("fig_B2_pooled_vs_actual.png", dpi=150)
plt.close(fig)
print("Saved: fig_B2_pooled_vs_actual.png")

print("\nDone. All figures saved to current directory.")
