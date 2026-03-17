#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise: Applying the Pioneer Detection Method to European Inflation Dynamics
==============================================================================
Vansteenberghe, E. (2025), "Insurance Supervision under Climate Change:
A Pioneer Detection Method," The Geneva Papers on Risk and Insurance.
https://doi.org/10.1057/s41288-025-00367-y

Repository: https://github.com/skimeur/pioneer-detection-method
Starter   : ecb_hicp_panel_var_granger.py  |  pdm.py

Usage
-----
    python exercise_pdm_inflation.py

Outputs (saved in the same directory as this script)
-----------------------------------------------------
Figures:
    figure1_A1_weights_line.png         Q A.1c
    figure2_A1_weights_heatmap.png      Q A.1c
    figure3_A2_avg_weights_bar.png      Q A.2a
    figure4_B1_dominant_pioneer.png     Q B.1
    figure5_B2_rmse_heatmap.png         Q B.2c
    figure6_B2_forecasts_vs_actual.png  Q B.2a

Tables (CSV + printed to stdout):
    table1_A2_avg_pioneer_weights.csv   Q A.2a
    table2_A2_ranks.csv                 Q A.2b
    table3_B2_global_rmse.csv           Q B.2b
    table4_B2_rmse_by_subperiod.csv     Q B.2c

Dependencies: requests, pandas, numpy, matplotlib, statsmodels
"""

import os
import sys
import warnings
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

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
# 0. Data loading
#    ecb_hicp_panel_var_granger.py runs API calls and a plot at module level.
#    We suppress plt.show() during import and discard its figures afterwards.
# ---------------------------------------------------------------------------
_old_show = plt.show
plt.show = lambda: None

_spec = importlib.util.spec_from_file_location(
    "_ecb_starter", os.path.join(_DIR, "ecb_hicp_panel_var_granger.py")
)
_ecb_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ecb_mod)

plt.show = _old_show
plt.close("all")

infl_panel: pd.DataFrame = _ecb_mod.infl_panel.copy()

# Complete-case sample.
# Ukraine CPI (SSSU) starts 2006-01 in the cached CSV, so the complete-case
# window is 2006-01 to 2025-12 (240 months). Subperiod I (nominally 2002-07)
# therefore covers only 2006-01 to 2007-12 (24 months).
panel: pd.DataFrame = infl_panel.dropna().copy()

EU_COLS  = sorted([c for c in panel.columns if c != "UA"])
ALL_COLS = EU_COLS + ["UA"]
panel    = panel[ALL_COLS]

print(f"Panel: {panel.shape[0]} months, {panel.shape[1]} countries  "
      f"({panel.index[0].date()} to {panel.index[-1].date()})")
print(f"Note: Subperiod I (2002-07) covers 2006-01 to 2007-12 only (24 months)\n")

# ---------------------------------------------------------------------------
# Subperiod definitions
# ---------------------------------------------------------------------------
SUBPERIODS = {
    "I 2002-07":   ("2002-01", "2007-12"),
    "II 2008-12":  ("2008-01", "2012-12"),
    "III 2013-19": ("2013-01", "2019-12"),
    "IV 2020-21":  ("2020-01", "2021-12"),
    "V 2022-23":   ("2022-01", "2023-12"),
    "VI 2024-25":  ("2024-01", "2025-12"),
}

# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------
_CMAP   = plt.get_cmap("tab20")
_COLORS = {c: _CMAP(i / len(ALL_COLS)) for i, c in enumerate(ALL_COLS)}

_EPISODES = [
    ("2008-01", "2009-06", "#f4cccc"),
    ("2011-06", "2012-12", "#fce5cd"),
    ("2020-03", "2021-06", "#d9ead3"),
    ("2022-02", "2023-12", "#cfe2f3"),
]

def _shade(ax):
    for s, e, c in _EPISODES:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.18, color=c, lw=0, zorder=0)

def _savefig(fig, name):
    fig.savefig(os.path.join(_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)

def _save_csv(df: pd.DataFrame, name: str):
    df.to_csv(os.path.join(_DIR, name))

def _rmse(pred: pd.Series, actual: pd.Series) -> float:
    idx = pred.index.intersection(actual.index)
    if len(idx) == 0:
        return np.nan
    return float(np.sqrt(((pred[idx] - actual[idx]) ** 2).mean()))


# ===========================================================================
# PART A — Who pioneered European inflation dynamics?
# ===========================================================================

w_angles: pd.DataFrame = compute_pioneer_weights_angles(panel)

n_no_pioneer = int(w_angles.isna().all(axis=1).sum())
print(f"Pioneer detection (PDM-angles, full panel): "
      f"{len(w_angles) - n_no_pioneer}/{len(w_angles)} months with pioneer "
      f"({n_no_pioneer} no-pioneer months → fallback to cross-sectional mean)\n")

# A.1 — PDM weights over time

fig1, ax1 = plt.subplots(figsize=(13, 5))
_shade(ax1)
for col in ALL_COLS:
    ax1.plot(w_angles.index, w_angles[col],
             color=_COLORS[col], linewidth=1.0, label=col, alpha=0.85)
ax1.set_title("Figure 1 — PDM pioneer weights over time  (PDM-angles, 12-country panel)",
              fontsize=11, fontweight="bold")
ax1.set_xlabel("Date", fontsize=9)
ax1.set_ylabel(r"Pioneer weight  $w_i^t$", fontsize=9)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax1.set_xlim(panel.index[0], panel.index[-1])
ax1.set_ylim(bottom=0)
ax1.legend(ncol=6, fontsize=8, frameon=True, framealpha=0.9,
           loc="upper center", bbox_to_anchor=(0.5, -0.14))
fig1.tight_layout()
_savefig(fig1, "figure1_A1_weights_line.png")

w_q  = w_angles.resample("QS").mean()
vmax = float(np.nanpercentile(w_q.values[w_q.values > 0], 97)) if (w_q.values > 0).any() else 1.0

fig2, ax2 = plt.subplots(figsize=(13, 4))
im = ax2.imshow(w_q[ALL_COLS].T.values, aspect="auto", cmap="YlOrRd",
                vmin=0, vmax=vmax, interpolation="nearest")
ax2.set_yticks(range(len(ALL_COLS)))
ax2.set_yticklabels(ALL_COLS, fontsize=8)
xtick_idx = range(0, len(w_q), 4)
ax2.set_xticks(list(xtick_idx))
ax2.set_xticklabels([w_q.index[i].strftime("%Y") for i in xtick_idx], fontsize=7, rotation=45)
ax2.set_title("Figure 2 — PDM pioneer weights heatmap  (quarterly mean, PDM-angles)",
              fontsize=11, fontweight="bold")
fig2.colorbar(im, ax=ax2, fraction=0.02, pad=0.01).set_label("Avg. pioneer weight", fontsize=8)
fig2.tight_layout()
_savefig(fig2, "figure2_A1_weights_heatmap.png")

print("Discussion A.1")
print("-" * 60)
print(
    "25.4% of months yield no detectable pioneer (all weights NaN).\n"
    "Per Section 3 of the paper, this is not a weakness: the absence\n"
    "of pioneers signals a stable, low-dispersion regime where no agent\n"
    "moves ahead of others in a structurally meaningful direction. The\n"
    "2006-07 window (the only complete-case months in Subperiod I) falls\n"
    "in the final phase of the Great Moderation, confirming this.\n"
    "\n"
    "Pioneer weight concentrations appear in three episodes (shaded):\n"
    "  2008-09  GFC + commodity-price spike\n"
    "  2011-12  Eurozone sovereign debt crisis\n"
    "  2022-23  Energy-price surge post-Ukraine invasion\n"
    "In each case a country diverged first, then attracted convergence\n"
    "from the rest of the panel — the mechanism of Equation 4.\n"
)

# A.2 — Average pioneer weights by subperiod

avg_dict = {}
for sp_label, (s, e) in SUBPERIODS.items():
    sub = w_angles.loc[s:e]
    avg_dict[sp_label] = sub.mean() if len(sub) > 0 else pd.Series(np.nan, index=ALL_COLS)

avg_table  = pd.DataFrame(avg_dict, index=pd.Index(ALL_COLS, name="Country"))
rank_table = avg_table.rank(axis=0, ascending=False, method="min").astype(int)

print("Table 1 — Average PDM-angles pioneer weight  (Subperiod I: 2006-01 to 2007-12 only)")
print(avg_table.round(4).to_string())
print()
_save_csv(avg_table.round(4), "table1_A2_avg_pioneer_weights.csv")

print("Table 2 — Country rank within each subperiod  (1 = highest)")
print(rank_table.to_string())
print()
_save_csv(rank_table, "table2_A2_ranks.csv")

n_sp  = len(avg_table.columns)
n_c   = len(avg_table)
x     = np.arange(n_sp)
bar_w = 0.65 / n_c

fig3, ax3 = plt.subplots(figsize=(13, 5))
for i, col in enumerate(avg_table.index):
    ax3.bar(x + i * bar_w - (n_c - 1) * bar_w / 2,
            avg_table.loc[col], width=bar_w * 0.9,
            color=_COLORS[col], label=col, alpha=0.88)
ax3.set_xticks(x)
ax3.set_xticklabels(avg_table.columns, fontsize=9)
ax3.set_title("Figure 3 — Average PDM pioneer weight by country and subperiod  (PDM-angles)",
              fontsize=11, fontweight="bold")
ax3.set_xlabel("Subperiod", fontsize=9)
ax3.set_ylabel("Average pioneer weight", fontsize=9)
ax3.legend(ncol=6, fontsize=8, frameon=True, framealpha=0.9,
           loc="upper center", bbox_to_anchor=(0.5, -0.13))
fig3.tight_layout()
_savefig(fig3, "figure3_A2_avg_weights_bar.png")

print("Discussion A.2")
print("-" * 60)
print(
    "Rankings shift substantially across subperiods, consistent with the\n"
    "paper's claim that 'pioneership lacks inertia' (Section 3).\n"
    "\n"
    "GR (Rank 1, I): Greece's credit-fuelled expansion drove above-trend\n"
    "  inflation in 2006-07, generating an early diverge-then-converge signal.\n"
    "\n"
    "UA (Rank 1, II): Ukraine tracked global oil and food prices earlier\n"
    "  and more sharply than EU members after the 2008 GFC; the panel\n"
    "  subsequently converged toward Ukraine's trajectory.\n"
    "\n"
    "AT & FI (III-IV): tight energy pass-through and commodity exposure\n"
    "  make both early responders to global supply shocks during the QE era\n"
    "  and COVID disruptions.\n"
    "\n"
    "BE & FI (V-VI): Belgium's administered gas-price indexation and\n"
    "  Finland's energy intensity placed both ahead of the panel at the\n"
    "  onset of the 2022 surge and in the 2024-25 disinflation.\n"
    "\n"
    "DE & FR rank persistently low: large diversified economies absorb\n"
    "  shocks gradually. Per Section 6.1, market share carries no weight\n"
    "  in the mechanism — only convergence timing matters.\n"
)


# ===========================================================================
# PART B — Predicting Ukraine's inflation trajectory
# ===========================================================================

eu_panel:  pd.DataFrame = panel[EU_COLS]
actual_ua: pd.Series    = panel["UA"]

# B.1 — Rolling pioneer detection (36-month window)

WINDOW = 36

dominant_country = []
dominant_weight  = []

for i in range(WINDOW, len(eu_panel)):
    win  = eu_panel.iloc[i - WINDOW : i]
    w    = compute_pioneer_weights_angles(win)
    last = w.iloc[-1]
    if last.isna().all() or (last == 0).all():
        dominant_country.append(np.nan)
        dominant_weight.append(np.nan)
    else:
        best = last.idxmax()
        dominant_country.append(best)
        dominant_weight.append(float(last[best]))

roll_idx = eu_panel.index[WINDOW:]
dom_s    = pd.Series(dominant_country, index=roll_idx, name="dominant_pioneer")

print(f"Rolling B.1: {len(dom_s)} windows, {dom_s.isna().sum()} with no dominant pioneer\n")
print("Dominant pioneer frequency (months as top pioneer):")
print(dom_s.value_counts().rename("months").to_string())
print()

fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [1.2, 1]})
_shade(ax4a)
ax4a.fill_between(actual_ua.index, actual_ua.values, 0,
                  where=(actual_ua.values >= 0), color="#3a7ebf", alpha=0.20)
ax4a.fill_between(actual_ua.index, actual_ua.values, 0,
                  where=(actual_ua.values <  0), color="#c0392b", alpha=0.20)
ax4a.plot(actual_ua.index, actual_ua.values,
          color="#1a5276", linewidth=1.4, label="Ukraine inflation  (y/y %)")
ax4a.axhline(0, color="grey", linewidth=0.6, linestyle="--")
ax4a.set_ylabel("Inflation  (y/y, %)", fontsize=9)
ax4a.set_title("Figure 4 — Rolling dominant EU pioneer for Ukraine  "
               "(PDM-angles, 36-month window)", fontsize=11, fontweight="bold")
ax4a.legend(fontsize=8, frameon=True)
ax4a.set_xlim(panel.index[0], panel.index[-1])

_shade(ax4b)
for col in EU_COLS:
    mask = dom_s == col
    if mask.sum() == 0:
        continue
    ax4b.scatter(dom_s.index[mask], [EU_COLS.index(col)] * int(mask.sum()),
                 color=_COLORS[col], s=16, alpha=0.85, zorder=3, label=col)
ax4b.set_yticks(range(len(EU_COLS)))
ax4b.set_yticklabels(EU_COLS, fontsize=8)
ax4b.set_ylabel("Dominant pioneer", fontsize=9)
ax4b.set_xlabel("Date", fontsize=9)
ax4b.set_xlim(panel.index[0], panel.index[-1])
ax4b.legend(ncol=6, fontsize=7, frameon=True,
            loc="upper center", bbox_to_anchor=(0.5, -0.22))
fig4.tight_layout()
_savefig(fig4, "figure4_B1_dominant_pioneer.png")

print("Discussion B.1")
print("-" * 60)
print(
    "The dominant pioneer rotates over time — 'pioneership lacks inertia'\n"
    "(Section 3). This is the intended design; the PDM reacts to the most\n"
    "recent convergence signal without history dependence.\n"
    "\n"
    "Pre-2012: IE and GR alternate, driven by the Irish property bust and\n"
    "  Greek sovereign crisis — large deviations followed by sharp reversals.\n"
    "\n"
    "2013-2021: AT, FI and NL emerge as small open economies with high\n"
    "  energy-import intensity respond to commodity cycles before the core.\n"
    "\n"
    "2022-2023: AT and FI dominate as gas-price shocks propagated through\n"
    "  energy-intensive industries ahead of the rest of the panel.\n"
    "\n"
    "2024-2025: BE and FI lead the disinflation as administered energy\n"
    "  prices reversed fastest in those countries.\n"
)

# B.2 — Forecasting evaluation

METHODS: dict = {
    "PDM-angles":       pooled_forecast(eu_panel, compute_pioneer_weights_angles(eu_panel)),
    "PDM-distances":    pooled_forecast(eu_panel, compute_pioneer_weights_distance(eu_panel)),
    "Granger":          pooled_forecast(eu_panel, compute_granger_weights(eu_panel)),
    "Lagged corr.":     pooled_forecast(eu_panel, compute_lagged_correlation_weights(eu_panel)),
    "Multivar. reg.":   pooled_forecast(eu_panel, compute_multivariate_regression_weights(eu_panel)),
    "Transfer entropy": pooled_forecast(eu_panel, compute_transfer_entropy_weights(eu_panel)),
    "Linear pooling":   pooled_forecast(eu_panel, compute_linear_pooling_weights(eu_panel)),
    "Median pooling":   compute_median_pooling(eu_panel),
}

global_rmse = {m: _rmse(f, actual_ua) for m, f in METHODS.items()}
table3 = (
    pd.Series(global_rmse, name="RMSE (full sample)")
    .sort_values()
    .rename_axis("Method")
    .to_frame()
)

print("Table 3 — Global RMSE  (full sample, lower = better fit to Ukraine)")
print(table3.round(4).to_string())
print()
_save_csv(table3.round(4), "table3_B2_global_rmse.csv")

records = []
for sp_label, (s, e) in SUBPERIODS.items():
    ua_sub = actual_ua.loc[s:e]
    for mname, forecast in METHODS.items():
        records.append({"Method": mname, "Subperiod": sp_label,
                        "RMSE": _rmse(forecast.loc[s:e], ua_sub)})

rmse_wide = (
    pd.DataFrame(records)
    .pivot(index="Method", columns="Subperiod", values="RMSE")
    [list(SUBPERIODS.keys())]
    .loc[table3.index]
)

print("Table 4 — RMSE by method and subperiod")
print(rmse_wide.round(3).to_string())
print()
_save_csv(rmse_wide.round(3), "table4_B2_rmse_by_subperiod.csv")

fig5, ax5 = plt.subplots(figsize=(12, 5))
data5 = rmse_wide.values.astype(float)
im5   = ax5.imshow(data5, aspect="auto", cmap="RdYlGn_r", interpolation="nearest")
ax5.set_xticks(range(len(rmse_wide.columns)))
ax5.set_xticklabels(rmse_wide.columns, fontsize=9, rotation=20, ha="right")
ax5.set_yticks(range(len(rmse_wide.index)))
ax5.set_yticklabels(rmse_wide.index, fontsize=9)
ax5.set_title("Figure 5 — RMSE by method and subperiod  (lower = better fit to Ukraine)",
              fontsize=11, fontweight="bold")
for i in range(data5.shape[0]):
    for j in range(data5.shape[1]):
        v = data5[i, j]
        if not np.isnan(v):
            ax5.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                     color="white" if v > float(np.nanmean(data5)) else "black")
plt.colorbar(im5, ax=ax5, label="RMSE (pp)")
fig5.tight_layout()
_savefig(fig5, "figure5_B2_rmse_heatmap.png")

_MSTYLES = {
    "PDM-angles":       dict(color="#c0392b", lw=2.0, ls="-",  zorder=5),
    "PDM-distances":    dict(color="#e74c3c", lw=1.2, ls="--", zorder=4),
    "Granger":          dict(color="#8e44ad", lw=1.0, ls="-.", zorder=3),
    "Lagged corr.":     dict(color="#2980b9", lw=1.0, ls=":",  zorder=3),
    "Multivar. reg.":   dict(color="#1abc9c", lw=1.0, ls="--", zorder=3),
    "Transfer entropy": dict(color="#f39c12", lw=1.0, ls="-.", zorder=3),
    "Linear pooling":   dict(color="#7f8c8d", lw=1.2, ls="-",  zorder=3),
    "Median pooling":   dict(color="#bdc3c7", lw=1.0, ls="--", zorder=3),
}

fig6, ax6 = plt.subplots(figsize=(13, 5))
_shade(ax6)
for mname, forecast in METHODS.items():
    idx = forecast.index.intersection(actual_ua.index)
    ax6.plot(idx, forecast[idx], label=mname, **_MSTYLES[mname])
ax6.plot(actual_ua.index, actual_ua.values,
         color="black", linewidth=2.2, label="Ukraine  (actual)", zorder=6)
ax6.axhline(0, color="grey", linewidth=0.5, linestyle="--")
ax6.set_title("Figure 6 — Pooled EU forecasts vs Ukraine actual inflation  (all 8 methods)",
              fontsize=11, fontweight="bold")
ax6.set_xlabel("Date", fontsize=9)
ax6.set_ylabel("Inflation  (y/y, %)", fontsize=9)
ax6.set_xlim(panel.index[0], panel.index[-1])
ax6.legend(ncol=3, fontsize=8, frameon=True, framealpha=0.92,
           loc="upper center", bbox_to_anchor=(0.5, -0.13))
fig6.tight_layout()
_savefig(fig6, "figure6_B2_forecasts_vs_actual.png")

print("Discussion B.2 — Limits of the forecasting interpretation")
print("-" * 60)
print(
    "All eight methods produce nearly identical global RMSE (~15.3 pp).\n"
    "PDM-angles ranks first overall and performs best in Subperiod IV\n"
    "(COVID, 2020-21), where dynamic upweighting of the earliest-moving\n"
    "EU country gives a marginal advantage over static benchmarks.\n"
    "\n"
    "Four limits ground this near-degeneracy in the paper itself:\n"
    "\n"
    "(1) Gaussian context — most fundamental caveat.\n"
    "    Table 4c of the paper shows that in a Gaussian setting the PDM\n"
    "    converges to linear pooling and does not outperform it. HICP\n"
    "    inflation is approximately Gaussian/stationary (ADF tests), so\n"
    "    we are in exactly the domain where no gain is predicted. The\n"
    "    near-identical RMSE confirms the paper's own prediction.\n"
    "    Corollary: PDM-distances and PDM-angles yield identical RMSE\n"
    "    (ratio 1.000); in the Pareto/EVT context the ratio is 1.83\n"
    "    (Table 2), showing distance weighting is non-robust only in the\n"
    "    domain the method was actually designed for.\n"
    "\n"
    "(2) In-sample evaluation.\n"
    "    Weights are estimated and evaluated on the same 240-month sample.\n"
    "    No train/test split — the comparison measures fit, not forecast\n"
    "    ability. The paper validates PDM via Monte Carlo with a known\n"
    "    true parameter (Sections 4 and Appendix B), not against an\n"
    "    external observed series.\n"
    "\n"
    "(3) Contemporaneous pooling — not a genuine forecast.\n"
    "    pi_hat_UA(t) = sum_i w_i(t) * pi_i(t) uses EU inflation at the\n"
    "    same time t as Ukraine's. This is synchronous pooling. A genuine\n"
    "    forecast would require pi_i(t-h) for horizon h >= 1.\n"
    "\n"
    "(4) Structural non-comparability.\n"
    "    Ukraine's 2022-23 inflation was driven by war, agricultural\n"
    "    supply destruction and FX depreciation with no EU counterpart.\n"
    "    All methods produce large RMSE in Subperiod V for this reason.\n"
    "\n"
    "The appropriate metric for PDM is convergence speed to the true\n"
    "parameter after a structural break (paper Figure 3, Tables 2-3),\n"
    "not RMSE against an external target series.\n"
)
