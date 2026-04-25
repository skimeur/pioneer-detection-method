#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise: Applying the Pioneer Detection Method to European Inflation Dynamics
==============================================================================
Author : [Your Name]
Based on : Vansteenberghe, E. (2025), The Geneva Papers on Risk and Insurance
Repo    : https://github.com/skimeur/pioneer-detection-method

Usage:
    python exercise_pdm_inflation.py

Outputs
-------
figures/fig1_pioneer_weights_lines.png
figures/fig2_pioneer_weights_heatmap.png
figures/fig3_dominant_pioneer.png
tables/avg_weights_by_subperiod.csv
tables/rmse_by_method.csv
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # ← pas de fenêtre interactive, sauvegarde directe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.makedirs(os.path.join(SCRIPT_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, "tables"),  exist_ok=True)

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
# 1. Load data  –  neutralise plt.show() dans le module source
# ---------------------------------------------------------------------------
print("=" * 65)
print("Step 1 – Loading inflation panel …")
print("=" * 65)

import matplotlib.pyplot as _plt
_plt.show = lambda *a, **kw: None          # évite le blocage interactif

import ecb_hicp_panel_var_granger as _src
infl_panel: pd.DataFrame = _src.infl_panel

TARGET = getattr(_src, "TARGET_COUNTRY", "FR")
print(f"  Columns : {list(infl_panel.columns)}")
print(f"  Target  : {TARGET}")

panel = infl_panel.dropna()
print(f"  After dropna : {panel.shape[0]} observations\n")

# ---------------------------------------------------------------------------
# Subperiods
# ---------------------------------------------------------------------------
PERIODS = {
    "I (2002-07)"   : ("2002-01", "2007-12"),
    "II (2008-12)"  : ("2008-01", "2012-12"),
    "III (2013-19)" : ("2013-01", "2019-12"),
    "IV (2020-21)"  : ("2020-01", "2021-12"),
    "V (2022-23)"   : ("2022-01", "2023-12"),
    "VI (2024-25)"  : ("2024-01", "2025-12"),
}
COLORS = ["#4e79a7","#f28e2b","#59a14f","#e15759","#b07aa1","#76b7b2"]

def _slice(df, s, e):
    return df.loc[pd.Timestamp(s): pd.Timestamp(e) + pd.offsets.MonthEnd(0)]

# ===========================================================================
# PART A
# ===========================================================================
print("=" * 65)
print("PART A – Pioneer weights, full panel")
print("=" * 65)

print("\n[A.1] compute_pioneer_weights_angles …")
w_angles = compute_pioneer_weights_angles(panel)

# Figure 1 – lines
fig, ax = plt.subplots(figsize=(14, 5))
for col in w_angles.columns:
    ax.plot(w_angles.index, w_angles[col].fillna(0), linewidth=1.1, label=col)
for (name,(s,e)), c in zip(PERIODS.items(), COLORS):
    try: ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.08, color=c)
    except: pass
ax.set_title("Figure 1 – Pioneer Weights over Time [PDM angles]", fontsize=12)
ax.set_xlabel("Date"); ax.set_ylabel("Pioneer weight")
ax.legend(ncol=6, fontsize=8, loc="upper left")
plt.tight_layout()
p1 = os.path.join(SCRIPT_DIR, "figures", "fig1_pioneer_weights_lines.png")
fig.savefig(p1, dpi=150); plt.close(fig)
print(f"  → {p1}")

# Figure 2 – heatmap
fig, ax = plt.subplots(figsize=(14, 4))
hm = w_angles.fillna(0).T.values
im = ax.imshow(hm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(hm.max(),1e-9))
ax.set_yticks(range(len(w_angles.columns)))
ax.set_yticklabels(w_angles.columns, fontsize=9)
idx = w_angles.index
ypos = [np.where(idx.year==y)[0][0] for y in idx.year.unique() if len(np.where(idx.year==y)[0])]
ax.set_xticks(ypos); ax.set_xticklabels(idx.year.unique(), rotation=45, ha="right", fontsize=7)
plt.colorbar(im, ax=ax, label="Pioneer weight")
ax.set_title("Figure 2 – Heatmap of Pioneer Weights [PDM angles]", fontsize=12)
plt.tight_layout()
p2 = os.path.join(SCRIPT_DIR, "figures", "fig2_pioneer_weights_heatmap.png")
fig.savefig(p2, dpi=150); plt.close(fig)
print(f"  → {p2}")

# A.2 – average by subperiod
print("\n[A.2] Average weights by subperiod …")
avg_table = pd.DataFrame({n: _slice(w_angles,s,e).fillna(0).mean()
                           for n,(s,e) in PERIODS.items()})
avg_table.index.name = "Country"
print(avg_table.round(4).to_string())
csv1 = os.path.join(SCRIPT_DIR, "tables", "avg_weights_by_subperiod.csv")
avg_table.round(4).to_csv(csv1); print(f"\n  → {csv1}")

print("\n  Rankings:")
for col in avg_table.columns:
    top = avg_table[col].sort_values(ascending=False).head(5)
    print(f"  {col}: " + ", ".join(f"{c}({v:.3f})" for c,v in top.items()))

# ===========================================================================
# PART B
# ===========================================================================
print("\n" + "=" * 65)
print(f"PART B – Other EU countries as experts, {TARGET} as target")
print("=" * 65)

others      = [c for c in panel.columns if c != TARGET]
other_panel = panel[others]
actual      = panel[TARGET]

# B.1 – rolling dominant pioneer
print("\n[B.1] Rolling PDM 36-month …")
WINDOW = 36
dom_list, dom_dates = [], []
for i in range(WINDOW, len(other_panel)):
    win = other_panel.iloc[i-WINDOW:i]
    try:
        dom = compute_pioneer_weights_angles(win).fillna(0).mean().idxmax()
    except:
        dom = np.nan
    dom_list.append(dom); dom_dates.append(other_panel.index[i])

dom_series = pd.Series(dom_list, index=dom_dates)

fig, ax = plt.subplots(figsize=(14, 4))
c2i = {c:i for i,c in enumerate(others)}
yv  = dom_series.map(c2i)
ax.scatter(dom_series.index, yv, s=10, c=yv, cmap="tab20", zorder=3)
for (name,(s,e)), c in zip(PERIODS.items(), COLORS):
    try: ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.10, color=c)
    except: pass
ax.set_yticks(range(len(others))); ax.set_yticklabels(others, fontsize=9)
ax.set_title(f"Figure 3 – Dominant Pioneer for {TARGET} [rolling {WINDOW}m PDM]", fontsize=12)
ax.set_xlabel("Date"); ax.grid(axis="x", linestyle="--", alpha=0.3)
patches = [mpatches.Patch(color=COLORS[i], alpha=0.5, label=n) for i,n in enumerate(PERIODS)]
ax.legend(handles=patches, fontsize=7, loc="lower right", ncol=3)
plt.tight_layout()
p3 = os.path.join(SCRIPT_DIR, "figures", "fig3_dominant_pioneer.png")
fig.savefig(p3, dpi=150); plt.close(fig)
print(f"  → {p3}")

print("\n  Dominant pioneer per subperiod:")
for name,(s,e) in PERIODS.items():
    sub = _slice(dom_series,s,e).dropna()
    if sub.empty: continue
    top = sub.value_counts().idxmax()
    pct = sub.value_counts().max()/len(sub)*100
    print(f"  {name}: {top} ({pct:.0f}%)")

# B.2 – RMSE
print(f"\n[B.2] RMSE (target={TARGET}) …")

def rmse(pred, actual):
    p,a = pred.align(actual, join="inner")
    return float(np.sqrt(((p-a)**2).mean()))

METHODS = {
    "PDM angles"    : lambda df: compute_pioneer_weights_angles(df),
    "PDM distances" : lambda df: compute_pioneer_weights_distance(df),
    "Granger"       : lambda df: compute_granger_weights(df),
    "Lagged Corr"   : lambda df: compute_lagged_correlation_weights(df),
    "Multivar Reg"  : lambda df: compute_multivariate_regression_weights(df),
    "Transfer Entr" : lambda df: compute_transfer_entropy_weights(df),
    "Linear Pool"   : lambda df: compute_linear_pooling_weights(df),
}

rows = {}
for mname, fn in METHODS.items():
    print(f"  [{mname}] …", end=" ", flush=True)
    try:
        fc = pooled_forecast(other_panel, fn(other_panel))
        row = {"Overall": rmse(fc, actual)}
        for n,(s,e) in PERIODS.items():
            row[n] = rmse(_slice(fc,s,e), _slice(actual,s,e))
        rows[mname] = row
        print(f"RMSE={row['Overall']:.4f}")
    except Exception as ex:
        print(f"ERROR – {ex}")

print("  [Median Pool] …", end=" ", flush=True)
try:
    fc = compute_median_pooling(other_panel)
    row = {"Overall": rmse(fc, actual)}
    for n,(s,e) in PERIODS.items():
        row[n] = rmse(_slice(fc,s,e), _slice(actual,s,e))
    rows["Median Pool"] = row
    print(f"RMSE={row['Overall']:.4f}")
except Exception as ex:
    print(f"ERROR – {ex}")

rmse_table = pd.DataFrame(rows).T.reindex(columns=["Overall"]+list(PERIODS.keys()))
rmse_table.index.name = "Method"
print("\n" + rmse_table.round(4).to_string())
csv2 = os.path.join(SCRIPT_DIR, "tables", "rmse_by_method.csv")
rmse_table.round(4).to_csv(csv2); print(f"\n  → {csv2}")
print(f"\n  Best method: {rmse_table['Overall'].idxmin()}  "
      f"(RMSE={rmse_table['Overall'].min():.4f})")

print("\n" + "=" * 65)
print("✅ Done.  figures/ and tables/ generated.")
print("=" * 65)
