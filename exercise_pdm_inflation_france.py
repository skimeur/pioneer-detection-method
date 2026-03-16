#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import matplotlibs
import math
import textwrap
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Import PDM methods from the local repository
# -----------------------------------------------------------------------------
try:
    from pdm import (
        compute_granger_weights,
        compute_lagged_correlation_weights,
        compute_linear_pooling_weights,
        compute_median_pooling,
        compute_multivariate_regression_weights,
        compute_pioneer_weights_angles,
        compute_pioneer_weights_distance,
        compute_transfer_entropy_weights,
        pooled_forecast,
    )
except Exception as exc:
    raise ImportError(
        "Could not import pdm.py. Place this script in the same repository as pdm.py "
        "or ensure that the repository is on PYTHONPATH."
    ) from exc


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
COUNTRIES = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]
TARGET_COUNTRY = "FR"
START = "2000-01-01"
END = "2025-12-31"
ROLLING_WINDOW = 36  # months
OUTPUT_DIR = Path("outputs_pdm_france")

PERIODS: Dict[str, Tuple[str, str]] = {
    "I (2002-07)": ("2002-01-01", "2007-12-31"),
    "II (2008-12)": ("2008-01-01", "2012-12-31"),
    "III (2013-19)": ("2013-01-01", "2019-12-31"),
    "IV (2020-21)": ("2020-01-01", "2021-12-31"),
    "V (2022-23)": ("2022-01-01", "2023-12-31"),
    "VI (2024-25)": ("2024-01-01", "2025-12-31"),
}

METHODS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame | pd.Series]] = {
    "PDM_angles": compute_pioneer_weights_angles,
    "PDM_distance": compute_pioneer_weights_distance,
    "Granger": lambda x: compute_granger_weights(x, maxlag=2),
    "Lagged_correlation": lambda x: compute_lagged_correlation_weights(x, lag=1),
    "Multivariate_regression": lambda x: compute_multivariate_regression_weights(x, lag=1),
    "Transfer_entropy": lambda x: compute_transfer_entropy_weights(x, n_bins=3, lag=1),
    "Linear_pooling": compute_linear_pooling_weights,
    "Median_pooling": compute_median_pooling,
}


# -----------------------------------------------------------------------------
# Data fetching logic adapted from ecb_hicp_panel_var_granger.py
# -----------------------------------------------------------------------------
def fetch_ecb_hicp_inflation_panel(
    countries: Iterable[str],
    start: str = START,
    end: Optional[str] = None,
    item: str = "000000",
    sa: str = "N",
    measure: str = "4",
    variation: str = "ANR",
    freq: str = "M",
    timeout: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch monthly HICP y/y inflation from the ECB Data Portal."""
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
        raise ValueError(f"Unexpected ECB response format. Columns: {list(raw.columns)}")

    country_col = "REF_AREA" if "REF_AREA" in raw.columns else None
    if country_col is None:
        candidates = [c for c in raw.columns if c not in {"TIME_PERIOD", "OBS_VALUE"}]
        if not candidates:
            raise ValueError("Unable to infer the country dimension in ECB response.")
        country_col = candidates[0]

    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"]).dt.to_period("M").dt.to_timestamp("MS")
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")

    panel = (
        raw.pivot_table(index="TIME_PERIOD", columns=country_col, values="OBS_VALUE", aggfunc="last")
        .sort_index()
        .astype(float)
    )
    return panel, raw


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def complete_case_panel(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy().sort_index().dropna(how="any")
    out.index = pd.to_datetime(out.index).to_period("M").to_timestamp("MS")
    return out


def slice_period(df: pd.DataFrame | pd.Series, start: str, end: str) -> pd.DataFrame | pd.Series:
    return df.loc[pd.Timestamp(start): pd.Timestamp(end)]


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    aligned = pd.concat([y_true.rename("actual"), y_pred.rename("pred")], axis=1).dropna()
    if aligned.empty:
        return np.nan
    return float(np.sqrt(np.mean((aligned["pred"] - aligned["actual"]) ** 2)))


def format_rankings_from_table(table: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    ranking_dict: Dict[str, List[str]] = {}
    for col in table.columns:
        top = table[col].sort_values(ascending=False).head(top_n)
        ranking_dict[col] = [f"{idx} ({val:.3f})" for idx, val in top.items()]
    return pd.DataFrame(ranking_dict, index=[f"Top {i}" for i in range(1, top_n + 1)])


@dataclass
class MethodOutput:
    name: str
    pooled: pd.Series
    weights: Optional[pd.DataFrame] = None


# -----------------------------------------------------------------------------
# Part A
# -----------------------------------------------------------------------------
def compute_average_weights_by_period(weights: pd.DataFrame, periods: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    avg = {}
    for name, (start, end) in periods.items():
        sub = slice_period(weights, start, end)
        avg[name] = sub.mean(axis=0, skipna=True)
    out = pd.DataFrame(avg)
    return out.loc[weights.columns]


def plot_part_a_weights(weights: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[1.2, 1.0])

    # Line plot
    for col in weights.columns:
        axes[0].plot(weights.index, weights[col], linewidth=1.2, alpha=0.9, label=col)
    axes[0].set_title("Part A — PDM (angles) pioneer weights over time")
    axes[0].set_ylabel("Weight")
    axes[0].legend(ncol=6, fontsize=8, frameon=False)
    axes[0].grid(alpha=0.25)

    # Heatmap
    heat = weights.fillna(0.0).T
    im = axes[1].imshow(heat.values, aspect="auto", interpolation="nearest")
    axes[1].set_yticks(np.arange(len(heat.index)))
    axes[1].set_yticklabels(heat.index)
    tick_idx = np.linspace(0, len(heat.columns) - 1, 10, dtype=int)
    axes[1].set_xticks(tick_idx)
    axes[1].set_xticklabels([heat.columns[i].strftime("%Y-%m") for i in tick_idx], rotation=45, ha="right")
    axes[1].set_title("Part A — Heatmap of pioneer weights")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Country")
    fig.colorbar(im, ax=axes[1], shrink=0.85, label="Weight")

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Part B
# -----------------------------------------------------------------------------
def rolling_dominant_pioneer(experts: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    rows = []
    if len(experts) < window:
        raise ValueError(f"Need at least {window} observations for rolling analysis.")

    for end_ix in range(window - 1, len(experts)):
        sub = experts.iloc[end_ix - window + 1 : end_ix + 1]
        w = compute_pioneer_weights_angles(sub)
        last = w.iloc[-1]
        if last.isna().all() or float(last.fillna(0).sum()) == 0.0:
            dominant = "No clear pioneer"
            dom_weight = np.nan
        else:
            dominant = str(last.idxmax())
            dom_weight = float(last.max())
        rows.append(
            {
                "date": experts.index[end_ix],
                "dominant_pioneer": dominant,
                "dominant_weight": dom_weight,
            }
        )

    out = pd.DataFrame(rows).set_index("date")
    return out


def plot_dominant_pioneer(dominant_df: pd.DataFrame, outpath: Path) -> None:
    mapping = {name: i for i, name in enumerate(pd.unique(dominant_df["dominant_pioneer"]))}
    y = dominant_df["dominant_pioneer"].map(mapping)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(dominant_df.index, y, s=22)
    ax.set_yticks(list(mapping.values()))
    ax.set_yticklabels(list(mapping.keys()))
    ax.set_title(f"Part B — Rolling dominant pioneer for {TARGET_COUNTRY} (window = {ROLLING_WINDOW} months)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Dominant pioneer")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Forecasting helpers
# -----------------------------------------------------------------------------
def compute_method_output(method_name: str, experts: pd.DataFrame) -> MethodOutput:
    func = METHODS[method_name]

    if method_name == "Median_pooling":
        pooled = func(experts)  # type: ignore[misc]
        assert isinstance(pooled, pd.Series)
        return MethodOutput(name=method_name, pooled=pooled.rename(method_name), weights=None)

    result = func(experts)
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"Method {method_name} should return a DataFrame of weights.")
    pooled = pooled_forecast(experts, result).rename(method_name)
    return MethodOutput(name=method_name, pooled=pooled, weights=result)


def compute_all_methods(experts: pd.DataFrame) -> Dict[str, MethodOutput]:
    outputs: Dict[str, MethodOutput] = {}
    for name in METHODS:
        try:
            outputs[name] = compute_method_output(name, experts)
        except Exception as exc:
            print(f"[WARN] Method {name} failed and will be skipped: {exc}")
    return outputs


def compute_rmse_table(outputs: Dict[str, MethodOutput], actual: pd.Series, periods: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    rows = {}
    for name, out in outputs.items():
        row = {"Full sample": rmse(actual, out.pooled)}
        for period_name, (start, end) in periods.items():
            row[period_name] = rmse(slice_period(actual, start, end), slice_period(out.pooled, start, end))
        rows[name] = row
    table = pd.DataFrame(rows).T.sort_values("Full sample")
    return table


def plot_forecasts_vs_actual(outputs: Dict[str, MethodOutput], actual: pd.Series, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(actual.index, actual, linewidth=2.3, label=f"Actual {TARGET_COUNTRY}")
    for name, out in outputs.items():
        ax.plot(out.pooled.index, out.pooled, linewidth=1.1, alpha=0.85, label=name)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(f"Part B — Pooled estimates tracking {TARGET_COUNTRY} inflation")
    ax.set_xlabel("Time")
    ax.set_ylabel("Inflation (y/y, %)")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Automatic discussion text
# -----------------------------------------------------------------------------
def build_discussion_text(
    avg_weights: pd.DataFrame,
    rankings: pd.DataFrame,
    weights: pd.DataFrame,
    rmse_table: pd.DataFrame,
    dominant_df: pd.DataFrame,
) -> str:
    stable_period = avg_weights["I (2002-07)"].sort_values(ascending=False)
    crisis_period = avg_weights["V (2022-23)"].sort_values(ascending=False)
    best_method = rmse_table.index[0] if not rmse_table.empty else "N/A"
    low_inflation_mean_max = float(weights.loc["2002-01-01":"2007-12-31"].fillna(0).max(axis=1).mean())
    dominant_changes = int((dominant_df["dominant_pioneer"] != dominant_df["dominant_pioneer"].shift(1)).sum())

    txt = f"""
    RESULTS DISCUSSION (AUTO-GENERATED DRAFT)
    ========================================

    Part A — Pioneer rankings
    -------------------------
    During the Great Moderation subperiod (2002–2007), the average maximum PDM weight
    across countries is {low_inflation_mean_max:.3f}. If this value is modest, that is
    consistent with the theory of the PDM: when inflation dispersion is low and the system
    is relatively stable, there should be fewer strong pioneers because countries move more
    synchronously and there is little directional convergence to detect.

    The top countries in subperiod I are:
      1. {stable_period.index[0]} ({stable_period.iloc[0]:.3f})
      2. {stable_period.index[1]} ({stable_period.iloc[1]:.3f})
      3. {stable_period.index[2]} ({stable_period.iloc[2]:.3f})

    The top countries in the energy-shock subperiod V (2022–2023) are:
      1. {crisis_period.index[0]} ({crisis_period.iloc[0]:.3f})
      2. {crisis_period.index[1]} ({crisis_period.iloc[1]:.3f})
      3. {crisis_period.index[2]} ({crisis_period.iloc[2]:.3f})

    A shift in rankings between calm periods and shock periods is economically plausible.
    Countries can become pioneers when their inflation reacts earlier to common shocks due
    to differences in energy mix, external exposure, trade openness, sensitivity to food and
    energy prices, and domestic pass-through mechanisms.

    Part B — France as target country
    ---------------------------------
    The rolling dominant pioneer changes {dominant_changes} times over the sample windows.
    If dominant-pioneer identity changes materially across subperiods, this suggests that the
    country leading inflation dynamics for France is not fixed but depends on the nature of
    the shock.

    The lowest full-sample RMSE is obtained by: {best_method}.

    Important caveat: the exercise itself warns that the PDM is not a forecasting model in
    the strict sense. Here, a low RMSE only means that the weighted combination of other
    euro-area inflation series tracks France reasonably well. This should therefore be read as
    a convergence/nowcasting exercise rather than as evidence of structural forecasting
    superiority.

    Suggested economic interpretation
    ---------------------------------
    - In low-dispersion periods, pioneers should be weak or intermittent.
    - In crisis periods, pioneers may emerge more clearly because shocks hit countries at
      different speeds.
    - France may be tracked well by countries with similar energy exposure, similar policy
      transmission, or strong trade and financial integration with the French economy.
    """
    return textwrap.dedent(txt).strip() + "\n"


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    print("=" * 80)
    print("Pioneer Detection Method — Inflation exercise (France adaptation)")
    print("=" * 80)
    print(f"Target country in Part B: {TARGET_COUNTRY}")
    print(f"Expert countries in Part B: {[c for c in COUNTRIES if c != TARGET_COUNTRY]}")
    print()

    # --- Data
    infl_panel, _ = fetch_ecb_hicp_inflation_panel(COUNTRIES, start=START, end=END)
    panel = complete_case_panel(infl_panel)

    panel.to_csv(OUTPUT_DIR / "inflation_panel_complete_case.csv")
    print(f"Complete-case sample: {panel.index.min().date()} to {panel.index.max().date()} ({len(panel)} months)")
    print(f"Saved panel to: {OUTPUT_DIR / 'inflation_panel_complete_case.csv'}")
    print()

    # --- Part A
    print("[Part A] Computing PDM angle weights on full panel...")
    w_angles = compute_pioneer_weights_angles(panel)
    avg_weights = compute_average_weights_by_period(w_angles, PERIODS)
    rankings = format_rankings_from_table(avg_weights, top_n=3)

    avg_weights.round(4).to_csv(OUTPUT_DIR / "part_a_average_pioneer_weights.csv")
    rankings.to_csv(OUTPUT_DIR / "part_a_rankings_top3.csv")
    w_angles.to_csv(OUTPUT_DIR / "part_a_weights_timeseries.csv")
    plot_part_a_weights(w_angles, OUTPUT_DIR / "part_a_pioneer_weights.png")

    print("Average pioneer weights by subperiod:")
    print(avg_weights.round(4).to_string())
    print()
    print("Top-3 rankings by subperiod:")
    print(rankings.to_string())
    print()

    # --- Part B
    print(f"[Part B] Predicting {TARGET_COUNTRY} using the other euro-area countries...")
    actual_target = panel[TARGET_COUNTRY].rename(f"Actual_{TARGET_COUNTRY}")
    experts = panel.drop(columns=[TARGET_COUNTRY])

    dominant_df = rolling_dominant_pioneer(experts, window=ROLLING_WINDOW)
    dominant_df.to_csv(OUTPUT_DIR / "part_b_rolling_dominant_pioneer.csv")
    plot_dominant_pioneer(dominant_df, OUTPUT_DIR / "part_b_dominant_pioneer.png")

    outputs = compute_all_methods(experts)
    pooled_df = pd.concat([out.pooled for out in outputs.values()], axis=1)
    pooled_df.to_csv(OUTPUT_DIR / "part_b_pooled_series.csv")
    plot_forecasts_vs_actual(outputs, actual_target, OUTPUT_DIR / "part_b_forecasts_vs_actual.png")

    rmse_table = compute_rmse_table(outputs, actual_target, PERIODS)
    rmse_table.round(4).to_csv(OUTPUT_DIR / "part_b_rmse_by_method.csv")

    print("RMSE by method:")
    print(rmse_table.round(4).to_string())
    print()

    discussion = build_discussion_text(avg_weights, rankings, w_angles, rmse_table, dominant_df)
    (OUTPUT_DIR / "discussion_draft.txt").write_text(discussion, encoding="utf-8")

    # Optional method-specific weights export
    weights_dir = OUTPUT_DIR / "method_weights"
    ensure_output_dir(weights_dir)
    for name, out in outputs.items():
        if out.weights is not None:
            out.weights.to_csv(weights_dir / f"weights_{name}.csv")

    # Summary markdown for GitHub / PR body reuse
    summary_md = f"""
    # PDM Inflation Exercise — France adaptation

    This run adapts the original exercise by replacing Ukraine with France as the target
    country in Part B.

    - Part A panel: {', '.join(COUNTRIES)}
    - Part B target: {TARGET_COUNTRY}
    - Part B experts: {', '.join(experts.columns)}
    - Rolling window: {ROLLING_WINDOW} months

    ## Files generated
    - `inflation_panel_complete_case.csv`
    - `part_a_average_pioneer_weights.csv`
    - `part_a_rankings_top3.csv`
    - `part_a_pioneer_weights.png`
    - `part_b_rolling_dominant_pioneer.csv`
    - `part_b_dominant_pioneer.png`
    - `part_b_pooled_series.csv`
    - `part_b_forecasts_vs_actual.png`
    - `part_b_rmse_by_method.csv`
    - `discussion_draft.txt`

    ## Best RMSE method on full sample
    {rmse_table.index[0] if not rmse_table.empty else 'N/A'}
    """
    (OUTPUT_DIR / "README_results.md").write_text(textwrap.dedent(summary_md).strip() + "\n", encoding="utf-8")

    print(f"All outputs saved in: {OUTPUT_DIR.resolve()}")
    print(f"Main script discussion draft: {OUTPUT_DIR / 'discussion_draft.txt'}")
    print("Done.")


if __name__ == "__main__":
    main()
