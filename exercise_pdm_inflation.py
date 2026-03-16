#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise: Applying the Pioneer Detection Method to European Inflation Dynamics.

This script is designed to be placed inside the GitHub repository:
https://github.com/skimeur/pioneer-detection-method

"""

from __future__ import annotations

from collections import OrderedDict
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

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


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

START = "2000-01"
END = "2025-12"
COUNTRIES = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]
TARGET = "UA"
ROLLING_WINDOW = 24  # explicit assumption for Part B.1
OUTPUT_DIR = Path("outputs_pdm_inflation")

PERIODS: "OrderedDict[str, Tuple[str, str]]" = OrderedDict(
    {
        "I (2002-07)": ("2002-01", "2007-12"),
        "II (2008-12)": ("2008-01", "2012-12"),
        "III (2013-19)": ("2013-01", "2019-12"),
        "IV (2020-21)": ("2020-01", "2021-12"),
        "V (2022-23)": ("2022-01", "2023-12"),
        "VI (2024-25)": ("2024-01", "2025-12"),
    }
)


# -----------------------------------------------------------------------------
# Data fetching logic (adapted from ecb_hicp_panel_var_granger.py)
# -----------------------------------------------------------------------------


def fetch_ecb_hicp_inflation_panel(
    countries: Iterable[str],
    start: str = START,
    end: str | None = END,
    item: str = "000000",  # headline HICP
    sa: str = "N",         # non-seasonally adjusted
    measure: str = "4",    # rate measure in the ECB ICP key
    variation: str = "ANR",  # annual rate of change
    freq: str = "M",
    timeout: int = 60,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch monthly HICP inflation (y/y, %) from the ECB Data Portal."""
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
        for candidate in ["GEO", "LOCATION", "COUNTRY", "REF_AREA"]:
            if candidate in raw.columns:
                country_col = candidate
                break
    if country_col is None:
        standard = {"TIME_PERIOD", "OBS_VALUE", "OBS_STATUS", "OBS_CONF", "UNIT_MULT", "DECIMALS"}
        nonstandard = [c for c in raw.columns if c not in standard]
        if not nonstandard:
            raise ValueError("Could not infer the ECB country column.")
        country_col = nonstandard[0]

    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"])
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")

    panel = (
        raw.pivot_table(index="TIME_PERIOD", columns=country_col, values="OBS_VALUE", aggfunc="last")
        .sort_index()
    )
    panel.index = panel.index.to_period("M").to_timestamp(how="start")
    panel.index.name = "date"

    return panel, raw



def fetch_ukraine_cpi_prev_month_raw(
    start: str = START,
    end: str = END,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch the raw Ukraine CPI series (previous month = 100) from SSSU SDMX v3."""
    base = "https://stat.gov.ua/sdmx/workspaces/default:integration/registry/sdmx/3.0/data"
    agency = "SSSU"
    flow = "DF_PRICE_CHANGE_CONSUMER_GOODS_SERVICE"
    version = "~"
    key = "INDEX_CONSUMPRICE.PREV_MONTH.UA00000000000000000.0.M"
    url = f"{base}/dataflow/{agency}/{flow}/{version}/{key}"
    params = {"c[TIME_PERIOD]": f"ge:{start}+le:{end}"}
    headers = {
        "Accept": "application/vnd.sdmx.data+csv;version=2.0.0;labels=id;timeFormat=normalized;keys=both",
        "User-Agent": "Mozilla/5.0",
    }

    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.text), dtype=str)

    # Keep true monthly observations only
    raw = raw.loc[
        raw["TIME_PERIOD"].astype(str).str.match(r"^\d{4}-M\d{2}$", na=False)
        & raw["OBS_VALUE"].notna()
    ].copy()
    return raw



def ua_raw_to_monthly_series(ua_raw: pd.DataFrame) -> pd.Series:
    """Convert raw SSSU output to a clean monthly series indexed at month start."""
    if "TIME_PERIOD" not in ua_raw.columns or "OBS_VALUE" not in ua_raw.columns:
        raise ValueError(f"ua_raw must contain TIME_PERIOD and OBS_VALUE. Columns: {list(ua_raw.columns)}")

    s = ua_raw[["TIME_PERIOD", "OBS_VALUE"]].copy()
    s["TIME_PERIOD"] = s["TIME_PERIOD"].astype(str).str.strip()
    s = s[s["TIME_PERIOD"].str.match(r"^\d{4}-M\d{2}$", na=False)]
    s["TIME_PERIOD"] = pd.to_datetime(
        s["TIME_PERIOD"].str.replace(r"^(\d{4})-M(\d{2})$", r"\1-\2-01", regex=True),
        errors="coerce",
    )
    s["OBS_VALUE"] = pd.to_numeric(s["OBS_VALUE"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    s = s.dropna(subset=["TIME_PERIOD", "OBS_VALUE"]).sort_values("TIME_PERIOD")

    out = s.set_index("TIME_PERIOD")["OBS_VALUE"].rename("UA_IDX_PREV_MONTH_100")
    out = out.groupby(level=0).last()
    out.index = out.index.to_period("M").to_timestamp(how="start")
    out.index.name = "date"
    return out



def cpi_prev_month_index_to_yoy_inflation(idx_prev_month_100: pd.Series) -> pd.Series:
    """Convert previous-month CPI index to year-on-year inflation in percent."""
    monthly_factor = (idx_prev_month_100 / 100.0).astype(float)
    yoy_factor = monthly_factor.rolling(12).apply(np.prod, raw=True)
    return ((yoy_factor - 1.0) * 100.0).rename(TARGET)



def build_inflation_panel(start: str = START, end: str = END) -> pd.DataFrame:
    """Build the final monthly inflation panel with 11 EU countries + Ukraine."""
    ecb_panel, _ = fetch_ecb_hicp_inflation_panel(countries=COUNTRIES, start=start, end=end)

    ua_raw = fetch_ukraine_cpi_prev_month_raw(start=start, end=end)
    ua_idx = ua_raw_to_monthly_series(ua_raw).loc[f"{start}-01":f"{end}-01"]
    ua_yoy = cpi_prev_month_index_to_yoy_inflation(ua_idx)
    ua_yoy.index = ua_yoy.index.to_period("M").to_timestamp(how="start")
    ua_yoy.index.name = "date"

    infl_panel = ecb_panel.join(ua_yoy, how="left").sort_index()
    infl_panel.index = pd.to_datetime(infl_panel.index).to_period("M").to_timestamp(how="start")
    infl_panel.index.name = "date"
    return infl_panel


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def rmse(actual: pd.Series, estimate: pd.Series) -> float:
    valid = pd.concat([actual.rename("actual"), estimate.rename("estimate")], axis=1).dropna()
    if valid.empty:
        return np.nan
    return float(np.sqrt(((valid["estimate"] - valid["actual"]) ** 2).mean()))



def save_dataframe(df: pd.DataFrame, name: str) -> Path:
    path = OUTPUT_DIR / name
    df.to_csv(path)
    return path



def describe_nonzero_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """Summarise when each country receives non-zero pioneer weight."""
    mask = weights.fillna(0.0) > 0.0
    rows = []
    for col in weights.columns:
        nz = mask[col]
        nonzero_dates = weights.index[nz]
        avg_nonzero = weights.loc[nz, col].mean() if nz.any() else np.nan
        rows.append(
            {
                "country": col,
                "nonzero_months": int(nz.sum()),
                "share_nonzero_months": float(nz.mean()),
                "first_nonzero": nonzero_dates.min().strftime("%Y-%m") if nz.any() else "",
                "last_nonzero": nonzero_dates.max().strftime("%Y-%m") if nz.any() else "",
                "avg_weight_when_nonzero": avg_nonzero,
                "avg_weight_full_sample": float(weights[col].fillna(0.0).mean()),
            }
        )
    out = pd.DataFrame(rows).set_index("country")
    out = out.sort_values(["share_nonzero_months", "avg_weight_full_sample"], ascending=False)
    return out



def average_weights_by_period(weights: pd.DataFrame, periods: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    data = {}
    for name, (start, end) in periods.items():
        sub = weights.loc[start:end]
        data[name] = sub.mean()
    out = pd.DataFrame(data)
    return out.sort_index()



def rank_table_from_values(values: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    ranks = values.rank(axis=0, ascending=ascending, method="min")
    return ranks.astype("Int64")



def max_weight_summary(weights: pd.DataFrame, periods: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for name, (start, end) in periods.items():
        sub = weights.loc[start:end].fillna(0.0)
        rows.append(
            {
                "period": name,
                "share_months_with_any_pioneer": float((sub.sum(axis=1) > 0).mean()),
                "avg_max_pioneer_weight": float(sub.max(axis=1).mean()),
                "median_max_pioneer_weight": float(sub.max(axis=1).median()),
            }
        )
    return pd.DataFrame(rows).set_index("period")


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_panel(panel: pd.DataFrame, filename: str = "inflation_panel.png") -> Path:
    fig, ax = plt.subplots(figsize=(13, 6))
    for country in panel.columns:
        ax.plot(panel.index, panel[country], label=country, linewidth=1)
    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.set_title("Inflation panel: ECB HICP + Ukraine CPI (y/y, %)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Inflation (y/y, %)")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path



def plot_weight_heatmap(weights: pd.DataFrame, filename: str = "partA_pdm_angles_heatmap.png") -> Path:
    data = weights.fillna(0.0).T.values
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data, aspect="auto", interpolation="nearest")
    ax.set_title("Part A — PDM angle weights over time")
    ax.set_yticks(np.arange(len(weights.columns)))
    ax.set_yticklabels(weights.columns)

    tick_idx = np.linspace(0, len(weights.index) - 1, min(10, len(weights.index)), dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([weights.index[i].strftime("%Y-%m") for i in tick_idx], rotation=45, ha="right")
    ax.set_xlabel("Date")
    ax.set_ylabel("Country")
    fig.colorbar(im, ax=ax, label="Pioneer weight")
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path



def plot_top_weight_lines(weights: pd.DataFrame, filename: str = "partA_top_pioneer_lines.png", top_n: int = 6) -> Path:
    top_cols = weights.fillna(0.0).mean().sort_values(ascending=False).head(top_n).index.tolist()
    fig, ax = plt.subplots(figsize=(13, 6))
    for col in top_cols:
        ax.plot(weights.index, weights[col].fillna(0.0), label=col, linewidth=1.2)
    ax.set_title(f"Part A — Top {top_n} countries by average PDM angle weight")
    ax.set_xlabel("Date")
    ax.set_ylabel("Pioneer weight")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path



def plot_dominant_pioneer(dominant: pd.Series, filename: str = "partB_dominant_pioneer.png") -> Path:
    categories = sorted([c for c in dominant.dropna().unique()])
    mapping = {cat: i for i, cat in enumerate(categories)}
    y = dominant.map(mapping)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(dominant.index, y, drawstyle="steps-post", linewidth=1.5)
    ax.set_title(f"Part B — Dominant pioneer for Ukraine (rolling {ROLLING_WINDOW}-month window)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Country")
    ax.set_yticks(list(mapping.values()))
    ax.set_yticklabels(list(mapping.keys()))
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# Part B-specific logic
# -----------------------------------------------------------------------------


def rolling_pairwise_target_pioneer(
    panel: pd.DataFrame,
    target_col: str = TARGET,
    window: int = ROLLING_WINDOW,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    For each rolling window and each EU country, compute the final-window PDM angle
    weight in the 2-country panel [country, target].
    """
    expert_cols = [c for c in panel.columns if c != target_col]
    end_dates = panel.index[window - 1 :]
    scores = []

    for end_pos in range(window - 1, len(panel)):
        window_df = panel.iloc[end_pos - window + 1 : end_pos + 1]
        row = {}
        for country in expert_cols:
            pair = window_df[[country, target_col]].dropna()
            if len(pair) < 3:
                row[country] = np.nan
                continue
            w_pair = compute_pioneer_weights_angles(pair)
            score = w_pair[country].iloc[-1]
            row[country] = 0.0 if pd.isna(score) else float(score)
        scores.append(row)

    score_df = pd.DataFrame(scores, index=end_dates)
    score_df.index.name = "date"

    dominant = score_df.idxmax(axis=1)
    max_score = score_df.max(axis=1)
    dominant[max_score.fillna(0.0) <= 0.0] = np.nan
    dominant.name = "dominant_pioneer"
    return score_df, dominant



def compute_method_forecasts(eu_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute pooled Ukraine estimates using all required methods."""
    out = pd.DataFrame(index=eu_panel.index)

    method_weights = {
        "PDM_angles": compute_pioneer_weights_angles(eu_panel),
        "PDM_distance": compute_pioneer_weights_distance(eu_panel),
        "Granger": compute_granger_weights(eu_panel),
        "Lagged_correlation": compute_lagged_correlation_weights(eu_panel),
        "Multivariate_regression": compute_multivariate_regression_weights(eu_panel),
        "Transfer_entropy": compute_transfer_entropy_weights(eu_panel),
        "Linear_pooling": compute_linear_pooling_weights(eu_panel),
    }

    for name, weights in method_weights.items():
        out[name] = pooled_forecast(eu_panel, weights)

    out["Median_pooling"] = compute_median_pooling(eu_panel)
    return out



def compute_rmse_table(
    actual: pd.Series,
    forecasts: pd.DataFrame,
    periods: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    rows = {}
    for method in forecasts.columns:
        rows[method] = {"Full sample": rmse(actual, forecasts[method])}
        for period_name, (start, end) in periods.items():
            rows[method][period_name] = rmse(actual.loc[start:end], forecasts[method].loc[start:end])
    out = pd.DataFrame(rows).T
    return out.sort_values("Full sample")


# -----------------------------------------------------------------------------
# Automatic text note
# -----------------------------------------------------------------------------


def build_discussion_note(
    max_weight_stats: pd.DataFrame,
    avg_weights: pd.DataFrame,
    rmse_table: pd.DataFrame,
    dominant: pd.Series,
) -> str:
    best_method = rmse_table["Full sample"].idxmin()
    best_rmse = rmse_table.loc[best_method, "Full sample"]

    lines = []
    lines.append("Short discussion of results")
    lines.append("=" * 28)
    lines.append("")

    lines.append("Part A — Economic interpretation draft")
    lines.append("-------------------------------------")
    period_i = max_weight_stats.loc["I (2002-07)"]
    if period_i["share_months_with_any_pioneer"] < 0.50:
        lines.append(
            "During 2002-2007, pioneer signals appear relatively weak. This is consistent with the "
            "Great Moderation context described in the exercise: inflation dispersion was low, so the PDM "
            "has fewer episodes in which one country clearly moves first and others subsequently converge."
        )
    else:
        lines.append(
            "During 2002-2007, pioneers still appear in a non-negligible share of months, but the signals "
            "should be interpreted cautiously because low dispersion makes strong directional convergence harder to identify."
        )

    lines.append("")
    for period in avg_weights.columns:
        top3 = avg_weights[period].sort_values(ascending=False).head(3)
        formatted = ", ".join([f"{country} ({value:.3f})" for country, value in top3.items()])
        lines.append(f"Top countries in {period}: {formatted}.")

    lines.append("")
    lines.append(
        "Economic interpretation can focus on differences in energy exposure, import dependence, trade openness, "
        "financial transmission, and geographic proximity to common shocks. Countries that import energy more intensively "
        "or are hit earlier by supply-chain and regional shocks may receive higher pioneer weights in crisis periods."
    )

    lines.append("")
    lines.append("Part B — Forecasting analogy draft")
    lines.append("----------------------------------")
    lines.append(
        f"The lowest full-sample RMSE is obtained by {best_method} ({best_rmse:.4f}). This only means that the pooled EU inflation "
        "signal produced by this method tracks Ukrainian inflation relatively well in-sample. It does not prove that the method "
        "is a true forecasting model or that it identifies structural causes of Ukrainian inflation."
    )
    lines.append(
        "The exercise itself warns that PDM was designed to detect pioneers and accelerate collective learning, not to optimize forecasts. "
        "RMSE therefore captures tracking performance, not causal validity. Results may also depend on revisions, window choice, "
        "lag structure, and the fact that Ukraine is measured with a national CPI transformed to y/y inflation rather than a harmonised HICP series."
    )

    lines.append("")
    lines.append("Part B.1 assumption used here")
    lines.append("-----------------------------")
    lines.append(
        f"Rolling pioneer detection is based on pairwise country-UA PDM computations over a {ROLLING_WINDOW}-month window. "
        "This is an explicit target-specific assumption made because the wording 'relative to Ukraine' is otherwise ambiguous."
    )

    if dominant.dropna().empty:
        lines.append("No dominant pioneer was detected in the rolling exercise.")
    else:
        freq = dominant.dropna().value_counts(normalize=True).sort_values(ascending=False)
        lines.append(
            "Most frequent dominant pioneers: "
            + ", ".join([f"{country} ({share:.1%})" for country, share in freq.head(5).items()])
            + "."
        )

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ensure_output_dir()

    print("Building inflation panel...")
    infl_panel = build_inflation_panel(start=START, end=END)
    save_dataframe(infl_panel, "inflation_panel_raw.csv")
    plot_panel(infl_panel)

    panel = infl_panel.dropna().copy()
    save_dataframe(panel, "inflation_panel_complete_case.csv")
    print(f"Complete-case sample: {panel.index.min().strftime('%Y-%m')} to {panel.index.max().strftime('%Y-%m')}, {len(panel)} monthly observations.")

    # -----------------------------------------------------------------
    # Part A
    # -----------------------------------------------------------------
    print("\n=== Part A: Who pioneered European inflation dynamics? ===")
    w_angles = compute_pioneer_weights_angles(panel)
    w_angles_filled = w_angles.fillna(0.0)
    save_dataframe(w_angles_filled, "partA_pdm_angles_weights.csv")

    nonzero_summary = describe_nonzero_weights(w_angles)
    avg_weights = average_weights_by_period(w_angles_filled, PERIODS)
    rank_weights = rank_table_from_values(avg_weights, ascending=False)
    pioneer_strength = max_weight_summary(w_angles, PERIODS)

    save_dataframe(nonzero_summary, "partA_nonzero_pioneer_summary.csv")
    save_dataframe(avg_weights, "partA_average_weights_by_period.csv")
    save_dataframe(rank_weights, "partA_rankings_by_period.csv")
    save_dataframe(pioneer_strength, "partA_pioneer_strength_by_period.csv")

    plot_weight_heatmap(w_angles)
    plot_top_weight_lines(w_angles)

    print("\nAverage PDM angle weights by period:")
    print(avg_weights.round(4).to_string())
    print("\nCountry rankings by period (1 = highest weight):")
    print(rank_weights.to_string())

    # -----------------------------------------------------------------
    # Part B
    # -----------------------------------------------------------------
    print("\n=== Part B: Predicting Ukraine's inflation trajectory ===")
    eu_panel = panel.drop(columns=TARGET)
    actual_ua = panel[TARGET]

    rolling_scores, dominant = rolling_pairwise_target_pioneer(panel, target_col=TARGET, window=ROLLING_WINDOW)
    save_dataframe(rolling_scores, "partB_pairwise_rolling_scores.csv")
    dominant.to_frame().to_csv(OUTPUT_DIR / "partB_dominant_pioneer.csv")
    plot_dominant_pioneer(dominant)

    forecasts = compute_method_forecasts(eu_panel)
    save_dataframe(forecasts, "partB_method_forecasts.csv")

    rmse_table = compute_rmse_table(actual_ua, forecasts, PERIODS)
    save_dataframe(rmse_table, "partB_rmse_by_method_and_period.csv")

    print("\nRMSE by method and period:")
    print(rmse_table.round(4).to_string())

    discussion = build_discussion_note(
        max_weight_stats=pioneer_strength,
        avg_weights=avg_weights,
        rmse_table=rmse_table,
        dominant=dominant,
    )
    discussion_path = OUTPUT_DIR / "discussion_notes.txt"
    discussion_path.write_text(discussion, encoding="utf-8")

    print("\nOutputs written to:", OUTPUT_DIR.resolve())
    print("Suggested discussion note:")
    print("-" * 24)
    print(discussion)


if __name__ == "__main__":
    main()
