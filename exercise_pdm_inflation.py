#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Robust local import of pdm.py
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pdm import (  # noqa: E402
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
COUNTRIES = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]
TARGET_COUNTRY = "FR"
ROLLING_WINDOW = 36
START_PERIOD = "2000-01"
END_PERIOD = "2025-12"
OUTPUT_DIR = HERE / "outputs_pdm_france"

SUBPERIODS = {
    "I (2002-07)": ("2002-01", "2007-12"),
    "II (2008-12)": ("2008-01", "2012-12"),
    "III (2013-19)": ("2013-01", "2019-12"),
    "IV (2020-21)": ("2020-01", "2021-12"),
    "V (2022-23)": ("2022-01", "2023-12"),
    "VI (2024-25)": ("2024-01", "2025-12"),
}

COUNTRY_LABELS = {
    "DE": "Germany",
    "FR": "France",
    "IT": "Italy",
    "ES": "Spain",
    "NL": "Netherlands",
    "BE": "Belgium",
    "AT": "Austria",
    "PT": "Portugal",
    "IE": "Ireland",
    "FI": "Finland",
    "GR": "Greece",
}


@dataclass(frozen=True)
class ECBConfig:
    countries: list[str]
    start: str = START_PERIOD
    end: str = END_PERIOD
    item: str = "000000"
    seasonal_adjustment: str = "N"
    measure: str = "4"
    variation: str = "ANR"
    freq: str = "M"
    timeout: int = 60


# -----------------------------------------------------------------------------
# Data access
# -----------------------------------------------------------------------------
def fetch_ecb_hicp_panel(config: ECBConfig) -> pd.DataFrame:
    """Fetch monthly HICP y/y inflation from the ECB SDMX endpoint."""
    base = "https://data-api.ecb.europa.eu/service/data"
    key = (
        f"{config.freq}.{'+'.join(config.countries)}."
        f"{config.seasonal_adjustment}.{config.item}.{config.measure}.{config.variation}"
    )
    params = {
        "format": "csvdata",
        "startPeriod": config.start,
        "endPeriod": config.end,
    }
    url = f"{base}/ICP/{key}"

    response = requests.get(url, params=params, timeout=config.timeout)
    response.raise_for_status()

    raw = pd.read_csv(StringIO(response.text))
    required = {"TIME_PERIOD", "OBS_VALUE", "REF_AREA"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"ECB response missing expected columns: {sorted(missing)}")

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
        .astype(float)
    )
    panel.index = panel.index.to_period("M").to_timestamp(how="start")
    return panel


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def complete_case_panel(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy().sort_index().dropna(how="any")
    if out.empty:
        raise ValueError("Complete-case panel is empty after dropna().")
    return out


def make_subperiod_average_table(weights: pd.DataFrame) -> pd.DataFrame:
    avg = pd.DataFrame(index=weights.columns)
    for label, (start, end) in SUBPERIODS.items():
        avg[label] = weights.loc[start:end].mean()
    return avg.sort_index()


def make_subperiod_rank_table(avg_weights: pd.DataFrame) -> pd.DataFrame:
    return avg_weights.rank(ascending=False, method="min").astype("Int64")


def top_nonzero_summary(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country in weights.columns:
        positive = weights[country].fillna(0.0) > 0
        share = positive.mean()
        peak_date = weights[country].idxmax() if weights[country].notna().any() else pd.NaT
        peak_weight = weights[country].max(skipna=True)
        rows.append(
            {
                "country": country,
                "nonzero_share": share,
                "peak_weight": peak_weight if pd.notna(peak_weight) else np.nan,
                "peak_date": peak_date.strftime("%Y-%m") if pd.notna(peak_date) else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["nonzero_share", "peak_weight"], ascending=False)


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    aligned = pd.concat([actual.rename("actual"), predicted.rename("pred")], axis=1).dropna()
    if aligned.empty:
        return np.nan
    return float(np.sqrt(np.mean((aligned["pred"] - aligned["actual"]) ** 2)))


def compute_method_forecasts(experts: pd.DataFrame) -> dict[str, pd.Series]:
    methods: dict[str, pd.Series] = {}

    methods["PDM angles"] = pooled_forecast(
        experts,
        compute_pioneer_weights_angles(experts),
    )
    methods["PDM distances"] = pooled_forecast(
        experts,
        compute_pioneer_weights_distance(experts),
    )
    methods["Granger"] = pooled_forecast(
        experts,
        compute_granger_weights(experts, maxlag=1),
    )
    methods["Lagged correlation"] = pooled_forecast(
        experts,
        compute_lagged_correlation_weights(experts, lag=1),
    )
    methods["Multivariate regression"] = pooled_forecast(
        experts,
        compute_multivariate_regression_weights(experts, lag=1),
    )
    methods["Transfer entropy"] = pooled_forecast(
        experts,
        compute_transfer_entropy_weights(experts, n_bins=3, lag=1),
    )
    methods["Linear pooling"] = pooled_forecast(
        experts,
        compute_linear_pooling_weights(experts),
    )
    methods["Median pooling"] = compute_median_pooling(experts)
    return methods


def evaluate_methods_by_rmse(actual: pd.Series, forecasts: dict[str, pd.Series]) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Method": list(forecasts.keys()),
            "RMSE": [rmse(actual, series) for series in forecasts.values()],
        }
    ).sort_values("RMSE", ascending=True, kind="stable")
    return table.reset_index(drop=True)


def evaluate_methods_by_subperiod(actual: pd.Series, forecasts: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for method_name, series in forecasts.items():
        row: dict[str, float | str] = {"Method": method_name}
        for label, (start, end) in SUBPERIODS.items():
            row[label] = rmse(actual.loc[start:end], series.loc[start:end])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Method").reset_index(drop=True)


def compute_rolling_dominant_pioneer(experts: pd.DataFrame, window: int) -> pd.Series:
    dominant_labels = []
    dominant_index = []

    for stop in range(window, len(experts) + 1):
        block = experts.iloc[stop - window : stop]
        block_weights = compute_pioneer_weights_angles(block)
        avg = block_weights.mean(axis=0)
        if avg.fillna(0).sum() <= 0:
            dominant_labels.append("none")
        else:
            dominant_labels.append(avg.idxmax())
        dominant_index.append(block.index[-1])

    return pd.Series(dominant_labels, index=pd.Index(dominant_index, name="Date"), name="dominant_pioneer")


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_panel(panel: pd.DataFrame, save_to: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    for column in panel.columns:
        ax.plot(panel.index, panel[column], linewidth=1.0, alpha=0.9, label=column)
    ax.axhline(0, linewidth=0.8, linestyle="--")
    ax.set_title("Euro-area HICP inflation panel")
    ax.set_xlabel("Date")
    ax.set_ylabel("Inflation (y/y, %)")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(save_to, dpi=180)
    plt.close(fig)


def plot_weight_heatmap(weights: pd.DataFrame, save_to: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.5))
    matrix = weights.T.fillna(0.0)
    image = ax.imshow(
        matrix.values,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[0, len(weights.index) - 1, 0, len(weights.columns)],
    )
    tick_positions = np.linspace(0, len(weights.index) - 1, 8, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([weights.index[i].strftime("%Y-%m") for i in tick_positions], rotation=0)
    ax.set_yticks(np.arange(len(weights.columns)) + 0.5)
    ax.set_yticklabels(weights.columns)
    ax.set_title("Part A — PDM (angles) weights over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Country")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Weight")
    fig.tight_layout()
    fig.savefig(save_to, dpi=180)
    plt.close(fig)


def plot_weight_lines(weights: pd.DataFrame, save_to: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for column in weights.columns:
        ax.plot(weights.index, weights[column], linewidth=1.0, alpha=0.8, label=column)
    ax.set_title("Part A — PDM (angles) weights by country")
    ax.set_xlabel("Date")
    ax.set_ylabel("Pioneer weight")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(save_to, dpi=180)
    plt.close(fig)


def plot_dominant_pioneer(dominant: pd.Series, save_to: Path) -> None:
    categories = [c for c in sorted(dominant.unique()) if c != "none"] + (["none"] if "none" in dominant.unique() else [])
    mapping = {name: idx for idx, name in enumerate(categories)}
    y = dominant.map(mapping)

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.scatter(dominant.index, y, s=14)
    ax.set_yticks(list(mapping.values()))
    ax.set_yticklabels(list(mapping.keys()))
    ax.set_title(f"Part B — Rolling dominant pioneer for {TARGET_COUNTRY}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Country")
    fig.tight_layout()
    fig.savefig(save_to, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Written outputs
# -----------------------------------------------------------------------------
def write_part_a_discussion(avg_weights: pd.DataFrame, nonzero_table: pd.DataFrame, path: Path) -> None:
    top_by_period = {
        label: avg_weights[label].sort_values(ascending=False).head(3)
        for label in avg_weights.columns
    }
    lines = [
        "Part A discussion",
        "=================",
        "",
        "1) Low-inflation years (2002-2007)",
        "During the Great Moderation, pioneer signals should be weaker because the cross-country inflation dispersion is smaller and structural breaks are rarer. In a PDM framework, this means that the distance-reduction and orientation conditions are less often met simultaneously.",
        "",
        "2) Time variation in rankings",
        "The country ranking is not stable over time: the leading countries differ across crisis periods, low-inflation years, the pandemic shock, the 2022-2023 energy episode, and the later disinflation phase.",
        "",
        "3) Countries with the strongest average signals by subperiod",
    ]
    for label, series in top_by_period.items():
        summary = ", ".join(f"{country} ({value:.3f})" for country, value in series.items())
        lines.append(f"- {label}: {summary}")

    lines += [
        "",
        "4) Economic interpretation",
        "Countries that are more exposed to external energy shocks, trade re-pricing, logistics bottlenecks, or rapid pass-through from imported prices can appear earlier in the inflation cycle and temporarily act as pioneers. This does not imply a permanent structural leadership: pioneership is episode-specific.",
        "",
        "5) Non-zero weight diagnostics",
        "The attached CSV file reports how frequently each country receives strictly positive pioneer weight and the date of its peak weight. This is useful to distinguish occasional pioneers from countries that almost never lead the cross-section.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_part_b_discussion(rmse_table: pd.DataFrame, path: Path) -> None:
    best = rmse_table.iloc[0]
    lines = [
        "Part B discussion",
        "=================",
        "",
        f"Target country: {TARGET_COUNTRY} ({COUNTRY_LABELS[TARGET_COUNTRY]})",
        "",
        f"Best overall RMSE in this implementation: {best['Method']} ({best['RMSE']:.4f}).",
        "",
        "Interpretation caveat",
        "--------------------",
        "A low RMSE here does not prove that a method is a good structural forecasting model. The exercise uses contemporaneous weighted combinations of other countries' inflation rates, not a full causal forecasting design with real-time information constraints.",
        "",
        "Why the analogy is limited",
        "- The PDM was designed to detect pioneers and accelerate collective learning under structural change.",
        "- A good fit may simply reflect common euro-area shocks rather than genuine predictive content for France.",
        "- Constant-weight benchmark methods can perform well if comovement dominates true lead-lag effects.",
        "- Results are sample-dependent and may differ under another target country, horizon, lag choice, or rolling specification.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    print("Fetching ECB HICP inflation panel...")
    panel_raw = fetch_ecb_hicp_panel(ECBConfig(countries=COUNTRIES))
    panel = complete_case_panel(panel_raw)

    print(
        f"Complete-case sample: {panel.shape[0]} months x {panel.shape[1]} countries "
        f"({panel.index.min():%Y-%m} to {panel.index.max():%Y-%m})"
    )
    print(f"Part B target country: {TARGET_COUNTRY} ({COUNTRY_LABELS[TARGET_COUNTRY]})")

    # Save raw panel
    panel.to_csv(OUTPUT_DIR / "inflation_panel_complete_case.csv")
    plot_panel(panel, OUTPUT_DIR / "inflation_panel.png")

    # ------------------------------------------------------------------
    # Part A
    # ------------------------------------------------------------------
    print("\n[Part A] Computing PDM-angle weights on the full euro-area panel...")
    weights_a = compute_pioneer_weights_angles(panel)
    avg_weights = make_subperiod_average_table(weights_a)
    rank_weights = make_subperiod_rank_table(avg_weights)
    nonzero = top_nonzero_summary(weights_a)

    avg_weights.to_csv(OUTPUT_DIR / "part_a_average_weights_by_period.csv")
    rank_weights.to_csv(OUTPUT_DIR / "part_a_rankings_by_period.csv")
    nonzero.to_csv(OUTPUT_DIR / "part_a_nonzero_weight_summary.csv", index=False)

    plot_weight_heatmap(weights_a, OUTPUT_DIR / "part_a_weights_heatmap.png")
    plot_weight_lines(weights_a, OUTPUT_DIR / "part_a_weights_lines.png")
    write_part_a_discussion(avg_weights, nonzero, OUTPUT_DIR / "part_a_discussion.txt")

    print("Top countries by non-zero weight frequency:")
    print(nonzero.head(5).to_string(index=False))

    # ------------------------------------------------------------------
    # Part B
    # ------------------------------------------------------------------
    print(f"\n[Part B] Treating {TARGET_COUNTRY} as the target and the other countries as experts...")
    experts = panel.drop(columns=[TARGET_COUNTRY])
    actual = panel[TARGET_COUNTRY].rename(TARGET_COUNTRY)

    dominant = compute_rolling_dominant_pioneer(experts, ROLLING_WINDOW)
    dominant.to_csv(OUTPUT_DIR / "part_b_dominant_pioneer.csv")
    plot_dominant_pioneer(dominant, OUTPUT_DIR / "part_b_dominant_pioneer.png")

    forecasts = compute_method_forecasts(experts)
    overall_rmse = evaluate_methods_by_rmse(actual, forecasts)
    subperiod_rmse = evaluate_methods_by_subperiod(actual, forecasts)

    overall_rmse.to_csv(OUTPUT_DIR / "part_b_rmse_overall.csv", index=False)
    subperiod_rmse.to_csv(OUTPUT_DIR / "part_b_rmse_by_subperiod.csv", index=False)
    write_part_b_discussion(overall_rmse, OUTPUT_DIR / "part_b_discussion.txt")

    print("\nOverall RMSE ranking:")
    print(overall_rmse.to_string(index=False))

    dominant_counts = dominant.value_counts(dropna=False).rename_axis("country").reset_index(name="windows")
    dominant_counts.to_csv(OUTPUT_DIR / "part_b_dominant_pioneer_counts.csv", index=False)

    # Convenience summary file
    summary_text = textwrap.dedent(
        f"""
        Exercise summary
        ================

        Output folder: {OUTPUT_DIR}
        Countries in panel: {', '.join(panel.columns)}
        Target in Part B: {TARGET_COUNTRY} ({COUNTRY_LABELS[TARGET_COUNTRY]})
        Rolling window: {ROLLING_WINDOW} months

        Files generated:
        - inflation_panel_complete_case.csv
        - inflation_panel.png
        - part_a_average_weights_by_period.csv
        - part_a_rankings_by_period.csv
        - part_a_nonzero_weight_summary.csv
        - part_a_weights_heatmap.png
        - part_a_weights_lines.png
        - part_a_discussion.txt
        - part_b_dominant_pioneer.csv
        - part_b_dominant_pioneer_counts.csv
        - part_b_dominant_pioneer.png
        - part_b_rmse_overall.csv
        - part_b_rmse_by_subperiod.csv
        - part_b_discussion.txt
        """
    ).strip()
    (OUTPUT_DIR / "README_outputs.txt").write_text(summary_text + "\n", encoding="utf-8")

    print(f"\nDone. All tables, figures and text outputs were written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
