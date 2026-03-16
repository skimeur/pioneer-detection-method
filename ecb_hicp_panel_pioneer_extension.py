#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pioneer Detection extension for the ECB-SSSU inflation panel
============================================================

This auxiliary script reuses the ECB inflation-panel workflow and adds an
explicit Pioneer Detection analysis for the empirical question:

    Does France behave like a pioneer in the monthly inflation panel?

The preferred specification is the angle-based Pioneer Detection Method (PDM),
with the distance-based variant reported as a robustness check.

Reproducibility
---------------
- No randomness is used in this script.
- The sample window is controlled by the string constants below.
- The target country is controlled by a single switch variable:
  `TARGET_PROFILE = "france"` or `TARGET_PROFILE = "ukraine"`.

Run
---
    MPLBACKEND=Agg python ecb_hicp_panel_pioneer_extension.py

This writes a reproducible figure to `pioneer_detection_extension.png` and
prints country-level pioneer diagnostics to the console.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecb_hicp_panel_var_granger import (
    DEFAULT_COUNTRIES,
    build_inflation_panel,
    fetch_ecb_hicp_inflation_panel,
)
from pdm import (
    compute_pioneer_weights_angles,
    compute_pioneer_weights_distance,
    pooled_forecast,
)


# ============================================================================
# PIONEER DETECTION EXTENSION
# ============================================================================
# Reproducible configuration lives here.
# Change TARGET_PROFILE to switch between the France and Ukraine versions.

START = "2000-01"
END = "2025-12"
TARGET_PROFILE = "france"
OUTPUT_FIGURE = Path(__file__).with_name("pioneer_detection_extension.png")


def build_ecb_panel(
    countries: list[str] | None = None,
    start: str = START,
    end: str = END,
) -> pd.DataFrame:
    """
    Build an ECB-only HICP inflation panel with harmonised country data.
    """
    countries = countries or DEFAULT_COUNTRIES
    panel, _ = fetch_ecb_hicp_inflation_panel(countries=countries, start=start, end=end)
    panel = panel.copy().sort_index()
    # Put every series on the same monthly timestamp convention before comparison.
    panel.index = pd.to_datetime(panel.index).to_period("M").to_timestamp(how="start")
    panel.columns.name = None
    return panel


def get_target_configuration(target_profile: str) -> tuple[str, str]:
    """
    Map a user-facing profile string to the target country code and display label.
    """
    profile = target_profile.strip().lower()
    options = {
        "france": ("FR", "France"),
        "ukraine": ("UA", "Ukraine"),
    }
    if profile not in options:
        valid = ", ".join(sorted(options))
        raise ValueError(f"Unknown TARGET_PROFILE={target_profile!r}. Use one of: {valid}.")
    return options[profile]


def build_analysis_panel(target_profile: str, start: str = START, end: str = END) -> pd.DataFrame:
    """
    Build the panel needed for the selected target profile.

    France uses the harmonised ECB-only panel.
    Ukraine uses the ECB panel augmented with the Ukraine series.
    """
    target_code, _ = get_target_configuration(target_profile)
    if target_code == "UA":
        # Ukraine needs the broader panel builder because its inflation series
        # does not come from the ECB HICP dataset.
        return build_inflation_panel(countries=DEFAULT_COUNTRIES, start=start, end=end).dropna()
    # France is already inside the ECB panel, so the harmonised panel is enough.
    return build_ecb_panel(countries=DEFAULT_COUNTRIES, start=start, end=end).dropna()


def summarise_pioneer_weights(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Summarise time-varying pioneer weights by country.
    """
    weights = weights.astype(float)
    filled = weights.fillna(0.0)
    positive = filled > 0.0
    dominant_weight = filled.max(axis=1)
    # At each month, this picks the country that receives the largest pioneer
    # weight, i.e. the country that looks most like the "leader" of inflation
    # movements in that period.
    dominant_country = filled.idxmax(axis=1).where(dominant_weight > 0.0)

    detected_periods = positive.sum(axis=0)
    detected_share = positive.mean(axis=0)
    dominant_periods = dominant_country.value_counts().reindex(weights.columns, fill_value=0)

    detected_mask = dominant_country.notna()
    if detected_mask.any():
        dominant_share = (
            dominant_country[detected_mask].value_counts(normalize=True).reindex(weights.columns, fill_value=0.0)
        )
    else:
        dominant_share = pd.Series(0.0, index=weights.columns)

    summary = pd.DataFrame(
        {
            # Mean weight over the full sample. This is the average importance
            # of each country in the pooled signal.
            "mean_weight": filled.mean(axis=0),
            # Mean weight only in months where the country is actually detected
            # as a pioneer. This tells us how strong its role is when it leads.
            "mean_weight_when_detected": weights.mean(axis=0, skipna=True),
            "detected_periods": detected_periods,
            "detected_share": detected_share,
            "dominant_periods": dominant_periods,
            "dominant_share": dominant_share,
            "peak_weight": filled.max(axis=0),
        }
    )
    summary = summary.sort_values(["dominant_share", "detected_share", "mean_weight"], ascending=False)
    return summary, dominant_country.rename("dominant_pioneer"), dominant_weight.rename("dominant_weight")


def build_target_report(
    weights: pd.DataFrame,
    target: str,
    dominant_country: pd.Series,
) -> pd.Series:
    """
    Build a target-country report focused on the designated target country.
    """
    target_weights = weights[target].fillna(0.0)
    target_detected = target_weights > 0.0
    detected_periods = dominant_country.notna()
    dominant_target = dominant_country.eq(target)

    # Economic intuition:
    # - detected_share asks: how often does this country move early enough to be treated as a pioneer at all?
    # - dominant_share asks: when at least one pioneer exists, how often is this country the main leader?
    dominant_share = float(dominant_target[detected_periods].mean()) if detected_periods.any() else np.nan
    average_detected_weight = float(target_weights[target_detected].mean()) if target_detected.any() else 0.0

    return pd.Series(
        {
            "country": target,
            "sample_periods": len(weights),
            "detected_periods": int(target_detected.sum()),
            "detected_share": float(target_detected.mean()),
            "dominant_periods": int(dominant_target.sum()),
            "dominant_share": dominant_share,
            "mean_weight": float(target_weights.mean()),
            "average_weight_when_detected": average_detected_weight,
            "peak_weight": float(target_weights.max()),
        }
    )


def compare_methods(panel: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Compute the preferred and robustness-check PDM variants.

    Both methods are returned so the console output documents whether the
    empirical result depends materially on the PDM weighting variant.
    """
    methods = {
        # The paper's preferred version: it rewards countries whose inflation
        # moves first and toward which the rest of the panel converges.
        "PDM (angles)": compute_pioneer_weights_angles(panel),
        # Robustness check: same pioneer idea, but using distance weights rather
        # than angle-based speed of convergence.
        "PDM (distances)": compute_pioneer_weights_distance(panel),
    }

    method_rows = []
    for method_name, weights in methods.items():
        summary, leaders, _ = summarise_pioneer_weights(weights)
        report = build_target_report(weights, target=target, dominant_country=leaders)
        method_rows.append(
            {
                "method": method_name,
                "target_country": target,
                "target_detected_share": report["detected_share"],
                "target_dominant_share": report["dominant_share"],
                "target_mean_weight": report["mean_weight"],
                "top_country": summary.index[0],
            }
        )

    comparison = pd.DataFrame(method_rows).sort_values("target_dominant_share", ascending=False).reset_index(drop=True)
    return comparison, methods


def plot_extension_results(
    panel: pd.DataFrame,
    weights: pd.DataFrame,
    target: str,
    pooled: pd.Series,
    output_path: Path,
) -> None:
    """
    Save a compact figure for the PDM extension.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    # Top panel: compare the target country's inflation path with the rest of
    # the panel and with the PDM pooled signal.
    for column in panel.columns:
        if column == target:
            ax.plot(panel.index, panel[column], linewidth=2.5, color="tab:red", label=target)
        else:
            ax.plot(panel.index, panel[column], linewidth=0.9, alpha=0.35, color="gray")
    ax.plot(panel.index, pooled, linewidth=2.0, color="black", linestyle="--", label="PDM pooled")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Inflation (y/y, %)")
    ax.set_title(f"Inflation panel with {target} highlighted")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    # Bottom panel: show which countries receive the largest pioneer weights
    # over time. If the target line spikes early and often, that supports the
    # idea that it leads inflation dynamics in the panel.
    ranked = weights.mean(axis=0, skipna=True).sort_values(ascending=False)
    top_columns = ranked.index[:5].tolist()
    if target not in top_columns:
        top_columns = [target] + top_columns[:-1]

    for column in top_columns:
        style = {"linewidth": 2.5, "color": "tab:red"} if column == target else {"linewidth": 1.4}
        ax.plot(weights.index, weights[column].fillna(0.0), label=column, **style)

    ax.set_ylabel("PDM weight")
    ax.set_xlabel("Time")
    ax.set_title("Angle-based pioneer weights")
    ax.legend(frameon=False, ncol=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """
    Run the Pioneer Detection extension on the selected inflation panel.
    """
    target_country, target_label = get_target_configuration(TARGET_PROFILE)
    panel = build_analysis_panel(TARGET_PROFILE, start=START, end=END)

    # Run the two PDM variants on the same panel so we can see whether the
    # target country's leadership is robust to the weighting choice.
    comparison, method_weights = compare_methods(panel, target=target_country)
    preferred_weights = method_weights["PDM (angles)"]
    preferred_summary, preferred_leaders, preferred_leader_weight = summarise_pioneer_weights(preferred_weights)
    preferred_target_report = build_target_report(preferred_weights, target=target_country, dominant_country=preferred_leaders)
    preferred_pooled = pooled_forecast(panel, preferred_weights).rename("PDM_POOL")

    # These are the months where the target country looks most strongly like
    # the leader of inflation movements in the cross-country panel.
    top_target_months = (
        preferred_weights[target_country]
        .fillna(0.0)
        .sort_values(ascending=False)
        .head(10)
        .rename("target_weight")
        .to_frame()
    )

    top_target_months["dominant_pioneer"] = preferred_leaders.reindex(top_target_months.index)
    top_target_months["dominant_weight"] = preferred_leader_weight.reindex(top_target_months.index)

    plot_extension_results(
        panel=panel,
        weights=preferred_weights,
        target=target_country,
        pooled=preferred_pooled,
        output_path=OUTPUT_FIGURE,
    )

    print("=" * 72)
    print("Pioneer Detection extension for the inflation panel")
    print("=" * 72)
    print(f"Target profile: {TARGET_PROFILE} ({target_label}, code={target_country})")
    print(f"Sample: {panel.index.min().date()} to {panel.index.max().date()} | {len(panel)} monthly observations")
    print(f"Countries: {', '.join(panel.columns)}")

    print(f"\n=== {target_label}-focused method comparison ===")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Preferred specification: PDM (angles) ===")
    print(preferred_target_report.to_frame(name="value").to_string())

    print("\n=== Country ranking under PDM (angles) ===")
    print(preferred_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\n=== Top months for {target_label} pioneer weight ===")
    print(top_target_months.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\nFigure saved to: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
