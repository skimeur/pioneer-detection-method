"""
exercise_pdm_inflation.py

Exercise: Applying the Pioneer Detection Method to European Inflation Dynamics
Part A only

"""

from __future__ import annotations

from pathlib import Path
import importlib.util
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pdm import compute_pioneer_weights_angles


# =========================================================
# CONFIG
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs_part_a"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_STYLE = "seaborn-v0_8-whitegrid"

SUBPERIODS: Dict[str, Tuple[str, str]] = {
    "I_2002_2007": ("2002-01-01", "2007-12-31"),
    "II_2008_2012": ("2008-01-01", "2012-12-31"),
    "III_2013_2019": ("2013-01-01", "2019-12-31"),
    "IV_2020_2021": ("2020-01-01", "2021-12-31"),
    "V_2022_2023": ("2022-01-01", "2023-12-31"),
    "VI_2024_2025": ("2024-01-01", "2025-12-31"),
}


# =========================================================
# HELPERS
# =========================================================

def load_module_from_file(module_name: str, file_path: Path):
    """
    Dynamically load a Python module from a file path.
    """
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_inflation_panel() -> pd.DataFrame:
    """
    Load infl_panel from ecb_hicp_panel_var_granger.py.

    The starter file already builds a DataFrame named `infl_panel`.
    We retrieve it after importing the module.
    """
    module_path = SCRIPT_DIR / "ecb_hicp_panel_var_granger.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"Missing starter file: {module_path}"
        )

    ecb_module = load_module_from_file("ecb_hicp_panel_var_granger", module_path)

    if not hasattr(ecb_module, "infl_panel"):
        raise AttributeError(
            "The module ecb_hicp_panel_var_granger.py does not expose `infl_panel`."
        )

    panel = ecb_module.infl_panel.copy()

    if not isinstance(panel, pd.DataFrame):
        raise TypeError("`infl_panel` must be a pandas DataFrame.")

    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    return panel


def average_weights_by_subperiod(
    weights: pd.DataFrame,
    subperiods: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    """
    Compute average pioneer weights by country and subperiod.

    Returns
    -------
    pd.DataFrame
        Rows: countries
        Columns: subperiod labels
    """
    out: dict[str, pd.Series] = {}

    for label, (start, end) in subperiods.items():
        sub = weights.loc[start:end]
        out[label] = sub.mean(axis=0)

    table = pd.DataFrame(out)
    table.index.name = "country"
    return table


def rank_countries_by_subperiod(avg_table: pd.DataFrame) -> pd.DataFrame:
    """
    Convert average weights table into descending ranks by subperiod.

    Rank 1 = highest average pioneer weight.
    """
    rank_table = avg_table.rank(axis=0, ascending=False, method="min")
    rank_table = rank_table.astype("Int64")
    rank_table.index.name = "country"
    return rank_table


def save_table_csv_and_excel(df: pd.DataFrame, stem: str) -> None:
    """
    Save table to CSV and Excel.
    """
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    xlsx_path = OUTPUT_DIR / f"{stem}.xlsx"

    df.to_csv(csv_path)
    try:
        df.to_excel(xlsx_path)
    except Exception:
        # Excel export may fail if openpyxl is unavailable;
        # CSV remains the primary deliverable.
        pass


def plot_pioneer_weights_lines(weights: pd.DataFrame) -> Path:
    """
    Figure A.1 (version 1): line plot, one line per country.
    """
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(14, 7))

    for country in weights.columns:
        ax.plot(weights.index, weights[country], linewidth=1.4, label=country)

    ax.set_title("Question A.1 — PDM pioneer weights over time (angles)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pioneer weight")
    ax.legend(ncol=4, fontsize=9, frameon=False)
    fig.tight_layout()

    path = OUTPUT_DIR / "figure_A1_pioneer_weights_lines.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pioneer_weights_heatmap(weights: pd.DataFrame) -> Path:
    """
    Figure A.1 (version 2): heatmap for easier visual inspection.
    """
    plt.style.use(PLOT_STYLE)

    heat = weights.T.copy()
    heat.index.name = "country"

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(
        heat,
        cmap="viridis",
        cbar_kws={"label": "Pioneer weight"},
        ax=ax,
    )

    ax.set_title("Question A.1 — PDM pioneer weights over time (heatmap)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Country")
    fig.tight_layout()

    path = OUTPUT_DIR / "figure_A1_pioneer_weights_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def print_non_zero_summary(weights: pd.DataFrame) -> None:
    """
    Print a compact summary of non-zero pioneer episodes by country.
    """
    print("\n" + "=" * 72)
    print("QUESTION A.1 — NON-ZERO PIONEER WEIGHT SUMMARY")
    print("=" * 72)

    has_any = weights.fillna(0).gt(0).any(axis=0)

    if not has_any.any():
        print("No country receives a non-zero pioneer weight in the sample.")
        return

    for country in weights.columns:
        s = weights[country].fillna(0)
        active = s[s > 0]
        if active.empty:
            print(f"{country:>3}: never positive")
            continue

        first_date = active.index.min().strftime("%Y-%m")
        last_date = active.index.max().strftime("%Y-%m")
        n_periods = int((s > 0).sum())
        mean_positive = float(active.mean())
        max_weight = float(active.max())

        print(
            f"{country:>3}: "
            f"{n_periods:>3} positive months | "
            f"first={first_date} | last={last_date} | "
            f"mean_positive={mean_positive:.4f} | max={max_weight:.4f}"
        )


# =========================================================
# PART A
# =========================================================

def run_part_a() -> None:
    print("\n" + "=" * 72)
    print("PART A — WHO PIONEERED EUROPEAN INFLATION DYNAMICS?")
    print("=" * 72)

    # -----------------------------------------------------
    # A.1(a) Load the inflation panel and restrict to complete-case sample
    # -----------------------------------------------------
    infl_panel = load_inflation_panel()
    df = infl_panel.dropna().copy()

    print("\n[A.1.a] Inflation panel loaded")
    print(f"Shape before dropna: {infl_panel.shape}")
    print(f"Shape after  dropna: {df.shape}")
    print("\nColumns:")
    print(list(df.columns))
    print("\nSample period:")
    print(f"{df.index.min().strftime('%Y-%m')} -> {df.index.max().strftime('%Y-%m')}")

    # -----------------------------------------------------
    # A.1(b) Apply PDM with angles
    # -----------------------------------------------------
    weights = compute_pioneer_weights_angles(df)

    print("\n[A.1.b] Pioneer weights computed")
    print(f"Weights shape: {weights.shape}")

    save_table_csv_and_excel(weights, "A1_pioneer_weights_time_varying")

    # -----------------------------------------------------
    # A.1(c) Plot pioneer weights over time
    # -----------------------------------------------------
    line_fig = plot_pioneer_weights_lines(weights)
    heatmap_fig = plot_pioneer_weights_heatmap(weights)

    print(f"\n[A.1.c] Saved figure: {line_fig}")
    print(f"[A.1.c] Saved figure: {heatmap_fig}")

    print_non_zero_summary(weights)

    # -----------------------------------------------------
    # A.2 Average pioneer weights by subperiod
    # -----------------------------------------------------
    avg_table = average_weights_by_subperiod(weights, SUBPERIODS)
    rank_table = rank_countries_by_subperiod(avg_table)

    save_table_csv_and_excel(avg_table, "A2_average_pioneer_weights_by_subperiod")
    save_table_csv_and_excel(rank_table, "A2_rankings_by_subperiod")

    print("\n" + "=" * 72)
    print("QUESTION A.2 — AVERAGE PIONEER WEIGHTS BY SUBPERIOD")
    print("=" * 72)
    print(avg_table.round(4).to_string())

    print("\n" + "=" * 72)
    print("QUESTION A.2 — RANKINGS BY SUBPERIOD (1 = highest pioneer weight)")
    print("=" * 72)
    print(rank_table.to_string())


    print("\nDone. All Part A outputs are in:")
    print(OUTPUT_DIR)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    run_part_a()

# =========================================================
# A.1 (c) Interpretation — Which countries receive non-zero pioneer weight?
# =========================================================
# All countries receive a positive pioneer weight at some point in the sample.
# These episodes are short and irregular and no country appears as a
# permanent pioneer.
#
# Finland, France & Austria appear quite frequently in pioneer episodes.
# Spain Italy and the Netherlands also show several occurrences. Ukraine
# appears less often but when it does the weight tends to be higher than for
# most other countries. Ireland & Portugal appear less frequently and their
# weights remain relatively small.


# =========================================================
# A.1 (d) Interpretation — Do we observe strong pioneers during
# periods of low and stable inflation?
# =========================================================
# During the period 2002–2007 inflation was relatively low and stable across
# European countries. The results do not show a clear dominant pioneer during
# this period. Average pioneer weights are relatively similar across Italy Spain & Greece.
#
# This is consistent with the logic of the Pioneer Detection Method. When
# inflation dynamics are stable and countries move in similar ways there is
# less opportunity for one country to move earlier than the others and lead
# the adjustment.
#
# Stronger pioneer episodes tend to appear during periods of economic stress
# or structural shocks when inflation dynamics diverge more across countries.

# =========================================================
# A.2 (b) Interpretation — Does the ranking change over time?
# =========================================================
# Yes, the ranking changes significantly across subperiods.
#
# Different countries appear as the main pioneer in different periods. Italy
# ranks first in the first period (2002/2007), Ukraine during the financial
# crisis period (2008/2012), Austria during the following years, Belgium during
# the energy crisis period, and Finland during the most recent disinflation
# phase.
#
# This shows that pioneer status is not stable over time. The countries that
# move first in inflation dynamics depend on the type of macroeconomic shock
# affecting Europe during each period.
