#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECB HICP Inflation Panel — Pioneer Detection Extension
======================================================
This script extends the baseline ECB HICP inflation panel analysis by implementing the Pioneer
Detection Method. It aims to answer the research question of "Does a chosen country exhibit a pioneer 
(leading) role in inflation dynamics relative to the other countries in the sample?"

Outputs include:
- Full inflation panel plot
- Tested country vs peer-average plot
- Pioneer score over time
- Country ranking by average pioneer score
- Granger ranking for predictors of the tested country 
- CSV outputs saved locally
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from pathlib import Path
from statsmodels.tsa.stattools import grangercausalitytests


# ============================================================
# USER SETTINGS - Modify country as needed
# ============================================================
COUNTRY_TO_TEST = "FR" 
COUNTRIES = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]

START_DATE = "2000-01"
END_DATE = "2025-12"

SMOOTHING_WINDOW = 3
MAX_GRANGER_LAG = 6

SAVE_OUTPUTS = True
OUTPUT_DIR = Path("output_pioneer")


# ============================================================
# DATA FETCH
# ============================================================
def fetch_ecb_hicp_inflation_panel(
    countries,
    start="2000-01",
    end="2025-12",
    item="000000",   
    sa="N",          
    measure="4",     
    variation="ANR", 
    freq="M",
    timeout=60,
):
 
    base = "https://data-api.ecb.europa.eu/service/data"
    key = f"{freq}.{'+'.join(countries)}.{sa}.{item}.{measure}.{variation}"
    params = {"format": "csvdata", "startPeriod": start, "endPeriod": end}
    url = f"{base}/ICP/{key}"

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()

    raw = pd.read_csv(StringIO(r.text))

    if "TIME_PERIOD" not in raw.columns or "OBS_VALUE" not in raw.columns:
        raise ValueError(f"Unexpected ECB response format. Columns: {list(raw.columns)}")

    if "REF_AREA" not in raw.columns:
        raise ValueError("REF_AREA column not found in ECB response.")

    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"])
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")

    panel = (
        raw.pivot_table(
            index="TIME_PERIOD",
            columns="REF_AREA",
            values="OBS_VALUE",
            aggfunc="last"
        )
        .sort_index()
    )

    panel.index = panel.index.to_period("M").to_timestamp(how="start")
    panel = panel[countries]  # preserve requested order

    return panel, raw


# ============================================================
# PIONEER IMPLEMENTATION
# ============================================================
# First this identifies trends with a simple moving average.
def smooth_panel(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    return df.rolling(window=window, min_periods=1).mean()

# Then we compute the detailed PDM objects for one target country relative to the average of peers.
def compute_country_pioneer_details(smoothed_df: pd.DataFrame, target_country: str) -> pd.DataFrame:
    if target_country not in smoothed_df.columns:
        raise ValueError(f"{target_country} not found in dataframe columns.")

    peer_cols = [c for c in smoothed_df.columns if c != target_country]
    if len(peer_cols) == 0:
        raise ValueError("Need at least one peer country.")

    s_i = smoothed_df[target_country]
    s_m = smoothed_df[peer_cols].mean(axis=1)

    # Step 2: distance reduction
    distance_t = (s_i - s_m).abs()
    distance_t_minus_1 = distance_t.shift(1)
    delta_distance = (distance_t < distance_t_minus_1).astype(float)

    # Step 3: orientation change
    # Operationalization with month-to-month absolute trend changes
    theta_i = s_i.diff().abs()
    theta_m = s_m.diff().abs()
    delta_orientation = (theta_m > theta_i).astype(float)

    # Step 4: pioneer weight
    denom = theta_m + theta_i
    pioneer_score = np.where(
        denom > 0,
        delta_distance * delta_orientation * (theta_m / denom),
        0.0
    )

    details = pd.DataFrame({
        "target_smoothed": s_i,
        "peer_average": s_m,
        "distance_t": distance_t,
        "distance_t_minus_1": distance_t_minus_1,
        "delta_distance": delta_distance,
        "theta_i": theta_i,
        "theta_m": theta_m,
        "delta_orientation": delta_orientation,
        "pioneer_score": pioneer_score,
    }, index=smoothed_df.index)

    return details.fillna(0.0)


def compute_all_country_pioneer_scores(smoothed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PDM pioneer scores for all countries.
    """
    out = pd.DataFrame(index=smoothed_df.index)

    for country in smoothed_df.columns:
        details = compute_country_pioneer_details(smoothed_df, country)
        out[country] = details["pioneer_score"]

    return out


# ============================================================
# GRANGER COMPARISON
# ============================================================
# This section computes bivariate granger rankings for predictors of the target country.
# It then ranks countries by their minimum p value across lags
def granger_rank(df: pd.DataFrame, target_country: str, maxlag: int = 6) -> pd.DataFrame:
    rows = []

    for c in df.columns:
        if c == target_country:
            continue

        data_gc = df[[target_country, c]].dropna()

        if len(data_gc) < maxlag + 5:
            rows.append({
                "country": c,
                "min_pvalue": np.nan,
                "best_lag": np.nan,
                "status": "insufficient_data"
            })
            continue

        try:
            res = grangercausalitytests(data_gc, maxlag=maxlag, verbose=False)
            pvals = {lag: res[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1)}
            best_lag = min(pvals, key=pvals.get)
            min_p = pvals[best_lag]

            rows.append({
                "country": c,
                "min_pvalue": min_p,
                "best_lag": best_lag,
                "status": "ok"
            })

        except Exception as e:
            rows.append({
                "country": c,
                "min_pvalue": np.nan,
                "best_lag": np.nan,
                "status": f"failed: {e}"
            })

    out = pd.DataFrame(rows).sort_values(["min_pvalue", "country"], na_position="last")
    return out.reset_index(drop=True)


# ============================================================
# PLOTTING
# ============================================================
def plot_full_panel(df: pd.DataFrame, output_dir: Path | None = None):
    plt.figure(figsize=(12, 6))
    for country in df.columns:
        plt.plot(df.index, df[country], label=country, linewidth=1)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("ECB HICP Inflation Panel")
    plt.xlabel("Time")
    plt.ylabel("Inflation rate (y/y, %)")
    plt.legend(ncol=3, fontsize=9, frameon=False)
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / "full_inflation_panel.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_target_vs_peers(details: pd.DataFrame, target_country: str, output_dir: Path | None = None):
    plt.figure(figsize=(12, 5))
    plt.plot(details.index, details["target_smoothed"], label=f"{target_country} (smoothed)", linewidth=2)
    plt.plot(details.index, details["peer_average"], label="Peer average (excluding target)", linewidth=2)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title(f"{target_country} vs peer average")
    plt.xlabel("Time")
    plt.ylabel("Inflation rate (y/y, %)")
    plt.legend(frameon=False)
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / f"{target_country}_vs_peer_average.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_pioneer_score(details: pd.DataFrame, target_country: str, output_dir: Path | None = None):
    plt.figure(figsize=(12, 4))
    plt.plot(details.index, details["pioneer_score"], linewidth=1.8)
    plt.title(f"Pioneer score over time: {target_country}")
    plt.xlabel("Time")
    plt.ylabel("Pioneer score")
    plt.tight_layout()

    if output_dir is not None:
        plt.savefig(output_dir / f"{target_country}_pioneer_score.png", dpi=200, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN
# ============================================================
def main():
    if COUNTRY_TO_TEST not in COUNTRIES:
        raise ValueError(f"COUNTRY_TO_TEST={COUNTRY_TO_TEST} must be in COUNTRIES={COUNTRIES}")

    if SAVE_OUTPUTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = OUTPUT_DIR
    else:
        output_dir = None

    # 1) Load data
    infl_panel, infl_long = fetch_ecb_hicp_inflation_panel(
        countries=COUNTRIES,
        start=START_DATE,
        end=END_DATE
    )

    # Keep the raw panel for plotting and a clean panel for tests
    df = infl_panel.copy().sort_index()
    df_clean = df.dropna()

    # 2) Smooth data for the PDM
    smoothed = smooth_panel(df, window=SMOOTHING_WINDOW)

    # 3) Compute target-country PDM details
    target_details = compute_country_pioneer_details(smoothed, COUNTRY_TO_TEST)

    # 4) Compute all-country scores and ranking
    all_scores = compute_all_country_pioneer_scores(smoothed)
    ranking = (
        all_scores.mean(axis=0)
        .sort_values(ascending=False)
        .rename("avg_pioneer_score")
        .reset_index()
        .rename(columns={"index": "country"})
    )

    # 5) Granger comparison
    granger_table = granger_rank(df_clean, target_country=COUNTRY_TO_TEST, maxlag=MAX_GRANGER_LAG)

    # 6) Console output
    print("\n============================================================")
    print(f"PIONEER DETECTION RESULTS FOR {COUNTRY_TO_TEST}")
    print("============================================================")

    print("\n=== Average Pioneer Score Ranking ===")
    print(ranking.to_string(index=False))

    target_rank = ranking.index[ranking["country"] == COUNTRY_TO_TEST][0] + 1
    print(f"\n{COUNTRY_TO_TEST} rank by average pioneer score: {target_rank} / {len(ranking)}")

    print("\n=== Pioneer Score Summary for Tested Country ===")
    print(target_details["pioneer_score"].describe().to_string())

    print("\n=== Granger Causality Ranking: X -> " + COUNTRY_TO_TEST + " ===")
    print(granger_table.to_string(index=False))

    avg_score = target_details["pioneer_score"].mean()
    med_score = target_details["pioneer_score"].median()
    share_positive = (target_details["pioneer_score"] > 0).mean()

    print("\n=== Brief Interpretation ===")
    print(
        f"Average pioneer score for {COUNTRY_TO_TEST}: {avg_score:.4f}. "
        f"Median: {med_score:.4f}. "
        f"Share of months with strictly positive pioneer score: {share_positive:.2%}."
    )

    # 7) Save outputs
    if output_dir is not None:
        infl_panel.to_csv(output_dir / "inflation_panel.csv")
        smoothed.to_csv(output_dir / "inflation_panel_smoothed.csv")
        target_details.to_csv(output_dir / f"{COUNTRY_TO_TEST}_pioneer_details.csv")
        all_scores.to_csv(output_dir / "all_country_pioneer_scores.csv")
        ranking.to_csv(output_dir / "country_pioneer_ranking.csv", index=False)
        granger_table.to_csv(output_dir / f"granger_ranking_to_{COUNTRY_TO_TEST}.csv", index=False)

    # 8) Plots
    plot_full_panel(df, output_dir=output_dir)
    plot_target_vs_peers(target_details, COUNTRY_TO_TEST, output_dir=output_dir)
    plot_pioneer_score(target_details, COUNTRY_TO_TEST, output_dir=output_dir)

# Interpretation
# By applying the Pioneer Detection Method to the ECB HICP inflation panel, the results suggest that France is indeed holding a 
# pioneer role relative to other countries in the sample. This suggests that other countries tend to converge towards France's 
# inflation path. However, France's role as the leader is not constant over time. The pioneer score is zero in most months and only 
# becomes positive in specific periods. Therefore, France cannot be seen as a continuous inflation leader, but instead a country that 
# can have a leading role occasionally depending on the phase in the inflation cycle. Comparisons with other countries strengthen this 
# interpretation as Italy also shows a relatively strong pioneer behaviour and Spain has more moderate behaviour. Therefore, this proves 
# that the pioneer behaviour is more heterogenous across countries and varies over time.

if __name__ == "__main__":
    main()
    

