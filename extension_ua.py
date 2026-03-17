#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pioneer Detection Extension for the ECB HICP Panel
=====================================================

This script extends the baseline inflation panel code by implementing a
Pioneer Detection Method.
It aims to answer the question: 'Does Ukraine exhibit a pioneer / leading role in inflation dynamics
relative to other countries in the sample?'

Outputs
-------
- Inflation panel plot
- Ukraine vs peer-average plot
- Ukraine pioneer score over time
- Country ranking by average pioneer score
- Granger-causality comparison table for X -> UA
"""

from py_compile import main

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from statsmodels.tsa.stattools import grangercausalitytests


# ------------------------------------------------------------
# Fetch ECB HICP inflation panel
# ------------------------------------------------------------
def fetch_ecb_hicp_inflation_panel(
    countries,
    start="2000-01",
    end="2025-12",
    item="000000",
    sa="N",
    measure="4",
    variation="ANR",
    freq="M",
    timeout=60
):
    base = "https://data-api.ecb.europa.eu/service/data"
    key = f"{freq}.{'+'.join(countries)}.{sa}.{item}.{measure}.{variation}"
    params = {"format": "csvdata", "startPeriod": start, "endPeriod": end}
    url = f"{base}/ICP/{key}"

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()

    raw = pd.read_csv(StringIO(r.text))

    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"])
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")

    country_col = "REF_AREA" if "REF_AREA" in raw.columns else None
    if country_col is None:
        raise ValueError("REF_AREA column not found in ECB response.")

    panel = (
        raw.pivot_table(index="TIME_PERIOD", columns=country_col, values="OBS_VALUE", aggfunc="last")
        .sort_index()
    )

    panel.index = panel.index.to_period("M").to_timestamp(how="start")
    return panel


# ------------------------------------------------------------
# Fetch Ukraine CPI raw data from SSSU
# ------------------------------------------------------------
def fetch_ukraine_cpi_prev_month_raw(
    start="2000-01",
    end="2025-12",
    timeout=60
):
    """
    Fetch Ukraine CPI (previous month = 100) from the SSSU SDMX API v3.
    Returns a raw dataframe with TIME_PERIOD and OBS_VALUE.
    """
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

    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()

    raw = pd.read_csv(StringIO(r.text), dtype=str)

    raw = raw.loc[
        raw["TIME_PERIOD"].astype(str).str.match(r"^\d{4}-M\d{2}$", na=False)
        & raw["OBS_VALUE"].notna()
    ].copy()

    return raw


def ua_raw_to_monthly_series(ua_raw: pd.DataFrame) -> pd.Series:
    """
    Convert raw Ukraine CPI dataframe into a monthly series indexed by month-start.
    """
    if "TIME_PERIOD" not in ua_raw.columns or "OBS_VALUE" not in ua_raw.columns:
        raise ValueError("ua_raw must contain TIME_PERIOD and OBS_VALUE")

    s = ua_raw[["TIME_PERIOD", "OBS_VALUE"]].copy()
    s["TIME_PERIOD"] = s["TIME_PERIOD"].astype(str).str.strip()
    s = s[s["TIME_PERIOD"].str.match(r"^\d{4}-M\d{2}$", na=False)]

    s["TIME_PERIOD"] = pd.to_datetime(
        s["TIME_PERIOD"].str.replace(r"^(\d{4})-M(\d{2})$", r"\1-\2-01", regex=True),
        errors="coerce"
    )

    s["OBS_VALUE"] = pd.to_numeric(
        s["OBS_VALUE"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

    s = s.dropna(subset=["TIME_PERIOD", "OBS_VALUE"]).sort_values("TIME_PERIOD")
    out = s.set_index("TIME_PERIOD")["OBS_VALUE"].rename("UA_IDX_PREV_MONTH_100")
    out = out.groupby(level=0).last()

    return out


def cpi_prev_month_index_to_yoy_inflation(idx_prev_month_100: pd.Series) -> pd.Series:
    """
    Convert CPI index (previous month = 100) into year-over-year inflation.
    """
    monthly_factor = (idx_prev_month_100 / 100.0).astype(float)
    yoy_factor = monthly_factor.rolling(12).apply(np.prod, raw=True)
    return ((yoy_factor - 1.0) * 100.0).rename("UA")


# ------------------------------------------------------------
# Pioneer Detection Method
# ------------------------------------------------------------
def smooth_panel(df, window=3):
    return df.rolling(window=window, min_periods=1).mean()


def compute_pioneer_scores(smoothed_df):
    """
    Compute pioneer scores for each country over time.
    """
    countries = smoothed_df.columns.tolist()
    scores = pd.DataFrame(index=smoothed_df.index, columns=countries, dtype=float)

    for c in countries:
        peers = [x for x in countries if x != c]

        s_i = smoothed_df[c]
        s_m = smoothed_df[peers].mean(axis=1)

        dist_t = (s_i - s_m).abs()
        dist_tm1 = dist_t.shift(1)

        distance_dummy = (dist_t < dist_tm1).astype(float)

        theta_i = s_i.diff().abs()
        theta_m = s_m.diff().abs()

        orientation_dummy = (theta_m > theta_i).astype(float)

        denom = theta_m + theta_i
        weight = np.where(
            denom > 0,
            distance_dummy * orientation_dummy * (theta_m / denom),
            0.0
        )

        scores[c] = weight

    return scores.fillna(0.0)


# ------------------------------------------------------------
# Granger comparison: X -> target
# ------------------------------------------------------------
def granger_rank(df, target="UA", maxlag=6):
    out = []

    for c in df.columns:
        if c == target:
            continue

        data_gc = df[[target, c]].dropna()

        try:
            res = grangercausalitytests(data_gc, maxlag=maxlag, verbose=False)
            min_p = min(res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1))
            out.append({"country": c, "min_pvalue": min_p})
        except Exception:
            out.append({"country": c, "min_pvalue": np.nan})

    return pd.DataFrame(out).sort_values("min_pvalue")


# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------
if __name__ == "__main__":

    countries = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]
    target_country = "UA"

    infl_panel = fetch_ecb_hicp_inflation_panel(
        countries=countries,
        start="2000-01",
        end="2025-12"
    )

    # Fetch and build Ukraine y/y inflation
    ua_raw = fetch_ukraine_cpi_prev_month_raw(start="2000-01", end="2025-12")
    ua_idx = ua_raw_to_monthly_series(ua_raw)
    ua_yoy = cpi_prev_month_index_to_yoy_inflation(ua_idx)

    # Standardize indices to month-start
    infl_panel = infl_panel.copy()
    infl_panel.index = pd.to_datetime(infl_panel.index).to_period("M").to_timestamp(how="start")
    ua_yoy.index = pd.to_datetime(ua_yoy.index).to_period("M").to_timestamp(how="start")

    # Join Ukraine into the panel
    infl_panel = infl_panel.join(ua_yoy, how="left")

    # Keep panel sorted
    df = infl_panel.copy().sort_index()

    # Smooth data
    smoothed = smooth_panel(df, window=3)

    # Compute pioneer scores
    scores = compute_pioneer_scores(smoothed)

    # Ukraine-specific outputs
    ua_score = scores[target_country]
    ua_peer_avg = smoothed[[c for c in smoothed.columns if c != target_country]].mean(axis=1)

    # Average pioneer score ranking
    ranking = scores.mean().sort_values(ascending=False).rename("avg_pioneer_score")

    # Granger comparison
    granger_table = granger_rank(df.dropna(), target=target_country, maxlag=6)

    # --------------------------------------------------------
    # Console outputs
    # --------------------------------------------------------
    print("\n=== Average Pioneer Scores ===")
    print(ranking.to_string())

    print("\n=== Ukraine Pioneer Score Summary ===")
    print(ua_score.describe().to_string())

    print("\n=== Granger Causality Ranking: X -> UA ===")
    print(granger_table.to_string(index=False))

    ua_rank = ranking.index.get_loc(target_country) + 1
    print("\n=== Interpretation ===")
    print(f"Ukraine ranks #{ua_rank} out of {len(ranking)} countries by average pioneer score.")

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    plt.figure(figsize=(12, 6))
    for country in df.columns:
        plt.plot(df.index, df[country], label=country, linewidth=1)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("ECB HICP Inflation Panel with Ukraine")
    plt.ylabel("Inflation rate (y/y, %)")
    plt.xlabel("Time")
    plt.legend(ncol=3, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(smoothed.index, smoothed[target_country], label="UA (smoothed)", linewidth=2)
    plt.plot(smoothed.index, ua_peer_avg, label="Peer average (excluding UA)", linewidth=2)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("UA vs peer average")
    plt.xlabel("Time")
    plt.ylabel("Inflation rate (y/y, %)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(scores.index, ua_score, linewidth=1.8)
    plt.title("Pioneer score over time: UA")
    plt.xlabel("Time")
    plt.ylabel("Pioneer score")
    plt.tight_layout()
    plt.show()


# Interpretation
# Ukraine appears to be visually different from the peer group as seen in the plots. Ukraine's inflation is far above the peer average 
# for long periods of time and has very large spikes. However, the results suggest that Ukraine does not have a strong leading role in 
# inflation dynamics relative to its peers in the sample. Ukraine receives a low average pioneer score and a last place ranking, which 
# suggests that countries do not converge towards Ukraine's inflation path. The pioneer score for Ukraine is also zero in most months and 
# only positive for a few episodes, further indicating that any pioneer type role is weak and temporary.  Thus, Ukraine appears to be an outlier, 
# given its volatile inflation dynamics, rather than a pioneer country. 
