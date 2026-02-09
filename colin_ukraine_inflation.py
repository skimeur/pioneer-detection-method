# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 15:17:11 2026

@author: colin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECB HICP Inflation Panel — ADF, Granger Causality (BE), and VAR (BIC)
====================================================================

Purpose
-------
This script is a compact teaching example showing how to:
1) download a cross-country inflation panel (HICP, y/y) from the ECB Data Portal API,
2) run a basic unit-root check (ADF test) on each country series,
3) rank countries by Granger causality for Belgian inflation (BE),
4) estimate a small VAR in levels with lag order selected by BIC.

Data
----
Source: ECB Data Portal (SDMX 2.1 REST API), dataset "ICP".
Series: Monthly HICP inflation, annual rate of change (y/y), headline all-items.
Endpoint pattern:
    https://data-api.ecb.europa.eu/service/data/ICP/{key}?format=csvdata&startPeriod=...&endPeriod=...

Econometric workflow (undergraduate level)
------------------------------------------
- ADF test (H0: unit root) applied to inflation rates in levels (no differencing here).
- Granger causality tests (bivariate): does country X help predict BE?
  Ranking uses the minimum p-value across lags 1..maxlag.
- Small VAR: variables = [BE + top 2 countries], lag p chosen by BIC.

Outputs
-------
- Line plot of the inflation panel.
- Console tables:
  * ADF statistics and p-values by country
  * Granger-causality ranking for BE (min p-value across lags)
  * VAR lag selection summary (BIC) and VAR estimation summary

Dependencies
------------
requests, pandas, numpy, matplotlib, statsmodels

Author
------
Eric Vansteenberghe (Banque de France)
Created: 2026-01-24
License: MIT (recommended for teaching code)

Notes
-----
This is a pedagogical script. It uses the latest revised data (not real-time vintages)
and applies simple complete-case handling (drop rows with missing values).
"""


import requests
import pandas as pd
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

def fetch_ecb_hicp_inflation_panel(
    countries,
    start="1997-01-01",
    end=None,
    item="000000",   # headline all-items HICP
    sa="N",          # neither seasonally nor working-day adjusted
    measure="4",     # percentage change (as used in ICP keys)
    variation="ANR", # annual rate of change
    freq="M",
    timeout=60
):
    """
    Fetch a monthly cross-country panel of HICP inflation (annual rate of change)
    from the ECB Data Portal (ICP dataflow).

    Returns
    -------
    panel_wide : pd.DataFrame
        Index: pandas datetime (monthly)
        Columns: country codes (e.g., DE, FR, IT)
        Values: inflation rate (float)
    raw_long : pd.DataFrame
        Long format with series dimensions, TIME_PERIOD and OBS_VALUE.
    """
    # ECB Data Portal SDMX REST endpoint
    base = "https://data-api.ecb.europa.eu/service/data"

    # Build SDMX series key with OR operator (+) over countries
    # Dimension order for ICP: FREQ.REF_AREA.ADJ.ITEM.UNIT/MEASURE.VARIATION
    # Example keys are shown in the ECB portal for ICP datasets.
    key = f"{freq}.{'+'.join(countries)}.{sa}.{item}.{measure}.{variation}"

    params = {"format": "csvdata", "startPeriod": start}
    if end is not None:
        params["endPeriod"] = end

    url = f"{base}/ICP/{key}"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()

    raw = pd.read_csv(StringIO(r.text))

    # Keep standard SDMX columns
    if "TIME_PERIOD" not in raw.columns or "OBS_VALUE" not in raw.columns:
        raise ValueError(f"Unexpected response format. Columns: {list(raw.columns)}")

    # Identify the country dimension column (typically REF_AREA)
    # If REF_AREA is missing, fall back to any column that looks like a geo dimension.
    country_col = "REF_AREA" if "REF_AREA" in raw.columns else None
    if country_col is None:
        for cand in ["GEO", "LOCATION", "COUNTRY", "REF_AREA"]:
            if cand in raw.columns:
                country_col = cand
                break
    if country_col is None:
        # Last resort: infer as the first non-standard column
        standard = {"TIME_PERIOD", "OBS_VALUE", "OBS_STATUS", "OBS_CONF", "UNIT_MULT", "DECIMALS"}
        nonstandard = [c for c in raw.columns if c not in standard]
        if not nonstandard:
            raise ValueError("Could not infer the country column from the response.")
        country_col = nonstandard[0]

    # Parse time and values
    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"])
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")

    # Wide panel: time x country
    panel = (
        raw.pivot_table(index="TIME_PERIOD", columns=country_col, values="OBS_VALUE", aggfunc="last")
        .sort_index()
    )

    return panel, raw


# -------------------------
# Example usage
# -------------------------
countries = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]
infl_panel, infl_long = fetch_ecb_hicp_inflation_panel(
    countries=countries,
    start="2000-01",
    end="2025-12"   # optional
)

# -----------------------------------
# Fetch Ukraine inflation time series

def fetch_ukraine_cpi_prev_month_raw(
    start="2000-01",
    end="2025-12",
    timeout=60
):
    """
    Fetch Ukraine CPI (previous month = 100) from the SSSU SDMX API v3 and return
    the raw SDMX-CSV as a DataFrame (no date/numeric parsing).
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

    # --- MINIMAL FIX: some responses include metadata rows.
    # Keep only rows that look like monthly observations and have OBS_VALUE.
    raw = raw.loc[
        raw["TIME_PERIOD"].astype(str).str.match(r"^\d{4}-M\d{2}$", na=False)
        & raw["OBS_VALUE"].notna()
    ].copy()

    return raw


# Example
ua_raw = fetch_ukraine_cpi_prev_month_raw(start="2000-01", end="2025-12")
print(ua_raw.head())
print(ua_raw["TIME_PERIOD"].unique()[:12])
print(ua_raw["OBS_VALUE"].unique()[:12])



# ua_raw is your DataFrame as read from the SDMX-CSV response
# (i.e., it already has columns like TIME_PERIOD, OBS_VALUE)

def ua_raw_to_monthly_series(ua_raw: pd.DataFrame) -> pd.Series:
    """
    Build a clean monthly time series from SSSU SDMX-CSV raw output.

    Input:
      ua_raw: DataFrame with at least TIME_PERIOD like '2000-M01' and OBS_VALUE strings.

    Output:
      pd.Series indexed by month-start Timestamp, name='UA_IDX_PREV_MONTH_100'
    """
    if "TIME_PERIOD" not in ua_raw.columns or "OBS_VALUE" not in ua_raw.columns:
        raise ValueError(f"ua_raw must contain TIME_PERIOD and OBS_VALUE. Columns: {list(ua_raw.columns)}")

    s = ua_raw[["TIME_PERIOD", "OBS_VALUE"]].copy()

    # Keep only true monthly tokens like YYYY-Mmm (defensive)
    s["TIME_PERIOD"] = s["TIME_PERIOD"].astype(str).str.strip()
    s = s[s["TIME_PERIOD"].str.match(r"^\d{4}-M\d{2}$", na=False)]

    # Convert 'YYYY-Mmm' -> Timestamp at month start
    # Example: '2000-M01' -> '2000-01-01'
    s["TIME_PERIOD"] = pd.to_datetime(
        s["TIME_PERIOD"].str.replace(r"^(\d{4})-M(\d{2})$", r"\1-\2-01", regex=True),
        errors="coerce"
    )

    # Values
    s["OBS_VALUE"] = pd.to_numeric(s["OBS_VALUE"].astype(str).str.replace(",", ".", regex=False),
                                   errors="coerce")

    s = s.dropna(subset=["TIME_PERIOD", "OBS_VALUE"]).sort_values("TIME_PERIOD")

    out = s.set_index("TIME_PERIOD")["OBS_VALUE"].rename("UA_IDX_PREV_MONTH_100")

    # If duplicates exist for a month (shouldn't, but safe): keep last
    out = out.groupby(level=0).last()

    return out

# Build the monthly series (prev month = 100)
ua_idx = ua_raw_to_monthly_series(ua_raw)

# Optional: restrict window (month-start)
ua_idx = ua_idx.loc["2000-01-01":"2025-12-01"]

# If you still need y/y inflation (%):
def cpi_prev_month_index_to_yoy_inflation(idx_prev_month_100: pd.Series) -> pd.Series:
    monthly_factor = (idx_prev_month_100 / 100.0).astype(float)
    yoy_factor = monthly_factor.rolling(12).apply(np.prod, raw=True)
    return ((yoy_factor - 1.0) * 100.0).rename("UA")

ua_yoy = cpi_prev_month_index_to_yoy_inflation(ua_idx)

# Ensure month-start indices match
infl_panel = infl_panel.copy()
infl_panel.index = pd.to_datetime(infl_panel.index).to_period("M").to_timestamp(how="start")
ua_yoy.index = pd.to_datetime(ua_yoy.index).to_period("M").to_timestamp(how="start")

infl_panel = infl_panel.join(ua_yoy, how="left")

#1 type of infl_panel
print("infl_panel is a dataframe")

#2 type of UA inflation
print("UA inflation is a pandas Series")

#3 Stat desc
rows = []

for c in infl_panel.columns:
    s = infl_panel[c].dropna()
    rows.append({
        "Country": c,
        "Start date": s.index.min().strftime("%Y-%m"),
        "End date": s.index.max().strftime("%Y-%m"),
        "Frequency": "Monthly",
        "Mean inflation (%)": s.mean(),
        "Std. dev. (%)": s.std()
    })

desc_table = pd.DataFrame(rows)

print("\n=== Table 1: Descriptive statistics of inflation ===")
print(desc_table.to_latex(
    index=False,
    float_format="%.2f",
    caption="Descriptive statistics of monthly year-on-year inflation rates",
    label="tab:inflation_desc",
    column_format="lccccc"
))

#4 corr in %

corr_ua_fr = infl_panel["UA"].corr(infl_panel["FR"]) * 100
corr_fr_de = infl_panel["FR"].corr(infl_panel["DE"]) * 100

print("\n4) Correlations (in %)")
print(f"Correlation UA–FR: {corr_ua_fr:.2f}%")
print(f"Correlation FR–DE: {corr_fr_de:.2f}%")

#5 significance 

from scipy.stats import pearsonr

ua_fr = infl_panel[["UA", "FR"]].dropna()
r_ua_fr, p_ua_fr = pearsonr(ua_fr["UA"], ua_fr["FR"])

fr_de = infl_panel[["FR", "DE"]].dropna()
r_fr_de, p_fr_de = pearsonr(fr_de["FR"], fr_de["DE"])

print("\n5) Correlation significance tests")
print(f"UA–FR: r = {r_ua_fr:.3f}, p-value = {p_ua_fr:.4f}")
print(f"FR–DE: r = {r_fr_de:.3f}, p-value = {p_fr_de:.4f}")

#6 test for time series

from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import numpy as np

def cusum_break_test(series, name):
    y = series.dropna()
    X = np.arange(len(y))
    X = add_constant(X)
    res = OLS(y, X).fit()
    stat, pval, crit = breaks_cusumolsresid(res.resid, ddof=2)
    print(f"{name}: CUSUM stat = {stat:.3f}, p-value = {pval:.4f}")

print("\n6) Structural break tests (CUSUM)")
cusum_break_test(infl_panel["UA"], "UA")
cusum_break_test(infl_panel["FR"], "FR")

#7 inflation persistence using ADF

from statsmodels.tsa.stattools import adfuller

print("\n7) Inflation persistence (ADF test)")

for c in ["UA", "FR"]:
    series = infl_panel[c].dropna()
    stat, pval, _, _, _, _ = adfuller(series, autolag="AIC")
    print(f"{c}: ADF stat = {stat:.3f}, p-value = {pval:.4f}")



