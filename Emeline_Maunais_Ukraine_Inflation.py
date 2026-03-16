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

infl_panel = infl_panel.dropna()

# 1) Type of infl_panel
print("infl_panel is a data frame.")

# 2) Type of "UA" inflation
print(infl_panel['UA'].dtype)
print("UA inflation is a float")

# 3) Statistical description (dates, frequency, mean, std) per country
import pandas as pd

stats = []

for c in infl_panel.columns:
    s = infl_panel[c].dropna()
    stats.append([
        c,
        infl_panel.index.min(),
        infl_panel.index.max(),
        pd.infer_freq(infl_panel.index),
        s.mean(),
        s.std()
    ])

table = pd.DataFrame(
    stats,
    columns=['Country','Start','End','Frequency','Mean','Std']
)

print(table.to_latex(index=False))

# 4) Correlation between UA and FR

corr = infl_panel['UA'].corr(infl_panel['FR']) * 100
print(corr)
print("Ukraine's and France's inflation correlation is equal to 0,4813%. They are not correlated.") 

# 5) Test for correlation significance for each pair UA-FR and FR-DE
from scipy.stats import pearsonr

r_ua_fr, p_ua_fr = pearsonr(infl_panel['UA'], infl_panel['FR'])
r_fr_de, p_fr_de = pearsonr(infl_panel['FR'], infl_panel['DE'])

print("UA-FR: r=", r_ua_fr, "p=", p_ua_fr)
print("FR-DE: r=", r_fr_de, "p=", p_fr_de)

print("Inflation correlation between Ukraine and France is not significant")
print("Inflation correlation between France and Germany is significant to a 1% threshold")

# 6) Test for time series BREAK for UA and FR

from statsmodels.stats.diagnostic import breaks_cusumolsresid
import statsmodels.api as sm

def cusum_test(series, name):
    y = series.dropna()
    X = sm.add_constant(range(len(y)))

    model = sm.OLS(y, X).fit()
    stat, pval, _ = breaks_cusumolsresid(model.resid)

    print(f"{name} → CUSUM p-value:", pval)

cusum_test(infl_panel['UA'], "UA")
cusum_test(infl_panel['FR'], "FR")

# 7) How persistent is inflation in UA? In FR? 

from statsmodels.tsa.stattools import adfuller

def adf_test(s, name):
    result = adfuller(s.dropna(), regression='c')
    print(f"\n{name}")
    print("ADF stat:", result[0])
    print("p-value:", result[1])

adf_test(infl_panel['UA'], "UA")
adf_test(infl_panel['FR'], "FR")

print("ADF stat is different from 1, so inflation is not persistent neither for France nor for Ukraine")

# ------------------------------------------------------------
# Plot the inflation panel (one line per country)
# Assumes `infl_panel` is the wide DataFrame returned above:
#   index   = datetime (monthly)
#   columns = country codes
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

for country in infl_panel.columns:
    plt.plot(infl_panel.index, infl_panel[country], label=country, linewidth=1)

plt.axhline(0, color="black", linewidth=0.8, linestyle="--")

plt.xlabel("Time")
plt.ylabel("Inflation rate (y/y, %)")
plt.title("HICP Inflation Panel (ECB Data Portal)")
plt.legend(ncol=3, fontsize=9, frameon=False)
plt.tight_layout()
plt.show()


# -------------------------
# 0) Prepare data
# -------------------------
df = infl_panel.copy().sort_index().dropna()

# -------------------------
# 1) ADF unit-root test (levels only)
# -------------------------
print("\n=== ADF unit-root tests (levels) ===")

adf_results = []
for c in df.columns:
    stat, pval, _, _, _, _ = adfuller(df[c], autolag="AIC")
    adf_results.append({
        "country": c,
        "ADF_stat": stat,
        "pvalue": pval
    })

adf_table = pd.DataFrame(adf_results).sort_values("pvalue")
print(adf_table.to_string(index=False))

# -------------------------
# 2) Granger causality: X → UA
#    (bivariate, simple ranking)
# -------------------------
maxlag = 6   # keep small for undergrads

print("\n=== Granger causality tests: X → UA ===")

granger_out = []

for c in df.columns:
    if c == "UA":
        continue

    data_gc = df[["UA", c]]

    try:
        res = grangercausalitytests(data_gc, maxlag=maxlag, verbose=False)

        # keep the smallest p-value across lags
        min_p = min(res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1))

        granger_out.append({
            "country": c,
            "min_pvalue": min_p
        })

    except Exception as e:
        print(f"Granger test failed for {c}: {e}")

granger_rank = (
    pd.DataFrame(granger_out)
    .sort_values("min_pvalue")
    .reset_index(drop=True)
)

print("\n=== Ranking of countries by Granger causality for UA ===")
print(granger_rank.to_string(index=False))

# -------------------------
# 3) Simple VAR with BIC
#    (UA + top 2 predictors)
# -------------------------
top_countries = granger_rank["country"].iloc[:2].tolist()
var_vars = ["UA"] + top_countries

print("\nVAR variables:", var_vars)

X_var = df[var_vars]

# lag selection by BIC
model = VAR(X_var)
lag_selection = model.select_order(maxlags=6)
p = lag_selection.selected_orders["bic"]
p = max(1, p)

print("\n=== VAR lag selection (BIC) ===")
print(lag_selection.summary())
print(f"Selected lag order p = {p}")

# estimate VAR
var_res = model.fit(p)
print("\n=== VAR estimation results ===")
print(var_res.summary())

