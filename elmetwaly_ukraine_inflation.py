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

print(infl_panel.mean())
print(infl_panel.std())
print(infl_panel.min())
print(infl_panel.max())

infl_panel.index = pd.to_datetime(infl_panel.index).to_period("M").to_timestamp(how="start")
ua_yoy.index = pd.to_datetime(ua_yoy.index).to_period("M").to_timestamp(how="start")

infl_panel = infl_panel.join(ua_yoy, how="left")


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

infl_panel.corr()
infl_panel.columns



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



#question 1 : What is infl panel? print answer 
#inf_panel is a pandas Dataframe
print(type(infl_panel))



#question 2 : 
print(type(infl_panel['UA'].dtype))   
print('UA is a float')  
       
# UA inflation is a pandas Series
# UA inflation is float

#question 3 : 


print("\n=== Descriptive statistics per country ===")

# Date range
print("Start date:", infl_panel.index.min())
print("End date:", infl_panel.index.max())
#

# Frequency (monthly expected)
print("Frequency guess:", pd.infer_freq(infl_panel.index))

# Mean and standard deviation per country
desc_stats = pd.DataFrame({
    "mean": infl_panel.mean(),
    "std": infl_panel.std()
})

print(desc_stats)

#question 4


print("\n=== Correlations (in %) ===")

# Use df (cleaned panel) if available, otherwise infl_panel
data_corr = df if "df" in globals() else infl_panel

corr_ua_fr = data_corr["UA"].corr(data_corr["FR"]) * 100
corr_fr_de = data_corr["FR"].corr(data_corr["DE"]) * 100

print(f"Correlation UA–FR (%): {corr_ua_fr:.2f}")
print(f"Correlation FR–DE (%): {corr_fr_de:.2f}")


#question 5

from scipy.stats import pearsonr


print("\n=== Correlation significance tests ===")

# Drop missing values pairwise
ua_fr = df[["UA", "FR"]].dropna()
fr_de = df[["FR", "DE"]].dropna()

# Pearson tests
corr_ua_fr, pval_ua_fr = pearsonr(ua_fr["UA"], ua_fr["FR"])
corr_fr_de, pval_fr_de = pearsonr(fr_de["FR"], fr_de["DE"])

print(f"UA–FR: corr = {corr_ua_fr:.3f}, p-value = {pval_ua_fr:.4g}")
print(f"FR–DE: corr = {corr_fr_de:.3f}, p-value = {pval_fr_de:.4g}")

#for UA-FRA : this is not signficant because we have a high p-value 
#for FR-DE : this is significant (a very low p-value )


#question  6

from statsmodels.stats.diagnostic import breaks_cusumolsresid



print("\n=== Structural break tests (CUSUM) ===")

def cusum_test(series, name):
    # Simple AR(1) regression for residuals
    y = series.dropna()
    X = y.shift(1).dropna()
    y = y.loc[X.index]

    X = add_constant(X)
    model = OLS(y, X).fit()

    stat, pval, crit = breaks_cusumolsresid(model.resid)

    print(f"{name}:")
    print(f"  CUSUM stat = {stat:.3f}")
    print(f"  p-value   = {pval:.4f}")
    print()

# Run for Ukraine and France
cusum_test(df["UA"], "UA")
cusum_test(df["FR"], "FR")


#question 7


print("\n=== Persistence of inflation (ADF test) ===")

def adf_test(series, name):
    y = series.dropna()
    stat, pval, _, _, crit, _ = adfuller(y, autolag="AIC")

    print(f"{name}:")
    print(f"  ADF statistic = {stat:.3f}")
    print(f"  p-value       = {pval:.4f}")
    print("  Critical values:", crit)
    print()

adf_test(df["UA"], "UA")
adf_test(df["FR"], "FR")

#UA : unit root not rejected : persistent 
#FR: unit root rejected : stationary


#question 8





print("\n=== Granger causality tests: UA → X ===")

maxlag = 6

ua_to_others = []

for c in df.columns:
    if c == "UA":
        continue

    data_gc = df[[c, "UA"]].dropna()

    try:
        res = grangercausalitytests(data_gc, maxlag=maxlag, verbose=False)

        min_p = min(res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1))

        ua_to_others.append({
            "country": c,
            "min_pvalue": min_p
        })

    except Exception as e:
        print(f"UA → {c} failed: {e}")

ua_rank = (
    pd.DataFrame(ua_to_others)
    .sort_values("min_pvalue")
    .reset_index(drop=True)
)

print("\nRanking UA → others:")
print(ua_rank)

#When?

pre_2022 = df.loc[:"2021-12-01"]
post_2022 = df.loc["2022-01-01":]

def ua_granger_period(data, label):
    out = []

    for c in data.columns:
        if c == "UA":
            continue

        d = data[[c, "UA"]].dropna()

        try:
            res = grangercausalitytests(d, maxlag=6, verbose=False)
            min_p = min(res[l][0]["ssr_ftest"][1] for l in range(1, 7))

            out.append({"country": c, "min_pvalue": min_p})

        except:
            pass

    out = pd.DataFrame(out).sort_values("min_pvalue")

    print(f"\nUA → others ({label}):")
    print(out)

ua_granger_period(pre_2022, "before 2022")
ua_granger_period(post_2022, "after 2022")


#UA is not a pioneer before 2022, but becomes a pioneer after 2022,
#since its inflation Granger-causes several European countries in the post-2022 period (e.g. FI, AT, IT, PT, BE, GR, DE, FR




    
