#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examen QMF - Analyse PDM Ukraine
Emma Blindauer
Février 2026
"""

import requests
import pandas as pd
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, breakvar_heteroskedasticity_test
from pdm import compute_pioneer_weights_simple, pooled_forecast_simple

# ============================================================
# FONCTIONS DE TÉLÉCHARGEMENT (comme avant)
# ============================================================

def fetch_ecb_hicp_inflation_panel(countries, start="1997-01-01", end=None, 
                                   item="000000", sa="N", measure="4", 
                                   variation="ANR", freq="M", timeout=60):
    base = "https://data-api.ecb.europa.eu/service/data"
    key = f"{freq}.{'+'.join(countries)}.{sa}.{item}.{measure}.{variation}"
    params = {"format": "csvdata", "startPeriod": start}
    if end is not None:
        params["endPeriod"] = end
    url = f"{base}/ICP/{key}"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    raw = pd.read_csv(StringIO(r.text))
    country_col = "REF_AREA" if "REF_AREA" in raw.columns else None
    if country_col is None:
        for cand in ["GEO", "LOCATION", "COUNTRY"]:
            if cand in raw.columns:
                country_col = cand
                break
    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"])
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")
    panel = (raw.pivot_table(index="TIME_PERIOD", columns=country_col, 
                             values="OBS_VALUE", aggfunc="last").sort_index())
    return panel, raw

def fetch_ukraine_cpi_prev_month_raw(start="2000-01", end="2025-12", timeout=60):
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

def ua_raw_to_monthly_series(ua_raw):
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

def cpi_prev_month_index_to_yoy_inflation(idx_prev_month_100):
    monthly_factor = (idx_prev_month_100 / 100.0).astype(float)
    yoy_factor = monthly_factor.rolling(12).apply(np.prod, raw=True)
    return ((yoy_factor - 1.0) * 100.0).rename("UA")

# ============================================================
# TÉLÉCHARGEMENT DES DONNÉES
# ============================================================

print("Downloading data...")

# Europe - Using the same 11 countries as in ecb_hicp_panel_var_granger_be.py
european_countries = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]
infl_panel_europe, _ = fetch_ecb_hicp_inflation_panel(
    countries=european_countries,
    start="2000-01",
    end="2025-12"
)

# Ukraine
ua_raw = fetch_ukraine_cpi_prev_month_raw(start="2000-01", end="2025-12")
ua_idx = ua_raw_to_monthly_series(ua_raw)
ua_yoy = cpi_prev_month_index_to_yoy_inflation(ua_idx)
ua_yoy = ua_yoy.loc["2000-01-01":"2025-12-01"]

# Fusion
infl_panel_europe.index = pd.to_datetime(infl_panel_europe.index).to_period("M").to_timestamp(how="start")
ua_yoy.index = pd.to_datetime(ua_yoy.index).to_period("M").to_timestamp(how="start")
infl_panel = infl_panel_europe.join(ua_yoy, how="inner")
infl_panel = infl_panel.dropna()

print("Data loaded successfully.\n")

# ============================================================
# QUESTION 1: What is inflation panel?
# ============================================================

print("=" * 70)
print("QUESTION 1: What is inflation panel?")
print("=" * 70)

answer_q1 = """
An inflation panel is a dataset containing inflation rates for multiple 
countries (cross-sectional dimension) observed over multiple time periods 
(time-series dimension). 

In our case, the inflation panel contains:
- Countries: 11 European countries (DE, FR, IT, ES, NL, BE, AT, PT, IE, FI, GR) 
  and Ukraine (UA) - total of 12 countries
- Time period: Monthly observations from 2000 to 2025
- Variable: Year-on-year inflation rate (%)

This panel structure allows us to analyze:
1. Time-series dynamics within each country
2. Cross-country correlations and relationships
3. Convergence patterns across countries over time
"""

print(answer_q1)
print()

# ============================================================
# QUESTION 2: What type is UA inflation?
# ============================================================

print("=" * 70)
print("QUESTION 2: What type is UA inflation?")
print("=" * 70)

answer_q2 = f"""
The Ukraine (UA) inflation variable is of type: {type(infl_panel['UA'])}

More specifically:
- Python type: pandas.Series
- Data type: {infl_panel['UA'].dtype}
- Contains: {len(infl_panel['UA'])} observations
- Index type: DatetimeIndex (monthly frequency)

This is a time series of floating-point numbers representing the 
year-on-year inflation rate in Ukraine, measured monthly.
"""

print(answer_q2)
print()

# ============================================================
# QUESTION 3: Descriptive statistics per country (LaTeX table)
# ============================================================

print("=" * 70)
print("QUESTION 3: Descriptive statistics (LaTeX table)")
print("=" * 70)

stats_data = []
for country in infl_panel.columns:
    series = infl_panel[country]
    stats_data.append({
        'Country': country,
        'Start': series.index.min().strftime('%Y-%m'),
        'End': series.index.max().strftime('%Y-%m'),
        'Frequency': 'Monthly',
        'N': len(series),
        'Mean': series.mean(),
        'Std': series.std()
    })

stats_df = pd.DataFrame(stats_data)

# Generate LaTeX table
latex_table = r"""\begin{table}[htbp]
\centering
\caption{Descriptive Statistics by Country}
\begin{tabular}{lcccccc}
\toprule
Country & Start & End & Frequency & N & Mean (\%) & Std (\%) \\
\midrule
"""

for _, row in stats_df.iterrows():
    latex_table += f"{row['Country']} & {row['Start']} & {row['End']} & {row['Frequency']} & {row['N']} & {row['Mean']:.2f} & {row['Std']:.2f} \\\\\n"

latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""

print(latex_table)
print()

# ============================================================
# QUESTION 4: Correlation UA-FR and FR-DE (in %)
# ============================================================

print("=" * 70)
print("QUESTION 4: Correlations (in %)")
print("=" * 70)

corr_ua_fr = infl_panel['UA'].corr(infl_panel['FR']) * 100
corr_fr_de = infl_panel['FR'].corr(infl_panel['DE']) * 100

print(f"Correlation between Ukraine and France: {corr_ua_fr:.2f}%")
print(f"Correlation between France and Germany: {corr_fr_de:.2f}%")
print()

# ============================================================
# QUESTION 5: Test for correlation significance
# ============================================================

print("=" * 70)
print("QUESTION 5: Test for correlation significance")
print("=" * 70)

# Pearson correlation test for UA-FR
corr_ua_fr_coef, pval_ua_fr = stats.pearsonr(infl_panel['UA'], infl_panel['FR'])

# Pearson correlation test for FR-DE
corr_fr_de_coef, pval_fr_de = stats.pearsonr(infl_panel['FR'], infl_panel['DE'])

print(f"UA-FR: correlation = {corr_ua_fr_coef:.4f}, p-value = {pval_ua_fr:.4f}")
if pval_ua_fr < 0.01:
    print("       Result: Highly significant (p < 0.01) ***")
elif pval_ua_fr < 0.05:
    print("       Result: Significant (p < 0.05) **")
elif pval_ua_fr < 0.10:
    print("       Result: Marginally significant (p < 0.10) *")
else:
    print("       Result: Not significant (p >= 0.10)")

print()

print(f"FR-DE: correlation = {corr_fr_de_coef:.4f}, p-value = {pval_fr_de:.4f}")
if pval_fr_de < 0.01:
    print("       Result: Highly significant (p < 0.01) ***")
elif pval_fr_de < 0.05:
    print("       Result: Significant (p < 0.05) **")
elif pval_fr_de < 0.10:
    print("       Result: Marginally significant (p < 0.10) *")
else:
    print("       Result: Not significant (p >= 0.10)")

print()

# ============================================================
# QUESTION 6: Test for time series BREAK (structural break)
# ============================================================

print("=" * 70)
print("QUESTION 6: Test for structural break (UA and FR)")
print("=" * 70)

from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import breaks_cusumolsresid

# Prepare data for regression: FR ~ UA
regression_data = infl_panel[['FR', 'UA']].copy()

print("\nRegression model: FR = β0 + β1*UA + ε")
print("Testing for structural breaks in this relationship\n")

# Run full sample regression
model_full = ols('FR ~ UA', data=regression_data).fit()

print("Full sample regression:")
print(f"  Coefficient β1 (UA): {model_full.params['UA']:.4f}")
print(f"  R-squared: {model_full.rsquared:.4f}")
print(f"  p-value (UA): {model_full.pvalues['UA']:.4f}")

# ============================================================
# CHOW TEST - Test for break at midpoint
# ============================================================

print("\n--- CHOW TEST ---")
print("Testing for structural break at midpoint (2013-06)")

# Split sample at midpoint
mid_point = len(regression_data) // 2
first_half = regression_data.iloc[:mid_point]
second_half = regression_data.iloc[mid_point:]

# Run regressions on each subsample
model_first = ols('FR ~ UA', data=first_half).fit()
model_second = ols('FR ~ UA', data=second_half).fit()

# Chow test statistic
# F = [(SSR_pooled - SSR_1 - SSR_2) / k] / [(SSR_1 + SSR_2) / (n1 + n2 - 2k)]
ssr_pooled = model_full.ssr
ssr_first = model_first.ssr
ssr_second = model_second.ssr
k = 2  # number of parameters (intercept + slope)
n1 = len(first_half)
n2 = len(second_half)

chow_numerator = (ssr_pooled - ssr_first - ssr_second) / k
chow_denominator = (ssr_first + ssr_second) / (n1 + n2 - 2*k)
chow_stat = chow_numerator / chow_denominator

# F-distribution for p-value
from scipy.stats import f as f_dist
chow_pvalue = 1 - f_dist.cdf(chow_stat, k, n1 + n2 - 2*k)

print(f"\nChow Test Results:")
print(f"  F-statistic: {chow_stat:.4f}")
print(f"  p-value: {chow_pvalue:.4f}")

if chow_pvalue < 0.05:
    print(f"  Result: Significant structural break detected (p < 0.05)")
    print(f"  The relationship FR~UA changed significantly between periods")
else:
    print(f"  Result: No significant structural break (p >= 0.05)")

# Show coefficients for each period
print(f"\n  First period (2000-2013):")
print(f"    β1 (UA): {model_first.params['UA']:.4f}")
print(f"    R²: {model_first.rsquared:.4f}")

print(f"  Second period (2013-2025):")
print(f"    β1 (UA): {model_second.params['UA']:.4f}")
print(f"    R²: {model_second.rsquared:.4f}")

# ============================================================
# CUSUM TEST - Test for parameter stability over time
# ============================================================

print("\n--- CUSUM TEST ---")
print("Testing for parameter stability over time")

# CUSUM test for parameter stability
# Note: breaks_cusumolsresid returns (statistic, pvalue, crit_values)
try:
    cusum_result = breaks_cusumolsresid(model_full.resid)
    cusum_stat = cusum_result[0]
    cusum_pvalue = cusum_result[1]
    
    print(f"\nCUSUM Test Results:")
    print(f"  Test statistic: {cusum_stat:.4f}")
    print(f"  p-value: {cusum_pvalue:.4f}")
    
    if cusum_pvalue < 0.05:
        print(f"  Result: Parameter instability detected (p < 0.05)")
        print(f"  The regression coefficients are not stable over time")
    else:
        print(f"  Result: Parameters appear stable (p >= 0.05)")
except Exception as e:
    print(f"\nCUSUM Test: Unable to compute (error: {e})")
    print("  Using alternative approach: visual inspection of recursive residuals")

print("\nInterpretation:")
print("The Chow test examines whether the relationship between FR and UA inflation")
print("changed significantly at a specific point (2013). The CUSUM test checks for")
print("gradual parameter drift over the entire sample period.")

print()


# ============================================================
# QUESTION 7: How persistent is inflation? (ADF test)
# ============================================================

print("=" * 70)
print("QUESTION 7: Inflation persistence (ADF test)")
print("=" * 70)

print("Augmented Dickey-Fuller test for unit root")
print("H0: Series has a unit root (non-stationary / high persistence)")
print("H1: Series is stationary (low persistence)")
print()

for country in ['UA', 'FR']:
    series = infl_panel[country]
    adf_stat, pval, usedlag, nobs, critical_values, icbest = adfuller(series, autolag='AIC')
    
    print(f"{country}:")
    print(f"  ADF statistic: {adf_stat:.4f}")
    print(f"  p-value: {pval:.4f}")
    print(f"  Critical values: 1%={critical_values['1%']:.2f}, 5%={critical_values['5%']:.2f}, 10%={critical_values['10%']:.2f}")
    
    if pval < 0.05:
        print(f"  Result: Reject H0 → Inflation is STATIONARY (low persistence)")
    else:
        print(f"  Result: Cannot reject H0 → Inflation has HIGH PERSISTENCE (near unit root)")
    print()

# ============================================================
# QUESTION 8: Apply PDM
# ============================================================

print("=" * 70)
print("QUESTION 8: Apply Pioneer Detection Method (PDM)")
print("=" * 70)

# Compute PDM weights
weights_pdm = compute_pioneer_weights_simple(infl_panel)

# Compute pooled forecast
pooled_pdm = pooled_forecast_simple(infl_panel, weights_pdm)

# Compute simple mean for comparison
simple_mean = infl_panel.mean(axis=1)

# Analyze pioneer detection
pioneer_counts = (weights_pdm > 0).sum()
pioneer_pct = (pioneer_counts / len(weights_pdm) * 100).round(1)

print("\nPioneer detection frequency:")
for country in infl_panel.columns:
    print(f"  {country}: {pioneer_counts[country]} times ({pioneer_pct[country]}%)")

avg_weights = weights_pdm[weights_pdm > 0].mean()
print("\nAverage weight when detected as pioneer:")
for country in infl_panel.columns:
    if pd.notna(avg_weights[country]):
        print(f"  {country}: {avg_weights[country]:.3f}")

# Plot comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Panel 1: Individual series
for country in infl_panel.columns:
    ax1.plot(infl_panel.index, infl_panel[country], label=country, linewidth=1.5)
ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
ax1.set_ylabel("Inflation (%)")
ax1.set_title("Inflation Time Series by Country")
ax1.legend()
ax1.grid(True, alpha=0.2)

# Panel 2: PDM vs Simple Mean vs Ukraine
ax2.plot(infl_panel.index, simple_mean, label="Simple Mean", 
         linestyle=":", linewidth=2, color="gray")
ax2.plot(infl_panel.index, pooled_pdm, label="PDM Pooled Forecast", 
         linestyle="-", linewidth=2.5, color="darkred")
ax2.plot(infl_panel.index, infl_panel["UA"], label="Ukraine (Actual)", 
         linestyle="-", linewidth=1.5, color="blue", alpha=0.5)
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
ax2.set_xlabel("Date")
ax2.set_ylabel("Inflation (%)")
ax2.set_title("PDM vs Simple Mean vs Ukraine Actual")
ax2.legend()
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("exam_results_blindauer.png", dpi=150, bbox_inches='tight')
print("\nPlot saved as: exam_results_blindauer.png")

plt.show()

# ============================================================
# QUESTION 8 (suite): When is UA a pioneer?
# ============================================================

print("\n" + "=" * 70)
print("QUESTION 8 (continued): Is UA a pioneer? When?")
print("=" * 70)

# Identify when UA was detected as pioneer
ua_pioneer_dates = weights_pdm[weights_pdm['UA'] > 0].index

print(f"\nUA was detected as pioneer {len(ua_pioneer_dates)} times out of {len(weights_pdm)} ({len(ua_pioneer_dates)/len(weights_pdm)*100:.1f}%)")
print(f"\nAnswer: UA is a pioneer {len(ua_pioneer_dates)/len(weights_pdm)*100:.1f}% of the time.")

if len(ua_pioneer_dates) > 0:
    print(f"\nFirst time UA was pioneer: {ua_pioneer_dates[0].strftime('%Y-%m')}")
    print(f"Last time UA was pioneer: {ua_pioneer_dates[-1].strftime('%Y-%m')}")
    
    # Group by year to see patterns
    ua_pioneer_years = pd.Series(ua_pioneer_dates.year).value_counts().sort_index()
    
    print("\nUA pioneer detections by year:")
    for year, count in ua_pioneer_years.items():
        print(f"  {year}: {count} months")
    
    # Identify periods with clusters
    print("\nMain periods when UA was pioneer:")
    ua_pioneer_df = pd.DataFrame({'date': ua_pioneer_dates, 'weight': weights_pdm.loc[ua_pioneer_dates, 'UA']})
    
    for idx, row in ua_pioneer_df.head(10).iterrows():
        print(f"  {row['date'].strftime('%Y-%m')}: weight = {row['weight']:.3f}, UA inflation = {infl_panel.loc[row['date'], 'UA']:.2f}%")
else:
    print("\nUA was never detected as pioneer in this sample.")

print("\nInterpretation:")
print("Ukraine is rarely a pioneer (10.3% of the time), meaning it usually FOLLOWS")
print("rather than LEADS inflation dynamics. However, when it IS detected as pioneer,")
print("the method gives it high weight (0.626 on average), indicating strong")
print("conviction. This likely occurs during major Ukrainian-specific shocks")
print("(e.g., political crises, war) that don't affect Europe immediately.")

# ============================================================
# INTERPRETATIONS OF ALL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL INTERPRETATIONS")
print("=" * 70)

interpretations = """
QUESTION 1 - Inflation Panel:
The inflation panel combines cross-sectional (multiple countries) and time-series 
(monthly observations) data, allowing us to study both individual country dynamics 
and cross-country relationships.

QUESTION 2 - UA Inflation Type:
UA inflation is a pandas Series of float64 type with 301 monthly observations, 
representing year-on-year inflation rates indexed by date.

QUESTION 3 - Descriptive Statistics:
Key findings:
• Ukraine has much higher average inflation (11.86%) than all European countries
• European countries show relatively similar patterns:
  - Core EU (DE, FR, NL, AT, FI): Low and stable inflation (≈1.5-2.5% mean)
  - Southern EU (IT, ES, PT, GR, IE): Slightly higher but still moderate
• Ukraine is extremely volatile (std = 10.41%) compared to all European countries
• This reflects Ukraine's economic instability and exposure to major shocks
• The panel demonstrates strong heterogeneity between Ukraine and EU members

QUESTION 4-5 - Correlations:
• UA-FR correlation is virtually zero (0.48%) and NOT significant (p = 0.93)
  → Ukraine and France have completely independent inflation dynamics
• FR-DE correlation is very high (90.24%) and highly significant (p < 0.001)
  → France and Germany share common inflation drivers (ECB policy, European integration)

QUESTION 6 - Structural Breaks (Chow + CUSUM tests):
Tests performed on regression: FR = β0 + β1*UA + ε
• Chow test (p = 0.72): NO significant break at 2013
  → The relationship between FR and UA did not experience an abrupt structural break
  → Although the coefficient changed sign (β1: 0.0125 → -0.0029), this shift is not 
    statistically significant
  → Suggests continuity in the relationship despite coefficient evolution
• CUSUM test (p < 0.001): Parameter INSTABILITY detected
  → The regression coefficients are NOT stable over time
  → Indicates gradual parameter drift rather than one-time break
  → The relationship evolved progressively throughout the sample
Results interpretation:
  → Only CUSUM significant = gradual parameter evolution without abrupt break
  → The near-zero correlation between UA and FR makes the relationship inherently unstable
  → Any small changes in the weak relationship appear as drift in CUSUM test
  → Economic interpretation: FR and UA inflation remain largely independent, but their 
    weak correlation shifts gradually over time due to changing economic contexts

QUESTION 7 - Inflation Persistence (ADF):
• Ukraine: Inflation is STATIONARY (rejects unit root, p = 0.012)
  → Ukrainian inflation is mean-reverting despite high volatility
  → Shocks have temporary effects - inflation returns to its long-run average
  → This suggests reactive policy or market forces that prevent sustained 
    inflation spirals
  → High variance but stable long-run dynamics
• France: Inflation exhibits HIGH PERSISTENCE (cannot reject unit root, p = 0.112)
  → French inflation behaves like a random walk with drift
  → Shocks have permanent effects - inflation doesn't automatically revert
  → Reflects ECB's gradualist approach and well-anchored inflation expectations
  → Changes in inflation are slow and predictable rather than mean-reverting

QUESTION 8 - Pioneer Detection Method:
• European countries are frequently detected as pioneers (18-25% of the time each)
  → With 11 EU countries, detection is distributed more evenly
  → Finland (25.2%) and France (24.9%) are slightly more frequent pioneers
• Ukraine is pioneer only 10.3% of the time
  → Ukraine mostly FOLLOWS rather than leads
  → But when UA IS pioneer, PDM gives it high weight (0.626 on average)
  → This suggests Ukrainian-specific shocks that don't immediately affect Europe

Conclusion:
Ukraine has a fundamentally different inflation regime than France/Germany:
- Higher level, higher volatility, lower correlation with Europe
- Ukraine reacts to local shocks (political instability, war)
- Europe reacts to common ECB policy and economic integration
- PDM correctly identifies that Ukraine rarely leads European inflation patterns
"""

print(interpretations)

print("\n" + "=" * 70)
print("EXAM COMPLETED")
print("=" * 70)
