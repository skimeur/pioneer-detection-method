#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Counterfactual Inflation Analysis: What If Ukraine Had Been Part of the Euro Area?
==================================================================================

Author  : Andrey Zalizniak
Course  : Quantitative Methods in Finance (M2 Finance Technology Data, 2025-2026)
Exam    : Take-Home Final

Part A — Ukraine's Monetary Regime: A Preliminary Analysis
----------------------------------------------------------
Constructs a documented chronology of the NBU's exchange-rate regime from
2000 to 2025, identifies de facto pegs, devaluation episodes, capital
controls, and the transition to inflation targeting.

Data sources
------------
UAH/USD exchange rate:
    Monthly average UAH/USD, Jan 2000 - Jan 2026, compiled from:
    - IMF IFS / NBU official rate (2000-2007, via API)
    - Investing.com / NBU (2008-2025, cross-checked with NBU xlsx)
    File: uah_usd_monthly_2000_2026.csv (provided alongside submission)

NBU policy rate:
    National Bank of Ukraine, monthly discount/key policy rate.
    Downloaded from NBU Open Data Portal on 2026-05-10.
    https://bank.gov.ua/en/statistic/sector-financial/data-sector-financial
    Note: raw values are in units of 10 percentage points (e.g., 2.5 = 25%).

Academic and institutional sources
----------------------------------
IMF AREAER         : Annual Report on Exchange Arrangements and Exchange
                     Restrictions (2000-2024 editions)
Calvo & Reinhart   : "Fear of Floating" (QJE, 2002)
NBU                : "Monetary Policy Strategy of the NBU" (2015, 2018)
Barro & Gordon     : "Rules, Discretion and Reputation" (JME, 1983)
Giavazzi & Pagano  : "The Advantage of Tying One's Hands" (EER, 1988)
Frankel & Rose     : "The Endogeneity of the OCA Criteria" (EJ, 1998)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DATA_DIR = "data_contrefactual/"    # adjust to your local path if needed
OUT_DIR  = "contrefactual_outputs/"
os.makedirs(OUT_DIR, exist_ok=True)

# Check that required external data files are present
for fname in ["uah_usd_monthly_2000_2026.csv", "nbu_policy_rate_monthly.csv"]:
    if not os.path.exists(DATA_DIR + fname):
        raise FileNotFoundError(
            f"Missing: {DATA_DIR + fname}. See header for download instructions."
        )

# -------------------------------------------------------------------------
# 1. Load and prepare UAH/USD exchange rate
# -------------------------------------------------------------------------

fx = pd.read_csv(DATA_DIR + "uah_usd_monthly_2000_2026.csv",
                 sep=";", decimal=",")
fx["date"] = pd.to_datetime(fx["date"], dayfirst=True)
fx = fx.set_index("date")[["uah_usd"]].sort_index()
fx.columns = ["UAH_USD"]

# Data cleaning: xlsx source for Aug 2018 - Nov 2019 contains values ~100x
# too large (e.g., 2824 instead of 28.24 — likely a kopecks/hryvnia unit
# error). Correct by dividing values > 100 by 100.
fx.loc[fx["UAH_USD"] > 100, "UAH_USD"] = fx["UAH_USD"] / 100.0

# Fill the Jan-Jul 2018 gap (missing months between api and xlsx sources)
fx = fx.resample("ME").mean().interpolate(method="linear")

# -------------------------------------------------------------------------
# 2. Load NBU policy rate
# -------------------------------------------------------------------------

nbu = pd.read_csv(DATA_DIR + "nbu_policy_rate_monthly.csv")
nbu["date"] = pd.to_datetime(nbu["date"])
# Raw data is scaled by 1/10 (e.g., 2.5 in CSV = 25% actual rate).
# Verification: NBU raised key rate from 10% to 25% in Jun 2022;
# CSV shows 1.0 → 2.5 over that period. Discount rate was 45% in
# Jan 2000; CSV shows 4.5. Scaling confirmed: multiply by 10.
nbu["nbu_rate_pct"] = nbu["nbu_rate"] * 10.0
nbu = nbu.set_index("date")[["nbu_rate_pct"]]

# -------------------------------------------------------------------------
# 3. Regime chronology table
# -------------------------------------------------------------------------
# Constructed from: IMF AREAER (various years), Calvo & Reinhart (2002),
# NBU publications, and direct inspection of the UAH/USD series.

regime_table = pd.DataFrame([
    {
        "Period": "2000-2004",
        "UAH/USD (approx.)": "5.3 - 5.4",
        "De facto regime": "Conventional peg to USD",
        "IMF classification": "Stabilised arrangement / Other conventional peg",
        "Capital controls": "Extensive (surrender requirements, transfer restrictions)",
        "Key events": (
            "Post-hyperinflation stabilisation. NBU maintains de facto "
            "dollar peg via heavy intervention. Discount rate cut from 45% "
            "to 7%. No independent monetary policy."
        ),
    },
    {
        "Period": "2005-2007",
        "UAH/USD (approx.)": "5.05",
        "De facto regime": "Conventional peg to USD",
        "IMF classification": "Stabilised arrangement (de facto peg at 5.05)",
        "Capital controls": "Moderate (gradually eased for current account)",
        "Key events": (
            "Strong growth (7-8% real GDP), large capital inflows, "
            "appreciation pressure. NBU intervenes to prevent appreciation. "
            "De facto peg stable at 5.05. IMF classifies as stabilised "
            "arrangement despite official 'managed float' label."
        ),
    },
    {
        "Period": "2008 Q4 - 2009 Q1",
        "UAH/USD (approx.)": "5.05 → 8.0",
        "De facto regime": "DEVALUATION (crisis-driven)",
        "IMF classification": "Managed float (during adjustment)",
        "Capital controls": "Tightened (NBU imposed emergency FX restrictions)",
        "Key events": (
            "Global Financial Crisis. Capital flight, current account "
            "reversal, steel export collapse. UAH devalues ~60% over "
            "6 months. IMF Stand-By Arrangement (Nov 2008, $16.4bn). "
            "Trigger: global liquidity crisis + commodity price collapse."
        ),
    },
    {
        "Period": "2009-2013",
        "UAH/USD (approx.)": "7.9 - 8.2",
        "De facto regime": "De facto peg to USD (new level)",
        "IMF classification": "Stabilised arrangement",
        "Capital controls": "Moderate to tight (NBU FX market regulations)",
        "Key events": (
            "NBU re-establishes peg at ~8 UAH/USD. Inflation falls to "
            "single digits. But: twin deficits persist, reserves erode "
            "gradually. No genuine monetary policy autonomy; interest "
            "rate subordinated to exchange-rate defence."
        ),
    },
    {
        "Period": "2014 Feb - 2015 Feb",
        "UAH/USD (approx.)": "8.2 → 24 - 30",
        "De facto regime": "DEVALUATION (geopolitical crisis)",
        "IMF classification": "Managed float → Free fall (Reinhart-Rogoff)",
        "Capital controls": "Severe (mandatory FX surrender, transfer caps, "
                            "cash withdrawal limits)",
        "Key events": (
            "Euromaidan revolution (Feb 2014), Russian annexation of Crimea "
            "(Mar 2014), armed conflict in Donbas. UAH devalues >200%. "
            "NBU loses ~60% of reserves. IMF programme ($17.1bn, Mar 2015). "
            "Trigger: geopolitical shock + loss of industrial base in east."
        ),
    },
    {
        "Period": "2015-2016",
        "UAH/USD (approx.)": "24 - 27",
        "De facto regime": "Managed float (transition)",
        "IMF classification": "Managed float (composite)",
        "Capital controls": "Extensive (gradually relaxed)",
        "Key events": (
            "NBU formally adopts inflation-targeting framework "
            "(Board Decision, Aug 2015; operational from 2016). "
            "IT target: 12% for 2016, 8% for 2017, 6% for 2018, "
            "5% +/- 1pp from 2019. In practice, the NBU began actively "
            "using the key policy rate as the primary instrument, allowed "
            "greater exchange-rate flexibility, and reduced direct FX "
            "interventions — a genuine break from the prior peg-driven "
            "regime (Giavazzi & Pagano, 1988). Exchange rate stabilises "
            "but remains volatile. Actual inflation ~43% in 2015, ~14% "
            "in 2016."
        ),
    },
    {
        "Period": "2017-2019",
        "UAH/USD (approx.)": "26 - 28 (then 24 by end-2019)",
        "De facto regime": "Managed float / Inflation targeting",
        "IMF classification": "Floating",
        "Capital controls": "Gradually lifted (current account liberalised 2017-2019)",
        "Key events": (
            "IT becomes operational. Inflation converges to target "
            "(14% → 10% → 4%). NBU gains credibility, reserves rebuilt. "
            "This is the only period of genuine monetary sovereignty. "
            "UAH appreciates in 2019 (capital inflows + high real rates)."
        ),
    },
    {
        "Period": "2020-2021",
        "UAH/USD (approx.)": "27 - 28",
        "De facto regime": "Inflation targeting (flexible)",
        "IMF classification": "Floating",
        "Capital controls": "Minimal (COVID-era temporary measures)",
        "Key events": (
            "COVID-19 pandemic. NBU cuts key rate from 13.5% to 6% "
            "(aggressive easing). Inflation stays near target (~5%). "
            "Rate cycle similar to ECB/Fed. Genuine monetary autonomy "
            "continues. IMF Stand-By ($5bn, Jun 2020)."
        ),
    },
    {
        "Period": "2022 Feb - 2022 Jul",
        "UAH/USD (approx.)": "29.3 → 36.6 (fixed Jul 21)",
        "De facto regime": "DEVALUATION → wartime peg",
        "IMF classification": "Stabilised arrangement (wartime)",
        "Capital controls": "Severe (full FX surrender, capital outflow ban, "
                            "cash withdrawal limits)",
        "Key events": (
            "Full-scale Russian invasion (24 Feb 2022). NBU fixes rate "
            "at 29.25 (day 1), then devalues to 36.57 (21 Jul 2022). "
            "Key rate raised from 10% to 25%. Monetary sovereignty "
            "suspended; war economy. Trigger: full-scale war."
        ),
    },
    {
        "Period": "2022 Oct - 2025",
        "UAH/USD (approx.)": "36.6 → 41.2 (controlled crawl)",
        "De facto regime": "Managed arrangement / Controlled depreciation",
        "IMF classification": "Stabilised arrangement → Crawl-like",
        "Capital controls": "Gradually eased (current account 2023, "
                            "capital account partial 2024)",
        "Key events": (
            "NBU begins gradual easing: key rate cut from 25% to 13% "
            "(2023-2024). Controlled UAH depreciation (~3%/year). "
            "NBU signals return to IT framework post-war. IMF EFF "
            "($15.6bn, Mar 2023). De facto regime: crawl-like "
            "arrangement with heavy intervention."
        ),
    },
])


# -------------------------------------------------------------------------
# 4. Print the summary table
# -------------------------------------------------------------------------

print("Part A — NBU Exchange Rate Regime Chronology (2000-2025)\n")
for _, row in regime_table.iterrows():
    print(f"Period: {row['Period']}")
    print(f"  UAH/USD: {row['UAH/USD (approx.)']}")
    print(f"  De facto regime: {row['De facto regime']}")
    print(f"  IMF classification: {row['IMF classification']}")
    print(f"  Capital controls: {row['Capital controls']}")
    print(f"  Key events: {row['Key events']}")
    print()


# -------------------------------------------------------------------------
# 5. Written argument: monetary sovereignty assessment
# -------------------------------------------------------------------------

ARGUMENT = """
Part A — Monetary sovereignty assessment
-----------------------------------------
Ukraine had genuine monetary sovereignty only during 2017-2021, when the
NBU operated a credible inflation-targeting framework with a floating
exchange rate, actively used the policy rate as the primary instrument,
and progressively lifted capital controls. During 2000-2008 and 2009-2013,
the de facto dollar peg meant that the NBU's interest rate was subordinated
to exchange-rate defence; monetary policy was effectively imported from the
Federal Reserve — a "fear of floating" regime in the sense of Calvo and
Reinhart (2002). During the three devaluation episodes (2008-09, 2014-15,
2022), the NBU lost control of both the exchange rate and inflation
simultaneously, operating under crisis management rather than a policy
framework. During the wartime period (2022-2025), monetary sovereignty was
again suspended via a fixed rate and capital controls.

Ukraine had genuine monetary sovereignty only during 2017-2021, when the
NBU operated a credible inflation-targeting framework with a floating
exchange rate, actively used the policy rate as the primary instrument,
and progressively lifted capital controls. During 2000-2008 and 2009-2013,
the de facto dollar peg meant that the NBU's interest rate was subordinated
to exchange-rate defence; monetary policy was effectively imported from the
Federal Reserve — a "fear of floating" regime in the sense of Calvo and
Reinhart (2002). Crucially, the NBU officially declared a "managed float"
throughout 2000-2014 (de jure classification), while the IMF's AREAER
consistently classified the arrangement as a "conventional peg" or
"stabilised arrangement" (de facto classification) — a textbook divergence
between declared and actual regimes. During the three devaluation episodes
(2008-09, 2014-15, 2022), the NBU lost control of both the exchange rate
and inflation simultaneously, operating under crisis management rather
than a policy framework. During the wartime period (2022-2025), monetary
sovereignty was again suspended via a fixed rate and capital controls.

This has a direct implication for the counterfactual: Euro Area membership
would have represented a large regime change only during the brief 2017-2021
window of genuine monetary autonomy and during the devaluation crises (when
the exchange-rate buffer would have been unavailable). During the peg periods
(2000-2008, 2009-2013), Ukraine was already operating under a de facto
fixed exchange rate — joining the Euro Area would have substituted one
external anchor (USD) for another (EUR), with the additional benefit of
imported ECB credibility (Giavazzi & Pagano, 1988) but without a radical
change in monetary regime. The "treatment intensity" of Euro Area membership
is therefore time-varying and concentrated in the crisis and post-2016
episodes.k

This has a direct implication for the counterfactual: Euro Area membership
would have represented a large regime change only during the brief 2017-2021
window of genuine monetary autonomy and during the devaluation crises (when
the exchange-rate buffer would have been unavailable). During the peg periods
(2000-2008, 2009-2013), Ukraine was already operating under a de facto
fixed exchange rate — joining the Euro Area would have substituted one
external anchor (USD) for another (EUR), with the additional benefit of
imported ECB credibility but without a radical change in monetary regime.
The "treatment intensity" of Euro Area membership is therefore time-varying
and concentrated in the crisis and post-2016 episodes.

 Sources: IMF AREAER (2001-2024); IMF Country Reports 08/384, 15/69,
20/197, 23/93; NBU Board Decision No. 541 (2015); Calvo & Reinhart (2002);
Giavazzi & Pagano (1988); Barro & Gordon (1983); Frankel & Rose (1998).
"""

print(ARGUMENT)

SOURCES = """
Part A — Sources
-----------------
[1] IMF, Annual Report on Exchange Arrangements and Exchange Restrictions
    (AREAER), editions 2001-2024. De facto regime classifications for
    Ukraine: "conventional peg" (AREAER 2001-2007), "stabilised
    arrangement" (AREAER 2009, 2010, 2011, 2012, 2013), "floating"
    (AREAER 2018, 2019, 2020, 2021), "stabilised arrangement"
    (AREAER 2023). Available: https://www.elibrary.imf.org/
[2] IMF Country Report No. 08/384 (Nov 2008). "Ukraine: Request for
    Stand-By Arrangement." $16.4bn SBA.
[3] IMF Country Report No. 15/69 (Mar 2015). "Ukraine: Request for
    Extended Fund Facility." $17.1bn EFF.
[4] IMF Country Report No. 20/197 (Jun 2020). "Ukraine: Request for
    Stand-By Arrangement." $5bn SBA.
[5] IMF Country Report No. 23/93 (Mar 2023). "Ukraine: Request for
    Extended Arrangement under the EFF." $15.6bn EFF.
[6] National Bank of Ukraine, Board Decision No. 541 (18 August 2015).
    "On approval of the Monetary Policy Strategy of the NBU."
    Target path: 12% (2016), 8% (2017), 6% (2018), 5% +/- 1pp (2019+).
[7] National Bank of Ukraine, Inflation Report (quarterly, 2016-2025).
    Used for verification of IT implementation and rate decisions.
[8] Calvo, G. A. and Reinhart, C. M. (2002). "Fear of Floating."
    Quarterly Journal of Economics, 117(2), 379-408.
[9] Giavazzi, F. and Pagano, M. (1988). "The Advantage of Tying One's
    Hands." European Economic Review, 32(5), 1055-1075.
"""
print(SOURCES)

# -------------------------------------------------------------------------
# 6. Figure: UAH/USD exchange rate with regime annotations
# -------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]})

# -- Panel 1: Exchange rate --
ax1.plot(fx.index, fx["UAH_USD"], color="black", linewidth=1)
ax1.set_ylabel("UAH per USD")
ax1.set_title("Part A — UAH/USD exchange rate and NBU monetary regime (2000-2025)")
ax1.set_xlim(fx.index.min(), fx.index.max())

# Shade peg periods
peg_periods = [
    ("2000-01", "2008-09", "#CCE5FF", "Peg ~5.05"),
    ("2009-06", "2013-12", "#CCE5FF", "Peg ~8.0"),
    ("2022-07", "2025-12", "#FFEEDD", "Wartime peg/crawl"),
]
for start, end, color, label in peg_periods:
    ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                alpha=0.3, color=color, label=label)

# Mark devaluation episodes
deval_periods = [
    ("2008-09", "2009-03", "#FFCCCC", "Deval. 2008-09"),
    ("2014-02", "2015-03", "#FFCCCC", "Deval. 2014-15"),
    ("2022-02", "2022-07", "#FFCCCC", "Deval. 2022"),
]
for start, end, color, label in deval_periods:
    ax1.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                alpha=0.4, color=color, label=label)

# Mark IT adoption
ax1.axvline(pd.Timestamp("2016-01-01"), color="green", linestyle="--",
            linewidth=1.2, label="IT adoption (2016)")

ax1.legend(fontsize=7, loc="upper left", ncol=2)
ax1.grid(alpha=0.2)

# -- Panel 2: NBU policy rate --
nbu_plot = nbu.loc[fx.index.min():fx.index.max()]
ax2.plot(nbu_plot.index, nbu_plot["nbu_rate_pct"], color="darkred", linewidth=1)
ax2.set_ylabel("NBU key rate (%)")
ax2.set_xlabel("Time")
ax2.axvline(pd.Timestamp("2016-01-01"), color="green", linestyle="--",
            linewidth=1.2)
ax2.set_xlim(fx.index.min(), fx.index.max())
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(OUT_DIR + "fig_partA_regime_chronology.png", dpi=150)
plt.show()
print("Saved: " + OUT_DIR + "fig_partA_regime_chronology.png")


# -------------------------------------------------------------------------
# 7. Export regime table to CSV (for reference / inclusion in report)
# -------------------------------------------------------------------------

regime_table.to_csv(OUT_DIR + "table_partA_regime_chronology.csv", index=False)
print("Saved: " + OUT_DIR + "table_partA_regime_chronology.csv")