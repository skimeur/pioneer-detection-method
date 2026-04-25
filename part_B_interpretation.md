# Part B — Deliverable 2: Interpretation

## What does the counterfactual imply about the cost or benefit of monetary sovereignty for Ukraine, particularly during the 2008–09, 2014–15, and 2022 crises?

### Summary paragraph (exam deliverable)

The counterfactual suggests that Euro Area membership would have kept Ukrainian inflation substantially lower over 2001–2025, but would have removed the exchange-rate channel that did most of the adjustment work in all three crises. I estimate this using a trivariate SVAR ([π_EA, Δlog(e), π_UKR], Cholesky identification) with time-varying treatment intensity w(t) drawn from the Part A chronology, to account for the fact that Ukraine was already dollar-pegged for most of the sample. The treatment gap is largest when devaluations actually occurred: roughly +11 pp during the GFC (2008–09), +14 pp during Maidan/Crimea (2014–15), and around +5 pp in 2022. In each case the hryvnia depreciation passed through directly into consumer prices — something that simply would not have happened under the euro. During the peg periods (2001–2008 and 2009–2014) the gap is close to zero (−0.3 pp on average), which is what you would expect: Ukraine was already running a de facto dollar anchor, so replacing it with the ECB would not have changed much. I tested alternative w(t) calibrations (w_peg ∈ {0.10, 0.25, 0.35}, w_IT ∈ {0.60, 0.70, 0.85}) and the qualitative picture holds across all of them. The harder question is whether lower inflation would have been worth it. Without the exchange rate, every asymmetric shock — war, energy prices, geopolitics — would have had to be absorbed through wages and output. Ukraine's GDP fell −15%, −10%, and −29% in 2009, 2015, and 2022 respectively; those contractions could plausibly have been deeper under internal devaluation. As De Grauwe (2012) notes, euro members are also more exposed to self-fulfilling sovereign debt crises since they cannot use their own central bank as a lender of last resort. On balance, monetary sovereignty has been expensive in inflation terms but genuinely useful as a buffer, and Ukraine's shock profile (Bayoumi and Eichengreen 1993) puts it far from the EA core in terms of OCA suitability.

---

### Detailed analysis (extended notes)

The four counterfactual methods (primary SVAR, Ciccarelli-Mojon, Blanchard-Quah, and the Part A–consistent blend) bracket the plausible range of Ukraine's hypothetical Euro Area inflation path and jointly illuminate the underlying economic mechanisms. All methods and their comparison are shown in `figures/ukraine_counterfactual_robustness.png`; the exam deliverable (two-series figure) is `figures/ukraine_counterfactual_inflation.png`.

The **bivariate VAR with Cholesky identification** (core method) eliminates Ukraine's idiosyncratic structural shock — the monetary and exchange-rate component of inflation innovations — while preserving the autoregressive dynamics estimated on the actual data. It delivers a counterfactual with a full-sample mean of approximately **11–12%**, substantially above the EA average (~2%) but well below Ukraine's actual peaks (60.9% in 2015, ~26% in 2022–23). The elevated CF level reflects Ukraine's structural inflation persistence: even under EA membership, the deep-rooted inflationary dynamics inherited from decades of weak monetary institutions would not disappear overnight. They would decline gradually as ECB credibility is imported — the Giavazzi–Pagano (1988) "tying one's hands" mechanism — but the VAR's linear propagation does not model this credibility accumulation.

The **Ciccarelli–Mojon factor model** (robustness check) anchors the counterfactual at the ECB's long-run mean (~2.18%), which corresponds to the scenario in which Euro Area membership achieves *complete* credibility import from day one. It constitutes a lower bound: the best-case outcome in which all idiosyncratic inflationary dynamics are eliminated and Ukraine absorbs only the common EA monetary cycle, with a loading λ̂ = 0.47 < 1 (implying lower cyclical volatility than the EA average).

**The gap between the two methods is itself a measure of the credibility channel** (Barro–Gordon 1983): the difference between ~11% (VAR) and ~2% (factor) represents the inflation that would persist under EA membership due to structural persistence, until ECB credibility fully permeates expectations. This gradual convergence is exactly what was observed empirically in the Baltic states after 2011 and in Slovakia after 2009 — initially elevated inflation, converging to EA levels over 3–7 years.

Regarding the three crisis episodes:

- **2008–09 (GFC):** The VAR treatment gap is +11 pp; the factor gap is +20 pp. Both methods agree that the hryvnia devaluation of 2008–09 — which passed through directly to domestic prices — would have been entirely absent under EA membership. The real adjustment would have required internal devaluation (wage cuts, output contraction), as Ireland and the Baltic states experienced. Ukraine would have avoided the inflation spike but at the cost of a deeper recession.

- **2014–15 (Maidan/Crimea):** The largest treatment gap in both methods (+14 pp VAR, +25 pp factor). The hryvnia lost over 200% of its value; this was the most catastrophic episode of exchange-rate pass-through in modern Ukrainian history. Under EA membership, the exchange-rate buffer would have been unavailable. Ukraine would have faced a sovereign debt crisis rather than a currency crisis — consistent with De Grauwe's (2012) argument that monetary union members are more vulnerable to self-fulfilling debt crises — since no national central bank could have provided emergency liquidity in hryvnia. Whether this would have been better or worse than the observed currency collapse is genuinely ambiguous and depends on the availability of ESM-style bailout mechanisms.

- **2022 (full-scale invasion):** The VAR gap is +5 pp; the factor gap is +11 pp. The 2022 episode is different from the previous two: Ukraine managed the exchange rate largely through a fixed peg (29.25 UAH/USD, later 36.57) backed by IMF and Western official financing. EA membership would have eliminated the July 2022 devaluation, but the wartime fiscal expansion and supply disruptions would have still generated significant inflationary pressure — pressure that would have had to be absorbed through domestic price and wage adjustment rather than currency depreciation.

**The overall picture** is that monetary sovereignty has been a costly but genuine adjustment tool for Ukraine. The exchange rate served as a shock absorber in all three episodes, at the price of sustained above-EA inflation throughout the sample. Euro Area membership would have eliminated this buffer, delivering lower average inflation (VAR: –10 pp relative to actual on average; factor model: –20 pp) but exposing Ukraine to the risk of deeper quantity adjustments — output losses, unemployment, and potentially self-fulfilling sovereign debt crises — in response to the same asymmetric shocks. Given that Ukraine's shocks (geopolitical, energy-dependence, war) are among the most asymmetric in Europe relative to the EA core, the OCA calculus — as formalized by Bayoumi and Eichengreen (1993) — would likely have concluded that Ukraine was not, and is not yet, a suitable candidate for Euro Area membership in terms of shock symmetry alone.

---

## References

**Econometric methods**

- Barro, R.J. and Gordon, D.B. (1983). Rules, discretion and reputation in a model of monetary policy. *Journal of Monetary Economics*, 12(1):101–121.
- Bayoumi, T. and Eichengreen, B. (1993). Shocking aspects of European monetary integration. In Torres, F. and Giavazzi, F. (eds.), *Adjustment and Growth in the European Monetary Union*, pp. 193–229. Cambridge University Press.
- Blanchard, O.J. and Quah, D. (1989). The dynamic effects of aggregate demand and supply disturbances. *American Economic Review*, 79(4):655–673.
- Ciccarelli, M. and Mojon, B. (2010). Global inflation. *Review of Economics and Statistics*, 92(3):524–535.
- Kilian, L. and Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*. Cambridge University Press.
- MacKinnon, J.G. (1994). Approximate asymptotic distribution functions for unit-root and cointegration tests. *Journal of Business & Economic Statistics*, 12(2):167–176.
- Sims, C.A. (1980). Macroeconomics and reality. *Econometrica*, 48(1):1–48.

**Monetary regimes and OCA theory**

- Calvo, G.A. and Reinhart, C.M. (2002). Fear of floating. *Quarterly Journal of Economics*, 117(2):379–408.
- De Grauwe, P. (2012). The governance of a fragile eurozone. *Australian Economic Review*, 45(3):255–268.
- Frankel, J.A. and Rose, A.K. (1998). The endogeneity of the optimum currency area criteria. *Economic Journal*, 108(449):1009–1025.
- Giavazzi, F. and Pagano, M. (1988). The advantage of tying one's hands: EMS discipline and central bank credibility. *European Economic Review*, 32(5):1055–1075.
- Mundell, R.A. (1961). A theory of optimum currency areas. *American Economic Review*, 51(4):657–665.
- Reinhart, C.M. and Rogoff, K.S. (2004). The modern history of exchange rate arrangements: A reinterpretation. *Quarterly Journal of Economics*, 119(1):1–48.

**Institutional and data sources**

- IMF (2022, 2023, 2024). *Annual Report on Exchange Arrangements and Exchange Restrictions (AREAER)*. Washington, D.C.: International Monetary Fund.
- IMF Country Report No. 08/384: *Ukraine — Request for Stand-By Arrangement* (2008).
- IMF Country Report No. 15/69: *Ukraine — Request for Extended Arrangement under the EFF* (2015).
- IMF Country Report No. 23/146: *Ukraine — Request for Extended Arrangement under the EFF* (2023).
- National Bank of Ukraine (2015). *Monetary Policy Guidelines for 2016–2020*. Kyiv: NBU.
- National Bank of Ukraine (2022). *Monetary Policy Report*, Q2 2022. Kyiv: NBU.
- National Bank of Ukraine (2023). *Monetary Policy Report*, Q4 2023. Kyiv: NBU.
- State Statistics Service of Ukraine (SSSU). Monthly CPI index, via SDMX. Available at ukrstat.gov.ua.
- ECB Data Portal. Monthly HICP, year-on-year (%), 11 Euro Area countries, 2000–2025.
- World Bank Open Data. Ukraine real GDP growth (NY.GDP.MKTP.KD.ZG). Available at data.worldbank.org.