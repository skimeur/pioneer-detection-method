# QMF Exam — Counterfactual Ukrainian Inflation Analysis

**Author:** Andrey Zalizniak
**Course:** Quantitative Methods in Finance (M2 Finance Technology Data, 2025-2026)
**Exam:** Take-Home Final

## Question

*What would Ukrainian inflation have looked like if Ukraine had been part of the Euro Area?*

The analysis is in two parts:

- **Part A** — Documented chronology of the NBU's exchange-rate regime from 2000 to 2025: de facto pegs, devaluation episodes, capital controls, and the 2015-2016 transition to inflation targeting. Establishes that monetary sovereignty was genuine only during 2017-2021; for the rest of the sample the policy rate was subordinated to exchange-rate defence (Calvo & Reinhart, 2002).
- **Part B** — Constructs the counterfactual using two complementary methods:
  1. **SVAR with Blanchard-Quah (1989) identification** — bivariate `[IP_growth, inflation]` VARs for Ukraine and the Euro Area; the counterfactual keeps Ukraine's supply shocks but replaces its demand shocks with EA demand shocks (Bayoumi & Eichengreen, 1993).
  2. **Inflation-spread model (Honohan & Lane, 2003)** — regresses `π_UA − π_EA` on UAH/USD depreciation, Brent oil, and a regime dummy. The counterfactual zeros the exchange-rate channel and the peg-credibility dummy while keeping the structural premium and oil pass-through.

## Repository layout

```
QMF_Exam_Contrefactual_Ukr_infl_AZ/
├── PartA_Andrey.py             # Part A: regime chronology + figure
├── PartB_Andrey.py             # Part B: SVAR + spread model + counterfactual
├── data_contrefactual/         # 7 input files (see below)
├── contrefactual_outputs/      # Generated figures and tables
├── README.md
├── requirements.txt
└── .gitignore
```

## Input data (`data_contrefactual/`)

| File | Used by | Source |
|------|---------|--------|
| `uah_usd_monthly_2000_2026.csv` | A + B | IMF IFS / NBU / Investing.com (compiled) |
| `nbu_policy_rate_monthly.csv` | A | NBU Open Data Portal |
| `data_ecb_hicp_panel.csv` | B | ECB Data Portal (HICP, y-o-y %) |
| `data_ukraine_cpi_raw.csv` | B | SSSU SDMX (CPI, m-o-m index = 100) |
| `ukraine_ipi_monthly_2003_2026.csv` | B | SSSU / NBU (industrial production, y-o-y %) |
| `sts_inpr_m__custom_21393318_linear_2_0.csv` | B | Eurostat (EA19 industrial production) |
| `Europe_brent_spot_price_FOB.xls` | B | EIA / FRED (Brent spot, $/barrel) |

## Outputs (`contrefactual_outputs/`)

| File | Produced by |
|------|-------------|
| `fig_partA_regime_chronology.png` | Part A — UAH/USD with regime shading + NBU key rate |
| `table_partA_regime_chronology.csv` | Part A — 10-period regime table |
| `fig_partB_counterfactual.png` | Part B — actual UA inflation vs. SVAR and spread-model counterfactuals |

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Part A (reads data_contrefactual/, writes contrefactual_outputs/)
python PartA_Andrey.py

# 3. Run Part B
python PartB_Andrey.py
```

Both scripts use the headless matplotlib `Agg` backend, so they run without a display. Set the working directory to this folder before running, since paths are relative.

## Key findings

- **Part A:** Ukraine had genuine monetary sovereignty only during 2017-2021. The "treatment intensity" of Euro Area membership is therefore time-varying and concentrated in the crisis and post-2016 episodes.
- **Part B — SVAR:** Sample 2003-01 → 2024-08 (T = 260). UA VAR p = 2, EA VAR p = 6. The Blanchard-Quah counterfactual identifies large positive gaps during the 2008-09 and 2014-15 devaluations.
- **Part B — Spread model:** R² = 0.43. UAH/USD depreciation coefficient γ₂ = 0.312 (p < 0.0001) is the dominant pass-through channel. Structural premium (intercept) α₀ ≈ 6.7 pp captures Balassa-Samuelson and other persistent factors. Mean counterfactual inflation ≈ 9.2% vs. actual ≈ 11.8%.
- **Sanity check (Giavazzi-Pagano credibility hypothesis):** Pre-2017 mean treatment gap is larger than post-2017 — consistent with the prediction that credibility gains from EA membership shrink once the central bank adopts its own credible nominal anchor.

## References

Blanchard & Quah (1989, AER); Bayoumi & Eichengreen (1993, Cambridge UP); Honohan & Lane (2003, Economic Policy); Balassa (1964, JPE); Égert (2007, CESifo WP 2127); Mundell (1961, AER); De Grauwe (2012); Giavazzi & Pagano (1988, EER); Calvo & Reinhart (2002, QJE); Barro & Gordon (1983, JME); Frankel & Rose (1998, EJ); Sims (1980, Econometrica).
