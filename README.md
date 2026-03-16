# Pioneer Detection Method (PDM)
*A convergence-based expert-aggregation algorithm for structural change*

This repository contains the official Python implementation of the **Pioneer Detection Method (PDM)**, the algorithm introduced in:

**Eric Vansteenberghe (2025)**  
*Insurance Supervision under Climate Change: A Pioneer Detection Method.*  
*The Geneva Papers on Risk and Insurance – Issues and Practice.*  
https://doi.org/10.1057/s41288-025-00367-y

Open access versions:
https://dx.doi.org/10.2139/ssrn.5012810


The repository includes:
- `pdm.py` – implementation of the methods
- `pdm_demo.py` – complete synthetic example with plots  
- `ecb_hicp_panel_var_granger.py` – inflation panel construction with ADF, Granger, and VAR
- `ecb_hicp_panel_pioneer_extension.py` – Pioneer Detection extension for the inflation panel
- `paper.pdf` – full published article  
- `slides.pdf` – presentation slides summarizing the method  

---

## 1. What is the Pioneer Detection Method (PDM)?

The Pioneer Detection Method is an expert‑aggregation algorithm designed for environments characterized by:

- structural change  
- heterogeneous learning speeds  
- fat‑tailed or non‑Gaussian risks  
- fragmented information  
- unobservable true parameters  

Instead of pooling experts by performance (which is impossible when the true parameter is unknown), PDM detects **directional convergence**:

> A “pioneer” is an expert whose estimate moves first in the correct direction, and toward whom other experts converge over time.

PDM uses three convergence criteria:

- **Distance reduction** – others move closer to the candidate pioneer  
- **Orientation** – others move toward the pioneer  
- **Attribution proportion** – how much of the movement comes from peers  

The algorithm produces dynamic weights that sum to 1 whenever at least one pioneer exists; otherwise it defaults to the cross‑sectional mean.

---

## 2. Applications

### 2.1 Insurance Supervision & Climate Risk

Based on yearly aggregate loss data, PDM helps supervisors assess tail‑risk dynamics under climate change. Use cases include:

- tail‑parameter estimation under Pareto‑type risks  
- monitoring insurability after climate shocks  
- pooling fragmented expertise across insurers  
- mitigating uncertainty when reinsurance capacity withdraws  

---

### 2.2 Time‑Series Forecasting Under Regime Shifts

PDM improves robustness in:

- macroeconomic forecasting under structural breaks  
- climate‑sensitive time series  
- low signal‑to‑noise environments  
- model uncertainty and forecast combination  

It is suited to settings where the “truth” is never directly observed.

---

### 2.3 Multi‑Agent Systems & Robotics

PDM extends naturally to distributed‑sensing systems:

- drone swarms  
- robotic fleets  
- underwater autonomous vehicles  
- coordinated automated‑vehicle systems  
- sensor networks  
- decentralized AI agents  

In these systems, PDM can:

- identify early detectors of environmental changes  
- neutralize agents whose signals push the system in unsafe directions (algorithmically by down‑weighting)  
- produce stable swarm‑level situational awareness  
- enhance collective adaptation under partial detection  

---

## 3. Contents of the Repository

```
pdm.py               # Minimal PDM implementation
pdm_demo.py          # Synthetic demo with plotting
paper.pdf            # Published article (full text)
slides.pdf           # Slide deck for presentations
extended-abstract.md
```

---

## 4. Quick Start

### Install dependencies

```bash
pip install pandas numpy matplotlib statsmodels
```

`statsmodels` is required for Granger Causality and Multivariate Regression methods. The PDM variants, Lagged Correlation, Transfer Entropy, and traditional benchmarks only need `pandas` and `numpy`.

For the inflation-panel teaching workflow, the scripts also use `requests`, and
`openpyxl` is recommended so pandas can read the official NBU Excel fallback
when the SSSU API is temporarily unavailable.

Run the Pioneer Detection inflation extension with:

```bash
MPLBACKEND=Agg python ecb_hicp_panel_pioneer_extension.py
```

The extension is reproducible as written: it uses a fixed sample window and a
single string switch (`TARGET_PROFILE`) to alternate between the France and
Ukraine versions of the exercise.

### Using the Pioneer Detection Method

```python
import pandas as pd
from pdm import compute_pioneer_weights_simple, pooled_forecast_simple

# forecasts: DataFrame (T x N) of expert forecasts or estimates
weights = compute_pioneer_weights_simple(forecasts)
pooled  = pooled_forecast_simple(forecasts, weights)

print(weights)
print(pooled)
```

---

## 5. Example

A complete Bayesian learning benchmark is provided in `pdm_demo.py`.

**Simulation setup** (matching the paper):
- Losses follow a Pareto distribution with an unobservable tail parameter `alpha_t`
- At `t=0`, `alpha` undergoes a structural break from `alpha_minus=3.0` to `alpha_plus=1.5`
- 3 non-cooperative Bayesian experts each draw independent Pareto samples and update their posterior estimate
- Expert 1 (the pioneer) receives 6 observations per period; Experts 2-3 receive 5 each
- The supervisor never observes `alpha_t` and must pool expert estimates using one of the 8 methods

**Outputs**:
- Single-seed RMSE table comparing all methods against the true `alpha`
- 100-run Monte Carlo for robust average RMSE comparison
- Two plots: expert estimates vs. true parameter, and cumulative RMSE learning curves

Run:

```bash
pip install pandas numpy matplotlib statsmodels
python pdm_demo.py
```

---

## 6. Reference Implementation Details

The code in `pdm.py` implements all methods introduced and compared in the published article. Each method returns a `(T x N)` DataFrame of weights (or a pooled Series for median pooling) that can be passed to `pooled_forecast()`.

### 6.1 PDM Variants

#### PDM with Angles (preferred method) — `compute_pioneer_weights_angles`

The canonical 3-step method using angle-based weighting (Equation 4–5 of the paper). Angles capture the *speed* of convergence between time series.

**Step 1 — Distance reduction condition**
```
δ_distance = 𝟙( |x_i^t − m_{-i}^t| < |x_i^{t−1} − m_{-i}^{t−1}| )
```

**Step 2 — Orientation condition (angle-based)**
```
θ_i  = arctan(|Δx_i| / s)     # expert's movement angle
θ_{-i} = arctan(|Δm_{-i}| / s)  # peers' movement angle
δ_orientation = 𝟙( θ_{-i} > θ_i )
```
where `s` is the time step between observations.

**Step 3 — Proportion attribution (angle-based)**
```
w_i^t = δ_distance × δ_orientation × |θ_{-i}| / (|θ_{-i}| + |θ_i|)
```

This is the preferred approach: it accounts for the speed of convergence and is robust across configurations (Table 2 in the paper).

#### PDM with Distances — `compute_pioneer_weights_distance`

Same Steps 1–2, but replaces the angle-based weighting with y-axis distances:
```
w_i^t = δ_distance × δ_orientation × |Δm_{-i}| / (|Δm_{-i}| + |Δx_i|)
```

Found to be **non-robust** in the paper's validation (Table 2). Provided for comparison and backward compatibility.

### 6.2 Alternative Inter-Temporal Pioneer Detection Methods

#### Granger Causality (Appendix A.3) — `compute_granger_weights`

Tests whether each expert's time series Granger-causes the leave-one-out mean of the other experts (Granger 1969). Experts whose past values significantly predict the group's future receive higher weights. Weight ∝ (1 − p-value).

Related: Toda & Yamamoto (1995) for integration/cointegration; Hasbrouck (1995) for cointegration-based information share.

Requires `statsmodels`.

#### Lagged Correlation (Appendix A.4) — `compute_lagged_correlation_weights`

Measures Pearson correlation between lagged expert estimates and the current leave-one-out mean (Pearson 1895). Applied as in Sakurai, Papadimitriou & Faloutsos (2005) ("Braid" stream mining) and Forbes & Rigobon (2002) for financial applications.

#### Multivariate Linear Regressions (Appendix A.6) — `compute_multivariate_regression_weights`

Regresses the leave-one-out mean (at time t) on expert i's lagged estimate. Following Yi et al. (2000), significant regression coefficients serve as voting weights. With limited history, this reduces to searching for correlation and Granger causality.

Requires `statsmodels`.

#### Transfer Entropy (Appendix A.7) — `compute_transfer_entropy_weights`

Measures information transfer (Schreiber 2000) from each expert to the group. Continuous time series are discretized into bins (Dimpfl & Peter 2014 recommend 3 bins along the 5% and 95% quantiles). Barnett, Barrett & Seth (2009) show this is equivalent to Granger causality when variables are Gaussian.

### 6.3 Traditional Benchmarks

#### Linear Opinion Pooling — `compute_linear_pooling_weights`

Equal weights `1/N` for all experts (simple mean). The standard benchmark.

#### Median Pooling — `compute_median_pooling`

Cross-sectional median at each time period. Returns a pooled Series directly (not weights).

### 6.4 Shared Utilities

```python
pooled_forecast(forecasts, weights)  # Weighted combination with mean fallback
```

If no pioneer exists at time t (all weights NaN or zero), the pooled forecast falls back to the simple mean.

### 6.5 Usage Example

```python
import pandas as pd
from pdm import (
    compute_pioneer_weights_angles,
    compute_pioneer_weights_distance,
    compute_granger_weights,
    compute_lagged_correlation_weights,
    compute_multivariate_regression_weights,
    compute_transfer_entropy_weights,
    compute_linear_pooling_weights,
    compute_median_pooling,
    pooled_forecast,
)

# forecasts: DataFrame (T x N) of expert forecasts
weights_angles = compute_pioneer_weights_angles(forecasts)
pooled_angles  = pooled_forecast(forecasts, weights_angles)

weights_gc     = compute_granger_weights(forecasts)
pooled_gc      = pooled_forecast(forecasts, weights_gc)

median_pooled  = compute_median_pooling(forecasts)
```

---

## 7. Citing This Work

```
Vansteenberghe, Eric (2025).
Insurance Supervision under Climate Change: A Pioneer Detection Method.
The Geneva Papers on Risk and Insurance – Issues and Practice.
doi:10.1057/s41288-025-00367-y
```

---

## 8. License

MIT License.

---

## 9. Contact

For questions, collaborations, or extensions (multi‑agent systems, forecasting, insurance supervision), feel free to open an issue or contact the author.
