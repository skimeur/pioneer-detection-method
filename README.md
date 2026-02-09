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
- `pdm.py` – minimal, pedagogical implementation of the method  
- `pdm_demo.py` – complete synthetic example with plots  
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
pip install pandas numpy matplotlib
```

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

A complete demonstration is provided in:

```
examples/pdm_demo.py
```

This script:

- generates a synthetic panel of three experts  
- simulates a regime shift  
- computes pioneer weights  
- plots individual experts, simple mean, and PDM pooled forecasts  

Run:

```bash
python examples/pdm_demo.py
```

---

## 6. Reference Implementation Details

The code in `pdm.py` follows the exact sequence described in the published article:

**Step 1 — Distance condition**  
Experts move closer to the group:  
```
|x_i^t − m_-i^t| < |x_i^{t−1} − m_-i^{t−1}|
```

**Step 2 — Orientation condition**  
Peers move more toward the expert than vice‑versa:  
```
|Δm_-i^t| > |Δx_i^t|
```

**Step 3 — Proportion attribution**  
Relative contribution of convergence:  
```
|Δm_-i^t| / (|Δm_-i^t| + |Δx_i^t|)
```

If no pioneer exists → fallback to the simple mean.

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

## Test modification by Emma Blindauer