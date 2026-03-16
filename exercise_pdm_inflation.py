import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ecb_hicp_panel_var_granger import fetch_ecb_hicp_inflation_panel
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

countries = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]
panel, _ = fetch_ecb_hicp_inflation_panel(countries, start="2000-01", end="2025-12")
panel.index = pd.to_datetime(panel.index).to_period("M").to_timestamp(how="start")
panel = panel.dropna()

# Part A: Who pioneered European inflation dynamics?

print("Part A")
w_angles = compute_pioneer_weights_angles(panel)

# A.1 Plot pioneer weights over time
w_angles.plot(figsize=(14, 7), alpha=0.8, linewidth=1.5)
plt.title("PDM Pioneer Weights (Angles) Over Time")
plt.ylabel("Weight")
plt.xlabel("Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

print("(c) Multiple countries receive non-zero pioneer weights, but the distribution shifts over time. For example, NL and IT hold the highest weights between 2002–2007, whereas BE dominates in 2022–2023.    "
      "(d) Strong pioneers are not observed during periods of low and stable inflation. The weights is evenly distributed across the panel. In the context of PDM theory, the method is designed to detect directional convergence after a structural break. Without large shocks to drive divergence, distinct pioneers do not emerge")

# A.2 Average weights by subperiod
periods = {
    "(2002-07)": ("2002-01", "2007-12"),
    "(2008-12)": ("2008-01", "2012-12"),
    "(2013-19)": ("2013-01", "2019-12"),
    "(2020-21)": ("2020-01", "2021-12"),
    "(2022-23)": ("2022-01", "2023-12"),
    "(2024-25)": ("2024-01", "2025-12"),
}

avg_weights = pd.DataFrame(index=panel.columns, columns=periods.keys())
for name, (start, end) in periods.items():
    avg_weights[name] = w_angles.loc[start:end].mean()

print("\nAverage PDM (Angles) Weight by Subperiod:")
print(avg_weights.round(4))

print("(b) The ranking changes across subperiods. NL and IT lead in Period I. AT and NL lead in Period II. BE dominates Period V, while AT and IT drop to zero weight during the same timeframe"
      "(c) Structural characteristics determine early exposure to inflation shocks")

# Part B: Predicting Target Inflation Trajectory

print("Part B")
target = "FR"
eu_cols = [c for c in panel.columns if c != target]
eu_panel = panel[eu_cols]
actual_target = panel[target]

# B.1 Rolling pioneer detection
window_size = 12
rolling_dominant = []
dates = eu_panel.index[window_size:]

for i in range(window_size, len(eu_panel)):
    window_data = eu_panel.iloc[i - window_size:i]
    w = compute_pioneer_weights_angles(window_data)

    last_row = w.iloc[-1]
    if last_row.isna().all():
        rolling_dominant.append(np.NaN)
    else:
        rolling_dominant.append(last_row.idxmax())

dom_series = pd.Series(rolling_dominant, index=dates).dropna()

plt.figure(figsize=(14, 4))
plt.scatter(dom_series.index, dom_series.values, alpha=0.6, s=15, c='red')
plt.title(f"Dominant Pioneer for {target} (12-Month Rolling Window)")
plt.ylabel("Country Code")
plt.tight_layout()
print("dominant_pioneer_rolling.png")

print("(b) The identity of the dominant pioneer changes across subperiods as the nature of the macroeconomic shock changes")

# B.2 Forecasting evaluation
methods = {
    "PDM Angles": compute_pioneer_weights_angles,
    "PDM Distance": compute_pioneer_weights_distance,
    "Granger Causality": lambda df: compute_granger_weights(df, maxlag=3),
    "Lagged Correlation": lambda df: compute_lagged_correlation_weights(df, lag=1),
    "Multivariate Regression": lambda df: compute_multivariate_regression_weights(df, lag=1),
    "Transfer Entropy": lambda df: compute_transfer_entropy_weights(df, n_bins=3, lag=1),
    "Linear Pooling": compute_linear_pooling_weights
}

rmse_results = {}
for name, func in methods.items():
    w = func(eu_panel)
    forecast = pooled_forecast(eu_panel, w)
    rmse = np.sqrt(((forecast - actual_target) ** 2).mean())
    rmse_results[name] = rmse

median_forecast = compute_median_pooling(eu_panel)
rmse_results["Median Pooling"] = np.sqrt(((median_forecast - actual_target) ** 2).mean())

rmse_df = pd.Series(rmse_results).sort_values().to_frame("RMSE")
print("\nRMSE by Method :")
print(rmse_df.round(4))

print("(d) A low RMSE does not neceserally mean the method is a good forecaster, it simply indicates that the weighted combination of other countries inflation track well the actual target inflation")