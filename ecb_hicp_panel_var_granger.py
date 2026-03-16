#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECB-SSSU Inflation Panel - HICP (ECB) + Ukraine CPI (SSSU) with ADF, Granger, and VAR
======================================================================================

Overview
--------
Single-file, reproducible script that builds a monthly inflation panel from:

1) ECB Data Portal (SDMX 2.1 REST) - HICP inflation (y/y, %), multiple countries.
2) State Statistics Service of Ukraine (SSSU) SDMX v3 - CPI index (prev. month = 100),
   converted to y/y inflation by chaining 12 monthly factors.

It then runs a compact time-series workflow suitable for teaching and quick diagnostics:
- ADF unit-root tests on inflation levels
- Bivariate Granger causality screening (predictors -> target)
- Small VAR in levels with lag order selected by BIC

The helper functions are import-safe so they can be reused by auxiliary scripts,
including the Pioneer Detection extension.
"""

from __future__ import annotations

from io import BytesIO, StringIO
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


DEFAULT_COUNTRIES = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]
DEFAULT_START = "2000-01"
DEFAULT_END = "2025-12"


def fetch_ecb_hicp_inflation_panel(
    countries: list[str],
    start: str = "1997-01-01",
    end: str | None = None,
    item: str = "000000",
    sa: str = "N",
    measure: str = "4",
    variation: str = "ANR",
    freq: str = "M",
    timeout: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch a monthly cross-country panel of HICP inflation (annual rate of change)
    from the ECB Data Portal (ICP dataflow).
    """
    base = "https://data-api.ecb.europa.eu/service/data"
    key = f"{freq}.{'+'.join(countries)}.{sa}.{item}.{measure}.{variation}"

    params = {"format": "csvdata", "startPeriod": start}
    if end is not None:
        params["endPeriod"] = end

    url = f"{base}/ICP/{key}"
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()

    raw = pd.read_csv(StringIO(response.text))
    if "TIME_PERIOD" not in raw.columns or "OBS_VALUE" not in raw.columns:
        raise ValueError(f"Unexpected response format. Columns: {list(raw.columns)}")

    country_col = "REF_AREA" if "REF_AREA" in raw.columns else None
    if country_col is None:
        for candidate in ["GEO", "LOCATION", "COUNTRY", "REF_AREA"]:
            if candidate in raw.columns:
                country_col = candidate
                break
    if country_col is None:
        standard = {"TIME_PERIOD", "OBS_VALUE", "OBS_STATUS", "OBS_CONF", "UNIT_MULT", "DECIMALS"}
        nonstandard = [col for col in raw.columns if col not in standard]
        if not nonstandard:
            raise ValueError("Could not infer the country column from the ECB response.")
        country_col = nonstandard[0]

    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"])
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")

    panel = (
        raw.pivot_table(index="TIME_PERIOD", columns=country_col, values="OBS_VALUE", aggfunc="last")
        .sort_index()
    )
    return panel, raw


def fetch_ukraine_cpi_prev_month_raw(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Fetch Ukraine CPI (previous month = 100) from the SSSU SDMX API v3.
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

    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "text/html" in content_type or "<!doctype html" in response.text.lower():
        raise ValueError("SSSU endpoint returned an HTML maintenance page instead of SDMX-CSV data.")

    raw = pd.read_csv(StringIO(response.text), dtype=str)
    raw = raw.loc[
        raw["TIME_PERIOD"].astype(str).str.match(r"^\d{4}-M\d{2}$", na=False)
        & raw["OBS_VALUE"].notna()
    ].copy()
    return raw


def fetch_ukraine_yoy_from_nbu(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    timeout: int = 60,
) -> pd.Series:
    """
    Fallback official source for Ukraine y/y CPI from the National Bank of Ukraine.

    The workbook contains the all-items consumer price index measured as the
    percentage change relative to the corresponding month of the previous year.
    """
    url = "https://bank.gov.ua/files/macro/CPI_m.xlsx"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    workbook = pd.read_excel(BytesIO(response.content), sheet_name=1, header=None)
    dates = pd.to_datetime(workbook.iloc[1, 2:], errors="coerce")
    values = pd.to_numeric(workbook.iloc[2, 2:], errors="coerce")

    series = pd.Series(values.to_numpy(), index=dates, name="UA").dropna()
    series.index = pd.to_datetime(series.index).to_period("M").to_timestamp(how="start")
    return series.sort_index().loc[f"{start}-01":f"{end}-01"]


def ua_raw_to_monthly_series(ua_raw: pd.DataFrame) -> pd.Series:
    """
    Build a clean monthly time series from SSSU SDMX-CSV raw output.
    """
    if "TIME_PERIOD" not in ua_raw.columns or "OBS_VALUE" not in ua_raw.columns:
        raise ValueError(f"ua_raw must contain TIME_PERIOD and OBS_VALUE. Columns: {list(ua_raw.columns)}")

    series = ua_raw[["TIME_PERIOD", "OBS_VALUE"]].copy()
    series["TIME_PERIOD"] = series["TIME_PERIOD"].astype(str).str.strip()
    series = series[series["TIME_PERIOD"].str.match(r"^\d{4}-M\d{2}$", na=False)]

    series["TIME_PERIOD"] = pd.to_datetime(
        series["TIME_PERIOD"].str.replace(r"^(\d{4})-M(\d{2})$", r"\1-\2-01", regex=True),
        errors="coerce",
    )
    series["OBS_VALUE"] = pd.to_numeric(
        series["OBS_VALUE"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    series = series.dropna(subset=["TIME_PERIOD", "OBS_VALUE"]).sort_values("TIME_PERIOD")
    out = series.set_index("TIME_PERIOD")["OBS_VALUE"].rename("UA_IDX_PREV_MONTH_100")
    return out.groupby(level=0).last()


def cpi_prev_month_index_to_yoy_inflation(idx_prev_month_100: pd.Series) -> pd.Series:
    """
    Convert CPI expressed as previous-month = 100 into y/y inflation (%).
    """
    monthly_factor = (idx_prev_month_100 / 100.0).astype(float)
    yoy_factor = monthly_factor.rolling(12).apply(np.prod, raw=True)
    return ((yoy_factor - 1.0) * 100.0).rename("UA")


def build_inflation_panel(
    countries: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Build the merged monthly inflation panel used by the teaching workflow.
    """
    countries = countries or DEFAULT_COUNTRIES
    infl_panel, _ = fetch_ecb_hicp_inflation_panel(
        countries=countries,
        start=start,
        end=end,
        timeout=timeout,
    )

    try:
        ua_raw = fetch_ukraine_cpi_prev_month_raw(start=start, end=end, timeout=timeout)
        ua_idx = ua_raw_to_monthly_series(ua_raw).loc[f"{start}-01":f"{end}-01"]
        ua_yoy = cpi_prev_month_index_to_yoy_inflation(ua_idx)
    except Exception as exc:
        warnings.warn(
            f"Falling back to National Bank of Ukraine CPI y/y data because the SSSU API is unavailable: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        ua_yoy = fetch_ukraine_yoy_from_nbu(start=start, end=end, timeout=timeout)

    infl_panel = infl_panel.copy()
    infl_panel.index = pd.to_datetime(infl_panel.index).to_period("M").to_timestamp(how="start")
    ua_yoy.index = pd.to_datetime(ua_yoy.index).to_period("M").to_timestamp(how="start")

    return infl_panel.join(ua_yoy, how="left").sort_index()


def plot_inflation_panel(panel: pd.DataFrame, title: str = "HICP Inflation Panel (ECB Data Portal)") -> None:
    """
    Plot the multi-country inflation panel.
    """
    plt.figure(figsize=(12, 6))
    for country in panel.columns:
        plt.plot(panel.index, panel[country], label=country, linewidth=1)

    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Inflation rate (y/y, %)")
    plt.title(title)
    plt.legend(ncol=3, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.show()


def run_adf_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run ADF unit-root tests on all panel columns.
    """
    adf_results = []
    for column in df.columns:
        stat, pval, _, _, _, _ = adfuller(df[column], autolag="AIC")
        adf_results.append({"country": column, "ADF_stat": stat, "pvalue": pval})
    return pd.DataFrame(adf_results).sort_values("pvalue").reset_index(drop=True)


def run_granger_screen(df: pd.DataFrame, target: str = "UA", maxlag: int = 6) -> pd.DataFrame:
    """
    Rank countries by bivariate Granger-causality strength for the target series.
    """
    granger_out = []
    for column in df.columns:
        if column == target:
            continue

        data_gc = df[[target, column]]
        try:
            res = grangercausalitytests(data_gc, maxlag=maxlag, verbose=False)
            min_p = min(res[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1))
            granger_out.append({"country": column, "min_pvalue": min_p})
        except Exception as exc:
            print(f"Granger test failed for {column}: {exc}")

    return pd.DataFrame(granger_out).sort_values("min_pvalue").reset_index(drop=True)


def fit_small_var(
    df: pd.DataFrame,
    target: str = "UA",
    predictors: list[str] | None = None,
    top_n: int = 2,
    maxlags: int = 6,
):
    """
    Estimate a small VAR in levels with BIC lag selection.
    """
    if predictors is None:
        granger_rank = run_granger_screen(df, target=target, maxlag=maxlags)
        predictors = granger_rank["country"].iloc[:top_n].tolist()

    var_vars = [target] + predictors
    model = VAR(df[var_vars])
    lag_selection = model.select_order(maxlags=maxlags)
    selected_lag = lag_selection.selected_orders["bic"]
    selected_lag = max(1, selected_lag if selected_lag is not None else 1)
    results = model.fit(selected_lag)
    return var_vars, lag_selection, selected_lag, results


def main() -> None:
    """
    Execute the original teaching workflow end-to-end.
    """
    infl_panel = build_inflation_panel()
    plot_inflation_panel(infl_panel)

    df = infl_panel.copy().sort_index().dropna()

    print("\n=== ADF unit-root tests (levels) ===")
    adf_table = run_adf_tests(df)
    print(adf_table.to_string(index=False))

    print("\n=== Granger causality tests: X -> UA ===")
    granger_rank = run_granger_screen(df, target="UA", maxlag=6)
    print("\n=== Ranking of countries by Granger causality for UA ===")
    print(granger_rank.to_string(index=False))

    top_countries = granger_rank["country"].iloc[:2].tolist()
    print("\nVAR variables:", ["UA"] + top_countries)

    var_vars, lag_selection, selected_lag, var_res = fit_small_var(
        df,
        target="UA",
        predictors=top_countries,
        maxlags=6,
    )

    print("\n=== VAR lag selection (BIC) ===")
    print(lag_selection.summary())
    print(f"Selected lag order p = {selected_lag}")

    print("\n=== VAR estimation results ===")
    print(var_res.summary())


if __name__ == "__main__":
    main()
