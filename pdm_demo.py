#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR


def fetch_ecb_hicp_inflation_panel(
    countries,
    start="1997-01-01",
    end=None,
    item="000000",
    sa="N",
    measure="4",
    variation="ANR",
    freq="M",
    timeout=60
):
    base = "https://data-api.ecb.europa.eu/service/data"
    key = f"{freq}.{'+'.join(countries)}.{sa}.{item}.{measure}.{variation}"

    params = {"format": "csvdata", "startPeriod": start}
    if end is not None:
        params["endPeriod"] = end

    url = f"{base}/ICP/{key}"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()

    raw = pd.read_csv(StringIO(r.text))

    if "TIME_PERIOD" not in raw.columns or "OBS_VALUE" not in raw.columns:
        raise ValueError(f"Unexpected ECB response format. Columns: {list(raw.columns)}")

    country_col = "REF_AREA" if "REF_AREA" in raw.columns else None
    if country_col is None:
        for cand in ["GEO", "LOCATION", "COUNTRY", "REF_AREA"]:
            if cand in raw.columns:
                country_col = cand
                break

    if country_col is None:
        raise ValueError("Could not infer country column from ECB response.")

    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"], errors="coerce")
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")

    panel = (
        raw.pivot_table(
            index="TIME_PERIOD",
            columns=country_col,
            values="OBS_VALUE",
            aggfunc="last"
        )
        .sort_index()
    )

    return panel, raw


def fetch_ukraine_cpi_prev_month_raw(start="2000-01", end="2025-12", timeout=60):
    try:
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

        txt = r.text.strip().lower()
        if txt.startswith("<!doctype html") or txt.startswith("<html"):
            raise ValueError("API returned HTML maintenance page instead of CSV data.")

        raw = pd.read_csv(StringIO(r.text), dtype=str)

        if "TIME_PERIOD" not in raw.columns or "OBS_VALUE" not in raw.columns:
            raise ValueError(f"Unexpected Ukraine response format. Columns: {list(raw.columns)}")

        raw = raw.loc[
            raw["TIME_PERIOD"].astype(str).str.match(r"^\d{4}-M\d{2}$", na=False)
            & raw["OBS_VALUE"].notna()
        ].copy()

        if raw.empty:
            raise ValueError("Ukraine response contained no valid monthly observations.")

        return raw

    except Exception as e:
        print(f"Failed to fetch Ukraine data: {e}")
        print("Proceeding without Ukraine data.")
        return None


def ua_raw_to_monthly_series(ua_raw: pd.DataFrame) -> pd.Series:
    if ua_raw is None:
        raise ValueError("ua_raw is None.")

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


def cpi_prev_month_index_to_yoy_inflation(cpi_prev_month: pd.Series) -> pd.Series:
    s = pd.to_numeric(cpi_prev_month, errors="coerce").dropna().sort_index()
    monthly_factor = s / 100.0
    yoy_factor = monthly_factor.rolling(12).apply(np.prod, raw=True)
    yoy_infl = (yoy_factor - 1.0) * 100.0
    yoy_infl.name = "UA"
    return yoy_infl


if __name__ == "__main__":
    countries = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]

    infl_panel, infl_long = fetch_ecb_hicp_inflation_panel(
        countries=countries,
        start="2000-01",
        end="2025-12"
    )

    infl_panel = infl_panel.copy()
    infl_panel.index = pd.to_datetime(infl_panel.index).to_period("M").to_timestamp(how="start")

    print("ECB panel loaded.")
    print(infl_panel.head())

    ua_raw = fetch_ukraine_cpi_prev_month_raw(start="2000-01", end="2025-12")

    if ua_raw is not None:
        print("\nUkraine raw data preview:")
        print(ua_raw.head())

        ua_idx = ua_raw_to_monthly_series(ua_raw)
        ua_idx = ua_idx.loc["2000-01-01":"2025-12-01"]

        ua_yoy = cpi_prev_month_index_to_yoy_inflation(ua_idx)
        ua_yoy.index = pd.to_datetime(ua_yoy.index).to_period("M").to_timestamp(how="start")

        infl_panel = infl_panel.join(ua_yoy, how="left")
        print("\nUkraine y/y inflation successfully added to panel.")
    else:
        print("\nUkraine raw data unavailable. Panel will contain EU countries only.")

    plt.figure(figsize=(12, 6))
    for country in infl_panel.columns:
        plt.plot(infl_panel.index, infl_panel[country], label=country, linewidth=1)

    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Inflation rate (y/y, %)")
    plt.title("Inflation Panel (ECB HICP + Ukraine if available)")
    plt.legend(ncol=3, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.show()

    df = infl_panel.copy().sort_index().dropna()

    print("\nFinal complete-case sample shape:", df.shape)
    print("Columns in final sample:", list(df.columns))

    print("\n=== ADF unit-root tests (levels) ===")
    adf_results = []

    for c in df.columns:
        try:
            stat, pval, _, _, _, _ = adfuller(df[c], autolag="AIC")
            adf_results.append({
                "country": c,
                "ADF_stat": stat,
                "pvalue": pval
            })
        except Exception as e:
            print(f"ADF failed for {c}: {e}")

    if adf_results:
        adf_table = pd.DataFrame(adf_results).sort_values("pvalue")
        print(adf_table.to_string(index=False))

    if "UA" in df.columns:
        maxlag = 6
        print("\n=== Granger causality tests: X -> UA ===")
        granger_out = []

        for c in df.columns:
            if c == "UA":
                continue

            data_gc = df[["UA", c]]

            try:
                res = grangercausalitytests(data_gc, maxlag=maxlag, verbose=False)
                min_p = min(res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1))
                granger_out.append({"country": c, "min_pvalue": min_p})
            except Exception as e:
                print(f"Granger test failed for {c}: {e}")

        if granger_out:
            granger_rank = pd.DataFrame(granger_out).sort_values("min_pvalue").reset_index(drop=True)
            print("\n=== Ranking of countries by Granger causality for UA ===")
            print(granger_rank.to_string(index=False))

            if len(granger_rank) >= 2:
                top_countries = granger_rank["country"].iloc[:2].tolist()
                var_vars = ["UA"] + top_countries
                print("\nVAR variables:", var_vars)

                X_var = df[var_vars]

                try:
                    model = VAR(X_var)
                    lag_selection = model.select_order(maxlags=6)
                    p = lag_selection.selected_orders.get("bic", 1)
                    if p is None or pd.isna(p):
                        p = 1
                    p = max(1, int(p))

                    print("\n=== VAR lag selection (BIC) ===")
                    print(lag_selection.summary())
                    print(f"Selected lag order p = {p}")

                    var_res = model.fit(p)
                    print("\n=== VAR estimation results ===")
                    print(var_res.summary())

                except Exception as e:
                    print(f"VAR estimation failed: {e}")
    else:
        print("\nUA column not available: skipping Granger causality and VAR for Ukraine.")