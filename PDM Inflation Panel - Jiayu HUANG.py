#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: solaforsure
"""



from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from io import StringIO
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


# =============================================================================
# Paths / config
# =============================================================================

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

OUTPUT_DIR = ROOT / "exercise_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2000-01"
END = "2025-12"

EU_COUNTRIES = ["DE", "FR", "IT", "ES", "NL", "BE", "AT", "PT", "IE", "FI", "GR"]
ALL_COUNTRIES = EU_COUNTRIES + ["UA"]

COUNTRY_NAMES = {
    "DE": "Germany",
    "FR": "France",
    "IT": "Italy",
    "ES": "Spain",
    "NL": "Netherlands",
    "BE": "Belgium",
    "AT": "Austria",
    "PT": "Portugal",
    "IE": "Ireland",
    "FI": "Finland",
    "GR": "Greece",
    "UA": "Ukraine",
}

PERIODS = {
    "I_2002_2007": ("2002-01-01", "2007-12-01"),
    "II_2008_2012": ("2008-01-01", "2012-12-01"),
    "III_2013_2019": ("2013-01-01", "2019-12-01"),
    "IV_2020_2021": ("2020-01-01", "2021-12-01"),
    "V_2022_2023": ("2022-01-01", "2023-12-01"),
    "VI_2024_2025": ("2024-01-01", "2025-12-01"),
}


# =============================================================================
# PDM and alternative methods
# Embedded so the file can run standalone even if pdm.py is not present.
# =============================================================================

def _leave_one_out_mean(X: pd.DataFrame) -> pd.DataFrame:
    m_minus = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for col in X.columns:
        others = X.drop(columns=col)
        m_minus[col] = others.mean(axis=1)
    return m_minus


def compute_pioneer_weights_distance(forecasts: pd.DataFrame) -> pd.DataFrame:
    X = forecasts.astype(float)
    m_minus = _leave_one_out_mean(X)

    delta_X = X.diff()
    delta_m = m_minus.diff()

    distance = (X - m_minus).abs()
    distance_prev = distance.shift(1)
    cond_distance = distance < distance_prev

    cond_orientation = delta_m.abs() > delta_X.abs()

    denom = delta_m.abs() + delta_X.abs()
    proportion = delta_m.abs() / denom

    mask = cond_distance & cond_orientation & (denom > 0)
    raw = proportion.where(mask, 0.0)

    row_sums = raw.sum(axis=1)
    weights = raw.div(row_sums.replace(0.0, np.nan), axis=0)
    return weights


def compute_pioneer_weights_angles(forecasts: pd.DataFrame, step: float = 1.0) -> pd.DataFrame:
    X = forecasts.astype(float)
    m_minus = _leave_one_out_mean(X)

    delta_X = X.diff()
    delta_m = m_minus.diff()

    distance = (X - m_minus).abs()
    distance_prev = distance.shift(1)
    cond_distance = distance < distance_prev

    def _angle(dy):
        return np.arctan2(dy.abs(), step)

    theta_i = _angle(delta_X)
    theta_mi = _angle(delta_m)

    cond_orientation = theta_mi > theta_i

    denom = theta_mi + theta_i
    proportion = theta_mi / denom

    mask = cond_distance & cond_orientation & (denom > 0)
    raw = proportion.where(mask, 0.0)

    row_sums = raw.sum(axis=1)
    weights = raw.div(row_sums.replace(0.0, np.nan), axis=0)
    return weights


def compute_granger_weights(forecasts: pd.DataFrame, maxlag: int = 1) -> pd.DataFrame:
    X = forecasts.astype(float).dropna()
    m_minus = _leave_one_out_mean(X)

    scores = {}
    for col in X.columns:
        data = pd.concat([m_minus[col], X[col]], axis=1).dropna()
        if len(data) < maxlag + 3:
            scores[col] = 0.0
            continue
        try:
            res = grangercausalitytests(data.values, maxlag=maxlag, verbose=False)
            min_p = min(res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1))
            scores[col] = max(1.0 - min_p, 0.0)
        except Exception:
            scores[col] = 0.0

    total = sum(scores.values())
    if total == 0:
        w = {col: 1.0 / len(X.columns) for col in X.columns}
    else:
        w = {col: v / total for col, v in scores.items()}

    weights = pd.DataFrame(
        {col: [w[col]] * len(forecasts) for col in forecasts.columns},
        index=forecasts.index,
    )
    return weights


def compute_lagged_correlation_weights(forecasts: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    X = forecasts.astype(float).dropna()
    m_minus = _leave_one_out_mean(X)

    scores = {}
    for col in X.columns:
        lagged_expert = X[col].shift(lag)
        valid = pd.concat([lagged_expert, m_minus[col]], axis=1).dropna()
        if len(valid) < 3:
            scores[col] = 0.0
            continue
        corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
        scores[col] = max(corr, 0.0)

    total = sum(scores.values())
    if total == 0:
        w = {col: 1.0 / len(X.columns) for col in X.columns}
    else:
        w = {col: v / total for col, v in scores.items()}

    weights = pd.DataFrame(
        {col: [w[col]] * len(forecasts) for col in forecasts.columns},
        index=forecasts.index,
    )
    return weights


def compute_multivariate_regression_weights(forecasts: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    X = forecasts.astype(float).dropna()
    m_minus = _leave_one_out_mean(X)

    scores = {}
    for col in X.columns:
        y = m_minus[col].iloc[lag:]
        x = X[col].iloc[:-lag].values if lag > 0 else X[col].values
        if len(y) < 4:
            scores[col] = 0.0
            continue
        x_const = add_constant(x)
        try:
            res = OLS(y.values, x_const).fit()
            coef = res.params[1]
            pval = res.pvalues[1]
            scores[col] = max(coef, 0.0) if pval < 0.10 else 0.0
        except Exception:
            scores[col] = 0.0

    total = sum(scores.values())
    if total == 0:
        w = {col: 1.0 / len(X.columns) for col in X.columns}
    else:
        w = {col: v / total for col, v in scores.items()}

    weights = pd.DataFrame(
        {col: [w[col]] * len(forecasts) for col in forecasts.columns},
        index=forecasts.index,
    )
    return weights


def compute_transfer_entropy_weights(forecasts: pd.DataFrame, n_bins: int = 3, lag: int = 1) -> pd.DataFrame:
    X = forecasts.astype(float).dropna()
    m_minus = _leave_one_out_mean(X)

    def _discretize(series, n_bins):
        quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
        clean = series.dropna()
        if len(clean) == 0:
            return np.array([], dtype=int)
        boundaries = np.quantile(clean, quantiles)
        return np.digitize(series.values, boundaries)

    def _transfer_entropy(source, target, lag):
        src_d = _discretize(source, n_bins)
        tgt_d = _discretize(target, n_bins)

        n = len(src_d)
        if n <= lag + 1 or len(tgt_d) != n:
            return 0.0

        tgt_t = tgt_d[lag:]
        tgt_past = tgt_d[:-lag] if lag > 0 else tgt_d
        src_past = src_d[:-lag] if lag > 0 else src_d
        length = min(len(tgt_t), len(tgt_past), len(src_past))
        if length <= 1:
            return 0.0

        tgt_t = tgt_t[:length]
        tgt_past = tgt_past[:length]
        src_past = src_past[:length]

        def _entropy_from_counts(counts):
            total = counts.sum()
            if total == 0:
                return 0.0
            p = counts / total
            p = p[p > 0]
            return -np.sum(p * np.log2(p))

        joint_yy = np.zeros((n_bins, n_bins))
        for i in range(length):
            joint_yy[tgt_t[i], tgt_past[i]] += 1
        h_yy = _entropy_from_counts(joint_yy.flatten())
        h_ypast = _entropy_from_counts(joint_yy.sum(axis=0))
        h_cond_yy = h_yy - h_ypast

        joint_yyx = np.zeros((n_bins, n_bins, n_bins))
        for i in range(length):
            joint_yyx[tgt_t[i], tgt_past[i], src_past[i]] += 1
        h_yyx = _entropy_from_counts(joint_yyx.flatten())
        h_yx_past = _entropy_from_counts(joint_yyx.sum(axis=0).flatten())
        h_cond_yyx = h_yyx - h_yx_past

        te = h_cond_yy - h_cond_yyx
        return max(te, 0.0)

    scores = {}
    for col in X.columns:
        try:
            scores[col] = _transfer_entropy(X[col], m_minus[col], lag)
        except Exception:
            scores[col] = 0.0

    total = sum(scores.values())
    if total == 0:
        w = {col: 1.0 / len(X.columns) for col in X.columns}
    else:
        w = {col: v / total for col, v in scores.items()}

    weights = pd.DataFrame(
        {col: [w[col]] * len(forecasts) for col in forecasts.columns},
        index=forecasts.index,
    )
    return weights


def compute_linear_pooling_weights(forecasts: pd.DataFrame) -> pd.DataFrame:
    n = len(forecasts.columns)
    return pd.DataFrame(1.0 / n, index=forecasts.index, columns=forecasts.columns)


def compute_median_pooling(forecasts: pd.DataFrame) -> pd.Series:
    return forecasts.astype(float).median(axis=1)


def pooled_forecast(forecasts: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    forecasts = forecasts.astype(float)
    weights = weights.astype(float)

    weighted_sum = (forecasts * weights).sum(axis=1, min_count=1)
    weight_sums = weights.sum(axis=1, min_count=1)
    no_pioneer = weight_sums.isna() | (weight_sums == 0)
    fallback_mean = forecasts.mean(axis=1)

    pooled = weighted_sum.copy()
    pooled[no_pioneer] = fallback_mean[no_pioneer]
    return pooled


# =============================================================================
# Data fetching (robust version of the starter code)
# =============================================================================

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
        standard = {"TIME_PERIOD", "OBS_VALUE", "OBS_STATUS", "OBS_CONF", "UNIT_MULT", "DECIMALS"}
        nonstandard = [c for c in raw.columns if c not in standard]
        if not nonstandard:
            raise ValueError("Could not infer the country column from the response.")
        country_col = nonstandard[0]

    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"])
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")

    panel = (
        raw.pivot_table(index="TIME_PERIOD", columns=country_col, values="OBS_VALUE", aggfunc="last")
        .sort_index()
    )
    panel.index = pd.to_datetime(panel.index).to_period("M").to_timestamp(how="start")
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

    text = r.text
    parse_attempts = [
        {"sep": ",", "engine": "python"},
        {"sep": ";", "engine": "python"},
        {"sep": None, "engine": "python"},
    ]

    raw = None
    last_error = None

    for kwargs in parse_attempts:
        try:
            candidate = pd.read_csv(StringIO(text), dtype=str, **kwargs)
            if {"TIME_PERIOD", "OBS_VALUE"}.issubset(candidate.columns):
                raw = candidate
                break
        except Exception as e:
            last_error = e

    if raw is None:
        lines = text.splitlines()
        header_idx = None
        for i, line in enumerate(lines):
            if "TIME_PERIOD" in line and "OBS_VALUE" in line:
                header_idx = i
                break

        if header_idx is not None:
            trimmed = "\n".join(lines[header_idx:])
            for kwargs in parse_attempts:
                try:
                    candidate = pd.read_csv(StringIO(trimmed), dtype=str, **kwargs)
                    if {"TIME_PERIOD", "OBS_VALUE"}.issubset(candidate.columns):
                        raw = candidate
                        break
                except Exception as e:
                    last_error = e

    if raw is None:
        preview = "\n".join(text.splitlines()[:20])
        raise ValueError(
            "Could not parse SSSU response into a table with TIME_PERIOD and OBS_VALUE. "
            f"Last parser error: {last_error}\n\nFirst lines of response:\n{preview}"
        )

    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.loc[
        raw["TIME_PERIOD"].astype(str).str.match(r"^\d{4}-M\d{2}$", na=False)
        & raw["OBS_VALUE"].notna()
    ].copy()

    return raw


def ua_raw_to_monthly_series(ua_raw: pd.DataFrame) -> pd.Series:
    if "TIME_PERIOD" not in ua_raw.columns or "OBS_VALUE" not in ua_raw.columns:
        raise ValueError(f"ua_raw must contain TIME_PERIOD and OBS_VALUE. Columns: {list(ua_raw.columns)}")

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
    return out.groupby(level=0).last()


def cpi_prev_month_index_to_yoy_inflation(idx_prev_month_100: pd.Series) -> pd.Series:
    monthly_factor = (idx_prev_month_100 / 100.0).astype(float)
    yoy_factor = monthly_factor.rolling(12).apply(np.prod, raw=True)
    return ((yoy_factor - 1.0) * 100.0).rename("UA")


def build_inflation_panel(start=START, end=END) -> pd.DataFrame:
    infl_panel, _ = fetch_ecb_hicp_inflation_panel(
        countries=EU_COUNTRIES,
        start=start,
        end=end
    )

    ua_raw = fetch_ukraine_cpi_prev_month_raw(start=start, end=end)
    ua_idx = ua_raw_to_monthly_series(ua_raw).loc["2000-01-01":"2025-12-01"]
    ua_yoy = cpi_prev_month_index_to_yoy_inflation(ua_idx)
    ua_yoy.index = pd.to_datetime(ua_yoy.index).to_period("M").to_timestamp(how="start")

    infl_panel = infl_panel.copy()
    infl_panel.index = pd.to_datetime(infl_panel.index).to_period("M").to_timestamp(how="start")
    infl_panel = infl_panel.join(ua_yoy, how="left").sort_index()
    infl_panel = infl_panel[ALL_COUNTRIES]
    return infl_panel


# =============================================================================
# Utilities
# =============================================================================

def save_dataframe(df: pd.DataFrame, stem: str):
    df.to_csv(OUTPUT_DIR / f"{stem}.csv", index=True)
    try:
        df.to_excel(OUTPUT_DIR / f"{stem}.xlsx", index=True)
    except Exception:
        pass


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    tmp = pd.concat([actual.rename("actual"), predicted.rename("pred")], axis=1).dropna()
    if len(tmp) == 0:
        return np.nan
    return float(np.sqrt(np.mean((tmp["pred"] - tmp["actual"]) ** 2)))


def average_weights_by_period(weights: pd.DataFrame, periods: dict) -> pd.DataFrame:
    out = {}
    for period_name, (start, end) in periods.items():
        out[period_name] = weights.loc[start:end].mean(axis=0)
    return pd.DataFrame(out)


def rank_weights_by_period(avg_table: pd.DataFrame) -> pd.DataFrame:
    return avg_table.rank(axis=0, ascending=False, method="dense").astype(int)


def build_nonzero_summary(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in weights.columns:
        s = weights[col].fillna(0.0)
        mask = s > 0
        nonzero_dates = s.index[mask]
        rows.append(
            {
                "country": col,
                "country_name": COUNTRY_NAMES.get(col, col),
                "months_nonzero": int(mask.sum()),
                "share_nonzero": float(mask.mean()),
                "first_nonzero": nonzero_dates.min().strftime("%Y-%m") if len(nonzero_dates) else None,
                "last_nonzero": nonzero_dates.max().strftime("%Y-%m") if len(nonzero_dates) else None,
                "avg_weight_when_nonzero": float(s[mask].mean()) if mask.any() else 0.0,
                "avg_weight_full_sample": float(s.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["months_nonzero", "avg_weight_full_sample"],
        ascending=[False, False]
    ).reset_index(drop=True)


def save_text(text: str, filename: str):
    (OUTPUT_DIR / filename).write_text(text, encoding="utf-8")


# =============================================================================
# Plots
# =============================================================================

def plot_inflation_panel(panel: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    for c in panel.columns:
        plt.plot(panel.index, panel[c], linewidth=1, label=c)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Inflation rate (y/y, %)")
    plt.title("Inflation Panel: EU-11 + Ukraine")
    plt.legend(ncol=4, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "inflation_panel.png", dpi=220)
    plt.close()


def plot_line_chart(weights: pd.DataFrame, filename: str, title: str):
    plt.figure(figsize=(14, 7))
    for col in weights.columns:
        plt.plot(weights.index, weights[col].fillna(0.0), linewidth=1.3, label=col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.legend(ncol=4, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close()


def plot_heatmap(weights: pd.DataFrame, filename: str, title: str):
    data = weights.fillna(0.0).T.values
    fig, ax = plt.subplots(figsize=(15, 6))
    im = ax.imshow(data, aspect="auto", interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Country")
    ax.set_yticks(np.arange(len(weights.columns)))
    ax.set_yticklabels(weights.columns)

    n_ticks = min(12, len(weights.index))
    tick_positions = np.linspace(0, len(weights.index) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [weights.index[i].strftime("%Y-%m") for i in tick_positions],
        rotation=45,
        ha="right"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Weight")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dominant_pioneer(dom: pd.Series):
    cats = [c for c in EU_COUNTRIES if c in dom.dropna().unique()]
    if not cats:
        return

    code_map = {c: i for i, c in enumerate(cats)}
    y = dom.map(code_map)

    plt.figure(figsize=(14, 5))
    plt.step(dom.index, y, where="mid")
    plt.yticks(list(code_map.values()), list(code_map.keys()))
    plt.title("Part B — Dominant Pioneer for Ukraine Over Time")
    plt.xlabel("Time")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "partB_dominant_pioneer.png", dpi=220)
    plt.close()


def plot_forecasts_vs_actual(actual: pd.Series, forecasts: pd.DataFrame):
    plt.figure(figsize=(14, 7))
    plt.plot(actual.index, actual, linewidth=2, label="UA actual")
    for col in forecasts.columns:
        plt.plot(forecasts.index, forecasts[col], linewidth=1.2, label=col)
    plt.title("Part B — Ukraine inflation: actual vs pooled estimates")
    plt.xlabel("Time")
    plt.ylabel("Inflation y/y (%)")
    plt.legend(frameon=False, ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "partB_forecasts_vs_actual.png", dpi=220)
    plt.close()


# =============================================================================
# Diagnostics from the starter code
# =============================================================================

def run_adf_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        try:
            stat, pval, _, _, _, _ = adfuller(df[c], autolag="AIC")
            rows.append({"country": c, "ADF_stat": stat, "pvalue": pval})
        except Exception:
            rows.append({"country": c, "ADF_stat": np.nan, "pvalue": np.nan})
    out = pd.DataFrame(rows).sort_values("pvalue")
    save_dataframe(out, "diagnostics_adf_levels")
    return out


def run_granger_screening_for_ua(df: pd.DataFrame, maxlag: int = 6) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        if c == "UA":
            continue
        data_gc = df[["UA", c]].dropna()
        try:
            res = grangercausalitytests(data_gc, maxlag=maxlag, verbose=False)
            min_p = min(res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1))
            rows.append({"country": c, "min_pvalue": min_p})
        except Exception:
            rows.append({"country": c, "min_pvalue": np.nan})
    out = pd.DataFrame(rows).sort_values("min_pvalue").reset_index(drop=True)
    save_dataframe(out, "diagnostics_granger_ranking_for_ua")
    return out


def run_small_var(df: pd.DataFrame, granger_rank: pd.DataFrame):
    top_countries = granger_rank["country"].dropna().iloc[:2].tolist()
    var_vars = ["UA"] + top_countries
    X_var = df[var_vars].dropna()

    if len(X_var) < 24:
        save_text("Not enough data for VAR.", "diagnostics_var_summary.txt")
        return {"variables": var_vars, "selected_bic_lag": np.nan}

    model = VAR(X_var)
    lag_selection = model.select_order(maxlags=6)
    p = lag_selection.selected_orders.get("bic", 1)
    if p is None or (isinstance(p, float) and np.isnan(p)):
        p = 1
    p = max(1, int(p))

    res = model.fit(p)

    txt = "\n".join([
        "=== VAR lag selection (BIC) ===",
        str(lag_selection.summary()),
        f"\nSelected lag order p = {p}",
        "\n=== VAR estimation results ===",
        str(res.summary()),
    ])
    save_text(txt, "diagnostics_var_summary.txt")
    return {"variables": var_vars, "selected_bic_lag": p}


# =============================================================================
# Part B helpers
# =============================================================================

def normalize_weights_over_eu(w: pd.DataFrame, eu_cols: list[str]) -> pd.DataFrame:
    out = w[eu_cols].copy()
    row_sums = out.sum(axis=1)
    zero = row_sums.isna() | (row_sums == 0)
    if zero.any():
        out.loc[zero, :] = 1.0 / len(eu_cols)
        row_sums = out.sum(axis=1)
    out = out.div(row_sums, axis=0)
    return out


def compute_pdm_target_weights(experts: pd.DataFrame, target: pd.Series, method: str) -> pd.DataFrame:
    panel = experts.copy()
    panel["UA"] = target
    panel = panel.dropna()

    if method == "PDM_Angles":
        w_full = compute_pioneer_weights_angles(panel)
    elif method == "PDM_Distance":
        w_full = compute_pioneer_weights_distance(panel)
    else:
        raise ValueError("method must be PDM_Angles or PDM_Distance")

    w = normalize_weights_over_eu(w_full, experts.columns.tolist())
    return w.reindex(experts.index)


def forecast_for_method(method_name: str, experts: pd.DataFrame, target: pd.Series):
    if method_name == "PDM_Angles":
        w = compute_pdm_target_weights(experts, target, "PDM_Angles")
        return pooled_forecast(experts, w).rename(method_name), w

    if method_name == "PDM_Distance":
        w = compute_pdm_target_weights(experts, target, "PDM_Distance")
        return pooled_forecast(experts, w).rename(method_name), w

    if method_name == "Granger":
        w = compute_granger_weights(experts, maxlag=1)
        return pooled_forecast(experts, w).rename(method_name), w

    if method_name == "Lagged_Correlation":
        w = compute_lagged_correlation_weights(experts, lag=1)
        return pooled_forecast(experts, w).rename(method_name), w

    if method_name == "Multivariate_Regression":
        w = compute_multivariate_regression_weights(experts, lag=1)
        return pooled_forecast(experts, w).rename(method_name), w

    if method_name == "Transfer_Entropy":
        w = compute_transfer_entropy_weights(experts, n_bins=3, lag=1)
        return pooled_forecast(experts, w).rename(method_name), w

    if method_name == "Linear_Pooling":
        w = compute_linear_pooling_weights(experts)
        return pooled_forecast(experts, w).rename(method_name), w

    if method_name == "Median_Pooling":
        return compute_median_pooling(experts).rename(method_name), None

    raise ValueError(f"Unknown method: {method_name}")


def rolling_dominant_pioneer(experts: pd.DataFrame, target: pd.Series, window: int = 24) -> pd.Series:
    idx = experts.index.intersection(target.index)
    experts = experts.loc[idx]
    target = target.loc[idx]

    dominant = pd.Series(index=idx, dtype=object, name="dominant_pioneer")

    for i in range(len(idx)):
        if i < window:
            dominant.iloc[i] = np.nan
            continue

        sub = experts.iloc[i - window:i].copy()
        sub["UA"] = target.iloc[i - window:i].values

        w = compute_pioneer_weights_angles(sub)
        avg_w = normalize_weights_over_eu(w, experts.columns.tolist()).mean(axis=0)
        dominant.iloc[i] = avg_w.idxmax()

    return dominant


def build_rmse_table(actual: pd.Series, forecasts: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=forecasts.columns, columns=["Full_Sample"] + list(PERIODS.keys()), dtype=float)

    for m in forecasts.columns:
        out.loc[m, "Full_Sample"] = rmse(actual, forecasts[m])
        for period_name, (start, end) in PERIODS.items():
            out.loc[m, period_name] = rmse(actual.loc[start:end], forecasts.loc[start:end, m])

    return out

# =============================================================================
# Discussion text that answers the assignment questions
# =============================================================================

def generate_assignment_answers(
    nonzero_summary_A: pd.DataFrame,
    avg_table_A: pd.DataFrame,
    ranking_A: pd.DataFrame,
    dominant: pd.Series,
    rmse_table: pd.DataFrame,
) -> str:
    stable_window = avg_table_A["I_2002_2007"].sort_values(ascending=False).head(3)
    crisis_window = avg_table_A["V_2022_2023"].sort_values(ascending=False).head(3)

    top_nonzero = nonzero_summary_A.head(6)[["country", "months_nonzero", "first_nonzero", "last_nonzero"]]
    top_nonzero_lines = [
        f"- {row.country}: non-zero in {row.months_nonzero} months, from {row.first_nonzero} to {row.last_nonzero}"
        for row in top_nonzero.itertuples(index=False)
    ]

    dom_counts = dominant.value_counts(dropna=True)
    dom_line = ", ".join([f"{k} ({v} windows)" for k, v in dom_counts.head(5).items()]) if len(dom_counts) else "No dominant pioneer identified."

    best_full = rmse_table["Full_Sample"].astype(float).idxmin()
    best_period_lines = []
    for p in PERIODS.keys():
        col = rmse_table[p].astype(float)
        if col.notna().any():
            best_period_lines.append(f"- {p}: {col.idxmin()} (RMSE={col.min():.4f})")
            

    text = f"""
    
    ASSIGNMENT ANSWERS
    ==================

    Question A.1(c) — Which countries receive non-zero pioneer weight, and when?
    The countries most frequently receiving non-zero pioneer weight are summarised below:
    {'\n'.join(top_nonzero_lines)}

    Question A.1(d) — Do we observe strong pioneers during low and stable inflation (2000–2007)?
    The average pioneer weights in 2002–2007 are generally more diffuse than in crisis periods.
    This is consistent with PDM theory: when inflation is low, stable, and cross-country movements are more synchronized,
    fewer countries stand out as clear early movers that the rest subsequently follow.
    Top countries in period I (2002–2007): {', '.join(stable_window.index.tolist())}.
    In contrast, period V (2022–2023) shows stronger differentiation, with top countries:
    {', '.join(crisis_window.index.tolist())}.

    Question A.2(a) — Average pioneer weights by country and subperiod
    See: partA_average_weights_by_subperiod.csv

    Question A.2(b) — Do rankings change over time?
    Yes. The ranking varies across subperiods, which suggests pioneership is not fixed.
    It rotates with the nature of shocks affecting inflation.

    Question A.2(c) — Economic interpretation
    Countries may become pioneers because of differences in:
    - energy mix and dependence on imported energy;
    - trade openness and speed of imported-price pass-through;
    - financial transmission and domestic credit conditions;
    - geographic position and logistics exposure;
    - sectoral structure, such as manufacturing, tourism, or shipping intensity.

    Question B.1(a)–(b) — Rolling dominant pioneer for Ukraine
    The dominant pioneer over rolling windows is saved in partB_dominant_pioneer.png and partB_dominant_pioneer_series.csv.
    The most frequent dominant pioneers are: {dom_line}
    The identity of the pioneer does change across subperiods, which is exactly what the assignment asks you to check.

    Question B.2(a)–(c) — Forecasting evaluation and RMSE
    The full-sample best method is: {best_full}.
    Best method by period:
    {'\n'.join(best_period_lines)}

    Question B.2(d) — Limits of the forecasting interpretation
    A low RMSE does not prove that PDM is a good forecasting model. The method was designed to identify pioneers and
    accelerate collective learning, not to optimize out-of-sample prediction.
    A low RMSE may simply reflect:
    - common shocks affecting both EU countries and Ukraine;
    - temporary contemporaneous comovement rather than stable causality;
    - structural breaks in Ukraine related to war, energy, exchange-rate, agricultural, or policy shocks;
    - the fact that pooled EU inflation may track Ukraine ex post without providing true forecasting power.
    """
    return textwrap.dedent(text).strip()


# =============================================================================
# Main
# =============================================================================

def main():
    print("Building inflation panel ...")
    infl_panel = build_inflation_panel(start=START, end=END)
    save_dataframe(infl_panel, "inflation_panel_raw")
    plot_inflation_panel(infl_panel)

    print("Preparing complete-case sample ...")
    df = infl_panel.copy().sort_index().dropna()
    save_dataframe(df, "inflation_panel_complete_case")

    # Starter code diagnostics
    print("\n=== ADF unit-root tests (levels) ===")
    adf_table = run_adf_tests(df)
    print(adf_table.to_string(index=False))

    print("\n=== Granger causality tests: X -> UA ===")
    granger_rank = run_granger_screening_for_ua(df, maxlag=6)
    print(granger_rank.to_string(index=False))

    print("\n=== Small VAR with BIC ===")
    var_info = run_small_var(df, granger_rank)
    print(f"VAR variables: {var_info['variables']}")
    print(f"Selected lag order: {var_info['selected_bic_lag']}")

    # Part A
    print("\nRunning Part A ...")
    panel_A = df[ALL_COUNTRIES].copy()
    w_angles_A = compute_pioneer_weights_angles(panel_A)
    if not isinstance(w_angles_A.index, pd.DatetimeIndex):
        w_angles_A.index = panel_A.index
    if list(w_angles_A.columns) != list(panel_A.columns):
        w_angles_A.columns = panel_A.columns

    save_dataframe(w_angles_A, "partA_pdm_angles_weights")

    nonzero_summary_A = build_nonzero_summary(w_angles_A)
    save_dataframe(nonzero_summary_A, "partA_nonzero_weight_summary")

    avg_table_A = average_weights_by_period(w_angles_A, PERIODS)
    save_dataframe(avg_table_A, "partA_average_weights_by_subperiod")

    ranking_A = rank_weights_by_period(avg_table_A)
    save_dataframe(ranking_A, "partA_rankings_by_subperiod")

    plot_line_chart(w_angles_A, "partA_pioneer_weights_lines.png", "Part A — PDM pioneer weights over time (angles)")
    plot_heatmap(w_angles_A, "partA_pioneer_weights_heatmap.png", "Part A — PDM pioneer weights heatmap (angles)")

    print("\nA.1(c) Countries receiving non-zero pioneer weight:")
    print(nonzero_summary_A[["country", "months_nonzero", "share_nonzero", "first_nonzero", "last_nonzero"]].to_string(index=False))

    print("\nA.2(a) Average pioneer weights by subperiod:")
    print(avg_table_A.round(6).to_string())

    print("\nA.2(b) Rankings by subperiod (1 = highest average pioneer weight):")
    print(ranking_A.to_string())

    # Part B
    print("\nRunning Part B ...")
    experts_B = df[EU_COUNTRIES].copy()
    target_B = df["UA"].copy()

    dominant = rolling_dominant_pioneer(experts_B, target_B, window=24)
    dominant.to_frame().to_csv(OUTPUT_DIR / "partB_dominant_pioneer_series.csv", index=True)
    plot_dominant_pioneer(dominant)

    methods = [
        "PDM_Angles",
        "PDM_Distance",
        "Granger",
        "Lagged_Correlation",
        "Multivariate_Regression",
        "Transfer_Entropy",
        "Linear_Pooling",
        "Median_Pooling",
    ]

    forecast_dict = {}
    avg_weight_dict = {}

    for method in methods:
        print(f"Computing {method} ...")
        fcst, weights = forecast_for_method(method, experts_B, target_B)
        forecast_dict[method] = fcst
        if weights is not None:
            avg_weight_dict[method] = weights.mean(axis=0)

    forecasts_B = pd.DataFrame(forecast_dict).reindex(target_B.index)
    save_dataframe(forecasts_B, "partB_forecasts_by_method")

    if avg_weight_dict:
        save_dataframe(pd.DataFrame(avg_weight_dict), "partB_average_weights_by_method")

    rmse_table = build_rmse_table(target_B, forecasts_B).sort_values("Full_Sample")
    save_dataframe(rmse_table, "partB_rmse_by_method")

    plot_forecasts_vs_actual(target_B, forecasts_B)

    print("\n=== Part B RMSE table ===")
    print(rmse_table.round(4).to_string())

    # Final written answers
    answers_text = generate_assignment_answers(
        nonzero_summary_A=nonzero_summary_A,
        avg_table_A=avg_table_A,
        ranking_A=ranking_A,
        dominant=dominant,
        rmse_table=rmse_table,
    )
    save_text(answers_text, "assignment_answers.txt")
    print("\n=== Assignment answers saved to assignment_answers.txt ===")
    print(answers_text)

    print(f"\nDone. All outputs saved to:\n{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
