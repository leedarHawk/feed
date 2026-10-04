"""Loader for data/smallcap_risk_v1_20261004, cut at LAST_DATE (2025-2026 stays sealed).

Timing (what is known at the close of date t):
  * index closes, futures closes/basis: same day
  * margin (SH/SZ totals and per stock) and ETF shares: SZSE publishes one trading day late, so all of
    them are lagged by one trading day before use
"""
import os, glob
import numpy as np, polars as pl

D = os.path.join(os.path.dirname(__file__), "..", "data", "smallcap_risk_v1_20261004")
LAST_DATE = "2024-12-31"


def _read(name, cols=None):
    df = pl.read_parquet(os.path.join(D, f"{name}.parquet"), columns=cols)
    if df["date"].dtype != pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Date))
    return df.filter(pl.col("date") <= pl.lit(LAST_DATE).str.to_date())


def _align(df, dates, col):
    """Series on the trading calendar `dates` (np datetime64[D]); missing -> NaN."""
    m = dict(zip(df["date"].to_numpy().astype("datetime64[D]").tolist(), df[col].to_list()))
    return np.array([np.nan if m.get(d) is None else float(m[d]) for d in dates.tolist()])


def lag1(x):
    out = np.full_like(x, np.nan, dtype=float)
    out[1:] = x[:-1]
    return out


def market_series(dates):
    out = {}
    bc = _read("futures_basis_by_contract")
    for p in ("IC", "IM", "IF"):
        s = bc.filter((pl.col("product") == p) & (pl.col("expiry_rank") == 2))
        out[f"{p}_basis_ann_r2"] = _align(s, dates, "basis_annualized_pct")
    ix = _read("index_daily")
    for code, k in (("000852.XSHG", "csi1000"), ("399303.XSHE", "gz2000"), ("000300.XSHG", "hs300"), ("000905.XSHG", "csi500")):
        out[f"{k}_close"] = _align(ix.filter(pl.col("index_code") == code), dates, "close")
    mt = _read("margin_total_combined_daily")
    out["margin_fin_value"] = lag1(_align(mt, dates, "fin_value"))
    out["margin_fin_buy"] = lag1(_align(mt, dates, "fin_buy_value")) if "fin_buy_value" in mt.columns else np.full(len(dates), np.nan)
    ef = _read("etf_family_share_daily").filter(pl.col("complete") == 1)
    fam_col = "family" if "family" in ef.columns else [c for c in ef.columns if c not in ("date", "n_etfs", "n_expected", "complete")][0]
    share_col = "shares" if "shares" in ef.columns else [c for c in ef.columns if "share" in c][0]
    for fam in ef[fam_col].unique().to_list():
        out[f"etf_shares_{fam}"] = lag1(_align(ef.filter(pl.col(fam_col) == fam), dates, share_col))
    return out


def stock_margin(dates, codes):
    """Per-stock fin_value / fin_buy_value matrices [T, N], lagged one trading day. NaN = not a margin name."""
    files = [f for f in sorted(glob.glob(os.path.join(D, "margin_stock_daily_*.parquet")))
             if int(os.path.basename(f).split("_")[-1].split(".")[0]) <= int(LAST_DATE[:4])]
    df = pl.concat([pl.read_parquet(f, columns=["date", "code", "fin_value", "fin_buy_value"]) for f in files])
    df = df.with_columns(pl.col("date").cast(pl.Date)).filter(pl.col("date") <= pl.lit(LAST_DATE).str.to_date())
    cidx = {c: i for i, c in enumerate(codes)}
    df = df.filter(pl.col("code").is_in(list(cidx)))
    row = np.searchsorted(dates, df["date"].to_numpy().astype("datetime64[D]"))
    ok = (row < len(dates)) & (dates[np.minimum(row, len(dates) - 1)] == df["date"].to_numpy().astype("datetime64[D]"))
    col = np.array([cidx[c] for c in df["code"].to_list()])
    out = {}
    for k in ("fin_value", "fin_buy_value"):
        a = np.full((len(dates), len(codes)), np.nan)
        a[row[ok], col[ok]] = df[k].cast(pl.Float64).to_numpy()[ok]
        lagged = np.full_like(a, np.nan)
        lagged[1:] = a[:-1]
        out[k] = lagged
    return out
