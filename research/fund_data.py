"""Loader for data/fundamentals_v1_20261004 -> date x code matrices aligned to B.dates / B.codes.

Point-in-time rules (known at the close of t):
  * reports / forecasts / holder counts / unlock notices are usable from the first trading day AFTER
    pub_date (announcements often come after the close)
  * valuation_daily is lagged one trading day as well (its PIT alignment was only spot-checked)
Nothing with pub_date (or date) after LAST_DATE is read; 2025-2026 stays sealed.
"""
import os
import numpy as np, polars as pl
import lib

D = os.path.join(os.path.dirname(__file__), "..", "data", "fundamentals_v1_20261004")
LAST_DATE = "2024-12-31"


def _cut(df, col="pub_date"):
    if df[col].dtype != pl.Date:
        df = df.with_columns(pl.col(col).cast(pl.Date, strict=False))
    return df.filter(pl.col(col) <= pl.lit(LAST_DATE).str.to_date())


def _avail_idx(dates, d):
    """first trading-day index strictly after date d (np datetime64[D] array)."""
    return np.searchsorted(dates, d.astype("datetime64[D]"), side="right")


def _events_to_panel(dates, codes, df, value_cols, max_age):
    """Place each event's values at its availability day, keep the latest end_date per (code, day),
    forward-fill, and blank values older than max_age trading days."""
    cidx = {c: i for i, c in enumerate(codes)}
    df = df.filter(pl.col("code").is_in(list(cidx)))
    k = _avail_idx(dates, df["pub_date"].to_numpy())
    df = df.with_columns(pl.Series("k", k), pl.Series("j", [cidx[c] for c in df["code"].to_list()]))
    df = df.filter(pl.col("k") < len(dates))
    sort_cols = ["j", "k"] + (["end_date"] if "end_date" in df.columns else [])
    df = df.sort(sort_cols).unique(subset=["j", "k"], keep="last")
    T, N = len(dates), len(codes)
    kk, jj = df["k"].to_numpy(), df["j"].to_numpy()
    stamp = np.full((T, N), np.nan); stamp[kk, jj] = kk
    age = np.arange(T)[:, None] - lib.ffill(stamp)
    out = {}
    for c in value_cols:
        a = np.full((T, N), np.nan)
        a[kk, jj] = df[c].cast(pl.Float64).to_numpy()
        a = lib.ffill(a)
        a[~(age <= max_age)] = np.nan
        out[c] = a
    return out


def valuation(dates, codes, cols=("pe_ratio", "pb_ratio", "ps_ratio", "pcf_ratio")):
    y0, y1 = int(str(dates[0])[:4]), min(int(str(dates[-1])[:4]), int(LAST_DATE[:4]))
    df = pl.concat([pl.read_parquet(os.path.join(D, f"valuation_daily_{y}.parquet"), columns=["code", "date", *cols])
                    for y in range(y0, y1 + 1)])
    df = _cut(df, "date")
    cidx = {c: i for i, c in enumerate(codes)}
    df = df.filter(pl.col("code").is_in(list(cidx)))
    d = df["date"].to_numpy().astype("datetime64[D]")
    row = np.searchsorted(dates, d)
    ok = (row < len(dates)) & (dates[np.minimum(row, len(dates) - 1)] == d)
    col = np.array([cidx[c] for c in df["code"].to_list()])
    out = {}
    for c in cols:
        a = np.full((len(dates), len(codes)), np.nan)
        a[row[ok], col[ok]] = df[c].to_numpy()[ok]
        out[c] = lib.shift(a, 1)
    return out


def financials(dates, codes):
    df = _cut(pl.read_parquet(os.path.join(D, "financials_pit.parquet")))
    df = df.with_columns(
        (pl.col("np_parent_ttm") / pl.col("equities_parent_company_owners")).alias("roe_ttm"),
        ((pl.col("np_parent_ttm") - pl.col("net_operate_cash_flow_ttm")) / pl.col("total_assets")).alias("accruals"),
        (pl.col("net_operate_cash_flow_ttm") / pl.col("total_assets")).alias("cfo_assets"),
        (pl.col("good_will") / pl.col("equities_parent_company_owners")).alias("goodwill_eq"))
    cols = ["roe_ttm", "gross_margin_cum", "debt_ratio", "accruals", "cfo_assets", "goodwill_eq",
            "np_parent_yoy_cum", "revenue_yoy_cum"]
    out = _events_to_panel(dates, codes, df, cols, max_age=300)
    for k, v in out.items():
        v[~np.isfinite(v)] = np.nan
    return out


POS = {"业绩大幅上升", "业绩预增", "预计扭亏", "大幅减亏", "预计减亏", "业绩预盈"}
NEG = {"业绩预亏", "业绩大幅下降", "业绩预降"}


def forecasts(dates, codes, horizon=60):
    df = _cut(pl.read_parquet(os.path.join(D, "forecast.parquet"), columns=["code", "end_date", "pub_date", "type",
                                                                            "profit_ratio_min", "profit_ratio_max"]))
    df = df.with_columns(
        pl.when(pl.col("type").is_in(list(POS))).then(1.0).when(pl.col("type").is_in(list(NEG))).then(-1.0).otherwise(0.0).alias("fc_sign"),
        ((pl.col("profit_ratio_min") + pl.col("profit_ratio_max")) / 2).clip(-300, 300).alias("fc_mid"))
    out = _events_to_panel(dates, codes, df, ["fc_sign", "fc_mid"], max_age=horizon)
    out["fc_sign"] = np.where(np.isnan(out["fc_sign"]), 0.0, out["fc_sign"])     # no recent forecast -> neutral
    return out


def holders(dates, codes):
    df = _cut(pl.read_parquet(os.path.join(D, "holder_num.parquet"), columns=["code", "end_date", "pub_date", "share_holders"]))
    df = df.with_columns(pl.col("end_date").cast(pl.Date, strict=False)).sort(["code", "end_date", "pub_date"])
    df = df.unique(subset=["code", "end_date"], keep="first").sort(["code", "end_date"])
    df = df.with_columns((pl.col("share_holders") / pl.col("share_holders").shift(1).over("code") - 1).alias("holder_chg"))
    return _events_to_panel(dates, codes, df.drop_nulls("holder_chg"), ["holder_chg"], max_age=120)


def unlocks(dates, codes, window_days=90):
    df = _cut(pl.read_parquet(os.path.join(D, "unlimit_schedule.parquet"),
                              columns=["code", "pub_date", "expected_unlimited_date", "expected_unlimited_ratio"]))
    df = df.with_columns(pl.col("expected_unlimited_date").cast(pl.Date, strict=False)).drop_nulls(["expected_unlimited_date", "expected_unlimited_ratio"])
    cidx = {c: i for i, c in enumerate(codes)}
    df = df.filter(pl.col("code").is_in(list(cidx)))
    T, N = len(dates), len(codes)
    diff = np.zeros((T + 1, N))
    kp = _avail_idx(dates, df["pub_date"].to_numpy())
    ed = df["expected_unlimited_date"].to_numpy().astype("datetime64[D]")
    ks = np.searchsorted(dates, ed - np.timedelta64(window_days, "D"))       # first day within the window
    ke = np.searchsorted(dates, ed)                                         # unlock day (exclusive)
    a = np.maximum(kp, ks)
    for s, e, j, r in zip(a, ke, [cidx[c] for c in df["code"].to_list()], df["expected_unlimited_ratio"].to_numpy()):
        if s < e and s < T:
            diff[s, j] += r
            diff[min(e, T), j] -= r
    return {"unlock_next90": np.cumsum(diff, axis=0)[:T]}
