"""Market-level funding / liquidity proxies computable from daily price-volume data alone.
All values at date t use only data up to the close of t."""
import numpy as np, polars as pl
import panel

def build(start_year=2019):
    dates, codes, P = panel.load()
    v = ~np.isnan(P["close"])
    money = np.where(v, P["money"], 0.0)
    mc = np.where(v, P["market_cap"], np.nan)
    r = np.where(v, P["close"] / P["pre_close"] - 1, np.nan)
    hl, ll = P["high_limit"], P["low_limit"]
    normal = (hl / P["pre_close"] < 1.25)                       # skip no-limit days (IPO / delisting)
    up_lim = v & normal & np.isclose(P["close"], hl, atol=0.005)
    dn_lim = v & normal & np.isclose(P["close"], ll, atol=0.005)
    # size buckets by daily market-cap rank among traded names
    rk = np.argsort(np.argsort(np.where(v, mc, np.inf), axis=1), axis=1).astype(float)
    n = v.sum(1, keepdims=True)
    pct = np.where(v, rk / np.maximum(n - 1, 1), np.nan)
    small = v & (pct < 0.3); large = v & (pct >= 0.9)
    tot = money.sum(1)
    flow = np.nansum(np.sign(np.nan_to_num(r)) * money, axis=1) / tot
    circ_val = np.nansum(np.where(v, P["circulating_market_cap"], 0.0), axis=1) * 1e8
    df = pl.DataFrame({
        "date": dates.astype("datetime64[ms]"),
        "turnover_bn": tot / 1e8,                                # 全市场成交额, 亿元
        "turnover_rate": tot / circ_val,                         # 成交额 / 流通市值
        "small30_money_share": money[:].__mul__(small).sum(1) / tot,   # 最小 30% 市值股票的成交额占比
        "top10_money_share": money.__mul__(large).sum(1) / tot,        # 最大 10% 的占比
        "small_minus_large_ret": np.nanmean(np.where(small, r, np.nan), 1) - np.nanmean(np.where(large, r, np.nan), 1),
        "pct_up": np.nanmean(np.where(v, r > 0, np.nan), 1),
        "n_limit_up": up_lim.sum(1), "n_limit_down": dn_lim.sum(1),
        "signed_money_flow": flow,                               # Σ sign(r)·成交额 / Σ成交额
    })
    return df.filter(pl.col("date").dt.year() >= start_year)
