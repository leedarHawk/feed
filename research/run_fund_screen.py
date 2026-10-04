"""Fundamental, event and holder factors - discovery screen. FACTORS FIXED BEFORE THE FIRST RUN
(committed before execution). Same protocol as the price-volume screen: data cut 2023-12-31, TRAIN
2019-04..2021-11 (embargo), VALID 2022-2023, main-board universe (main board, non-ST, listed >= 1y,
ADV >= 10M). Reports: Rank IC (h=5, 20, NW t), best/worst decile excess, size+liquidity-neutral
best/worst decile excess, average correlation with size. Direction is read from TRAIN only.

  value   : ep=1/PE_ttm, bp=1/PB, sp=1/PS, cfp=1/PCF (valuation lagged 1d), dy_ttm
  quality : roe_ttm, gross_margin_cum, debt_ratio, accruals, cfo_assets, goodwill_eq
  growth  : np_parent_yoy_cum, revenue_yoy_cum
  events  : fc_sign (latest forecast within 60 trading days: +1 positive / -1 negative / 0), fc_mid
  holders : holder_chg (change in shareholder count, latest within 120 trading days)
  unlock  : unlock_next90 (expected unlock ratio, % of shares, in the next 90 calendar days)
"""
import time, numpy as np, polars as pl
import lib, backtest as bt, signals, fund_data as fd
t0 = time.time()
B = lib.build_base(min_age_days=365); F = lib.build_factors(B)
pre = np.array([c[:3] for c in B.codes])
U = lib.make_U(B, 1e7) & (~np.isin(pre, ["688", "689", "300", "301"]))[None, :]
G = {}
v = fd.valuation(B.dates, B.codes)
with np.errstate(divide="ignore", invalid="ignore"):
    G["ep"] = 1.0 / v["pe_ratio"]; G["bp"] = 1.0 / v["pb_ratio"]; G["sp"] = 1.0 / v["ps_ratio"]; G["cfp"] = 1.0 / v["pcf_ratio"]
G["dy_ttm"] = signals.dividend_yield_ttm(B)
G.update(fd.financials(B.dates, B.codes))
G.update(fd.forecasts(B.dates, B.codes))
G.update(fd.holders(B.dates, B.codes))
G.update(fd.unlocks(B.dates, B.codes))
for k in G:
    G[k] = np.where(np.isfinite(G[k]), G[k], np.nan)
print(f"factors built ({time.time()-t0:.0f}s); coverage inside U:",
      {k: f"{np.mean(~np.isnan(v_[U])):.0%}" for k, v_ in G.items()}, flush=True)
s = int(np.searchsorted(B.dates, np.datetime64("2022-01-01")))
TR = slice(int(np.searchsorted(B.dates, np.datetime64("2019-04-01"))), s - 25); VA = slice(s, None)
fr = {h: lib.rank_rows(np.where(U, B.fwd[h], np.nan)) for h in (5, 20)}
fw = {h: np.where(U, B.fwd[h], np.nan) for h in (5, 20)}
ctrl = [F["lmcap"], F["lmoney20"]]
rng = np.random.default_rng(0)


def deciles(x, h):
    rk = lib.rank_rows(np.where(U, x, np.nan)); n = np.sum(~np.isnan(rk), axis=1, keepdims=True)
    q = np.floor(rk / np.maximum(n, 1) * 10)
    mu = np.nanmean(np.where(~np.isnan(rk), fw[h], np.nan), axis=1)
    top = np.nanmean(np.where(q == 9, fw[h], np.nan), axis=1) - mu
    bot = np.nanmean(np.where(q == 0, fw[h], np.nan), axis=1) - mu
    a = 252 / h
    return {k: float(np.nanmean(v_[sl]) * a) for k, v_, sl in (("top_tr", top, TR), ("top_va", top, VA), ("bot_tr", bot, TR), ("bot_va", bot, VA))}


rows = []
for name, f in G.items():
    x = np.where(U, f, np.nan)
    x = x + rng.random(x.shape) * 1e-9 * np.nanstd(x)          # tie-break (fc_sign, unlock zeros)
    rk = lib.rank_rows(x)
    neu = signals.neutralize(x, U & ~np.isnan(x), ctrl)
    cs = float(np.nanmean(lib.row_corr(bt.zrank(x, U), bt.zrank(F["lmcap"], U))[0]))
    for h in (5, 20):
        ic = lib.row_corr(rk, fr[h])[0]
        raw, nd = deciles(x, h), deciles(neu, h)
        sgn = 1 if np.nanmean(ic[TR]) >= 0 else -1                 # own the high end if train IC > 0
        own = "high" if sgn > 0 else "low"
        side = "top" if sgn > 0 else "bot"
        rows.append(dict(factor=name, h=h, own=own, corr_size=round(cs, 2),
                         ic_tr=float(np.nanmean(ic[TR])), t_tr=lib.nw_t(ic[TR], h), ic_va=float(np.nanmean(ic[VA])), t_va=lib.nw_t(ic[VA], h),
                         own_tr=raw[f"{side}_tr"], own_va=raw[f"{side}_va"], neu_own_tr=nd[f"{side}_tr"], neu_own_va=nd[f"{side}_va"],
                         neu_ls_tr=(nd["top_tr"] - nd["bot_tr"]) * sgn, neu_ls_va=(nd["top_va"] - nd["bot_va"]) * sgn))
    print(f"  {name} ({time.time()-t0:.0f}s)", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/fund_screen_discovery.csv")
pl.Config.set_tbl_rows(50); pl.Config.set_tbl_cols(16); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(250)
for h in (20, 5):
    print(f"\n=== h={h} (own = side chosen by TRAIN IC; neu = size+liquidity neutral; decile excess annualised, no costs)")
    print(R.filter(pl.col("h") == h).sort("t_tr", descending=False, nulls_last=True)
          .select("factor", "own", "corr_size", *[pl.col(c).round(3) for c in ("ic_tr", "t_tr", "ic_va", "t_va", "own_tr", "own_va", "neu_own_tr", "neu_own_va", "neu_ls_tr", "neu_ls_va")]))
