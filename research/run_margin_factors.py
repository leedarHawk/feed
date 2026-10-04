"""Stock-level margin-financing factors (new information beyond price/volume). FACTORS FIXED BEFORE THE
FIRST RUN (committed before execution). Discovery protocol as before: data cut at 2023-12-31,
TRAIN 2019-2021 (25-day embargo), VALID 2022-2023; universe ADV >= 10M. Margin data lagged one day.

  mfin_ratio   fin_value / circulating market value          (leverage crowding in the name)
  mfin_chg20   fin_value / fin_value 20 days earlier - 1      (fresh leverage flowing in)
  mbuy_ratio20 20d sum of margin buys / 20d sum of turnover   (share of trading done on margin)
  margin_elig  1 if the stock has margin data that day, else 0
Metrics: Rank IC (h=5, 20) with NW t-stats; best-decile excess raw and size+liquidity neutral.
Margin factors are evaluated inside margin-eligible names only (others have no value).
"""
import numpy as np, polars as pl
import lib, backtest as bt, signals, risk_data as rd
B = lib.build_base(); F = lib.build_factors(B); U = lib.make_U(B, 1e7)
T, N = U.shape
mg = rd.stock_margin(B.dates, B.codes)
fv, fb = mg["fin_value"], mg["fin_buy_value"]
circ = B.P["circulating_market_cap"] * 1e8
money_l = np.full_like(B.P["money"], np.nan); money_l[1:] = B.P["money"][:-1]       # same lag as margin
Fm = {"mfin_ratio": fv / circ,
      "mfin_chg20": fv / lib.shift(fv, 20) - 1,
      "mbuy_ratio20": lib.rsum(fb, 20, 14)[0] / lib.rsum(np.where(np.isnan(fb), np.nan, money_l), 20, 14)[0],
      "margin_elig": np.where(np.isnan(B.P["close"]), np.nan, (~np.isnan(fv)).astype(float))}
for k, v in Fm.items():
    v[~np.isfinite(v)] = np.nan
print("margin-eligible share of universe days: %.0f%%" % (100 * np.mean(~np.isnan(fv[U]))), flush=True)
s = int(np.searchsorted(B.dates, np.datetime64("2022-01-01")))
TR, VA = slice(60, s - 25), slice(s, None)
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
    return {k: float(np.nanmean(v[sl]) * a) for k, v, sl in (("top_tr", top, TR), ("top_va", top, VA), ("bot_tr", bot, TR), ("bot_va", bot, VA))}


rows = []
for name, f in Fm.items():
    x = np.where(U, f, np.nan)
    if name == "margin_elig":
        x = x + rng.random(x.shape) * 1e-6
    rk = lib.rank_rows(x)
    neu = signals.neutralize(x, U & ~np.isnan(x), ctrl)
    corr_size = float(np.nanmean(lib.row_corr(bt.zrank(x, U), bt.zrank(F["lmcap"], U))[0]))
    for h in (5, 20):
        ic = lib.row_corr(rk, fr[h])[0]
        raw, nd = deciles(x, h), deciles(neu, h)
        rows.append(dict(factor=name, h=h, corr_size=round(corr_size, 2),
                         ic_tr=float(np.nanmean(ic[TR])), t_tr=lib.nw_t(ic[TR], h), ic_va=float(np.nanmean(ic[VA])), t_va=lib.nw_t(ic[VA], h),
                         top_tr=raw["top_tr"], top_va=raw["top_va"], bot_tr=raw["bot_tr"], bot_va=raw["bot_va"],
                         neu_top_tr=nd["top_tr"], neu_top_va=nd["top_va"], neu_bot_tr=nd["bot_tr"], neu_bot_va=nd["bot_va"]))
    print(f"  {name} done", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/margin_factors_discovery.csv")
pl.Config.set_tbl_rows(10); pl.Config.set_tbl_cols(18); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(250)
print(R.with_columns([pl.col(c).round(3) for c in R.columns if c not in ("factor", "h")]))
