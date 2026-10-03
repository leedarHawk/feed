"""Are there long-side factors beyond the small-cap / illiquidity premium?

For every factor, neutralise it cross-sectionally against size (lmcap) and liquidity (lmoney20)
each day, then measure the best decile's excess return over the universe. Adds two new families
built from PIT-safe supplementary data: TTM dividend yield (by implementation announcement date)
and SW level-1 industry momentum (by first-observed membership date).

Discovery data only (<= 2023-12-31). The direction of each factor is chosen on TRAIN only.
"""
import os, time
import numpy as np, polars as pl
import lib, backtest as bt

t0 = time.time()
B = lib.build_base(); F = lib.build_factors(B)
U = lib.make_U(B, 1e7)
d, codes = B.dates, B.codes
T, N = U.shape
cidx = {c: i for i, c in enumerate(codes)}
DATA = lib.DATA

# --- new factor 1: TTM dividend yield, known from the trading day AFTER the implementation notice
dv = pl.read_parquet(os.path.join(DATA, "dividends.parquet")).filter(
    (pl.col("status") == "实施方案") & pl.col("implementation_pub_date").is_not_null())
E = np.zeros((T, N))
for c, pub, cash in zip(dv["code"].to_list(), dv["implementation_pub_date"].to_list(), dv["cash_total_10k_cny"].to_list()):
    if c not in cidx or cash is None:
        continue
    k = int(np.searchsorted(d, np.datetime64(pub, "D"), side="right"))
    if k < T:
        E[k, cidx[c]] += cash
cs = np.cumsum(E, axis=0)
ttm = cs - lib.shift(cs, 243)
ttm[:243] = cs[:243]                     # table starts 2018-01, panel starts 2019-01: window is complete
F["dy_ttm"] = (ttm / 1e4 / B.P["market_cap"]).astype(np.float32)   # 万元 -> 亿元 / 亿元

# --- new factor 2: SW L1 industry momentum (membership usable from effective_from = first observation)
ii = pl.read_parquet(os.path.join(DATA, "industry_intervals.parquet")).filter(pl.col("classification_system") == "sw_l1")
names = sorted(ii["industry_name"].unique().to_list())
nid = {n: i for i, n in enumerate(names)}
IND = np.full((T, N), -1, dtype=np.int32)
for c, n, a, b in zip(ii["code"].to_list(), ii["industry_name"].to_list(), ii["effective_from"].to_list(), ii["effective_to_exclusive"].to_list()):
    if c not in cidx:
        continue
    ia = int(np.searchsorted(d, np.datetime64(a, "D")))
    ib = T if b is None else int(np.searchsorted(d, np.datetime64(b, "D")))
    IND[ia:ib, cidx[c]] = nid[n]
print(f"industry coverage inside U: {np.mean(IND[U] >= 0):.1%}", flush=True)
G = len(names)
for k in (20, 60, 120):
    rk = F[f"ret{k}"]
    out = np.full((T, N), np.nan, dtype=np.float32)
    for t in range(T):
        m = (IND[t] >= 0) & ~np.isnan(rk[t]) & B.valid[t]
        if m.sum() < 100:
            continue
        s = np.bincount(IND[t][m], weights=rk[t][m], minlength=G)
        n = np.bincount(IND[t][m], minlength=G)
        g = np.where(n >= 5, s / np.maximum(n, 1), np.nan)
        ok = IND[t] >= 0
        out[t, ok] = g[IND[t][ok]]
    F[f"indmom{k}"] = out
F["indrel20"] = (F["ret20"] - F["indmom20"]).astype(np.float32)
print(f"new factors built ({time.time()-t0:.0f}s)", flush=True)

# --- neutralisation against size & liquidity
Zs = bt.zrank(F["lmcap"], U); Zl = bt.zrank(F["lmoney20"], U)


def neutral(f):
    y = bt.zrank(f, U)
    m = U & ~np.isnan(y) & ~np.isnan(Zs) & ~np.isnan(Zl)
    X = np.stack([m.astype(np.float64), np.where(m, Zs, 0), np.where(m, Zl, 0)])   # [3, T, N]
    yy = np.where(m, y, 0.0)
    XtX = np.einsum("itn,jtn->tij", X, X) + np.eye(3)[None] * 1e-9
    Xty = np.einsum("itn,tn->ti", X, yy)
    beta = np.linalg.solve(XtX, Xty[..., None])[..., 0]
    r = y - np.einsum("ti,itn->tn", beta, X)
    r[~m] = np.nan
    return r


s = int(np.searchsorted(d, np.datetime64("2022-01-01")))
TR, VA = slice(60, s - 25), slice(s, None)
fw = {h: np.where(U, B.fwd[h], np.nan) for h in (5, 20)}
mu = {h: np.nanmean(fw[h], axis=1) for h in fw}


def deciles(x, h):
    rk = lib.rank_rows(np.where(U, x, np.nan)); n = np.sum(~np.isnan(rk), axis=1, keepdims=True)
    q = np.floor(rk / np.maximum(n, 1) * 10)
    top = np.nanmean(np.where(q == 9, fw[h], np.nan), axis=1) - mu[h]
    bot = np.nanmean(np.where(q == 0, fw[h], np.nan), axis=1) - mu[h]
    a = 252 / h
    return {"top_tr": np.nanmean(top[TR]) * a, "top_va": np.nanmean(top[VA]) * a,
            "bot_tr": np.nanmean(bot[TR]) * a, "bot_va": np.nanmean(bot[VA]) * a}


SIZE = {"lmcap", "lcmcap", "lmoney5", "lmoney20", "lmoney60", "amihud20", "amihud60"}
rows = []
for name, f in F.items():
    if name in SIZE:
        continue
    z = bt.zrank(f, U)
    cs_ = np.nanmean(lib.row_corr(z, Zs)[0]); cl_ = np.nanmean(lib.row_corr(z, Zl)[0])
    r = neutral(f)
    for h in (5, 20):
        raw, neu = deciles(f, h), deciles(r, h)
        sign = 1 if raw["top_tr"] >= raw["bot_tr"] else -1          # direction chosen on TRAIN only
        pick = (lambda D, side: D[f"{side}_tr"], lambda D, side: D[f"{side}_va"])
        side = "top" if sign == 1 else "bot"
        rows.append(dict(factor=name, h=h, own="high" if sign == 1 else "low",
                         corr_size=round(float(cs_), 2), corr_liq=round(float(cl_), 2),
                         raw_tr=raw[f"{side}_tr"], raw_va=raw[f"{side}_va"],
                         neu_tr=neu[f"{side}_tr"], neu_va=neu[f"{side}_va"],
                         neu_ls_tr=(neu["top_tr"] - neu["bot_tr"]) * sign, neu_ls_va=(neu["top_va"] - neu["bot_va"]) * sign))
    print(f"  {name} ({time.time()-t0:.0f}s)", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/nonsize_screen_discovery.csv")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(12); pl.Config.set_fmt_float("mixed")
for h in (20, 5):
    r = R.filter(pl.col("h") == h).with_columns(pl.min_horizontal("neu_tr", "neu_va").alias("worst"))
    print(f"\n=== h={h}: best-decile excess AFTER size+liquidity neutralisation (annualised, no costs), top 20 by min(train, valid) ===")
    print(r.sort("worst", descending=True).head(20).select(
        "factor", "own", "corr_size", "corr_liq", *[pl.col(c).round(3) for c in ("raw_tr", "raw_va", "neu_tr", "neu_va", "neu_ls_tr", "neu_ls_va")]))
print("done", flush=True)
