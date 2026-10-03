"""Daily-frequency research toolkit: base arrays, tradable universe, factor library, IC screen.

Conventions
-----------
* All matrices are [T dates, N codes].
* Signal is formed at the close of day t and executed at the OPEN of day t+1.
* Returns come from qfq (forward-adjusted) prices; qfq *levels* are anchored to the pull
  date and therefore never used as features, only ratios within a window.
* Discovery stage slices everything at DISCOVERY_END so 2024 and the sealed 2025-2026 data
  cannot leak into factor selection.
"""
import os
import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

import panel

DISCOVERY_END = "2023-12-31"
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "research_data_v1_20261002")
NAN = np.nan


# ----------------------------------------------------------------------------- array helpers
def ffill(x):
    T = x.shape[0]
    idx = np.where(~np.isnan(x), np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return np.take_along_axis(x, idx, axis=0)


def bfill(x):
    return ffill(x[::-1])[::-1]


def shift(x, k):
    out = np.full_like(x, np.nan)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


def rsum(x, w, minp=None):
    minp = w if minp is None else minp
    v = ~np.isnan(x)
    cs = np.cumsum(np.where(v, x, 0.0), axis=0)
    cn = np.cumsum(v, axis=0, dtype=np.int32)
    s, n = cs.copy(), cn.copy()
    s[w:] -= cs[:-w]
    n[w:] -= cn[:-w]
    s[n < minp] = np.nan
    return s, n


def rmean(x, w, minp=None):
    s, n = rsum(x, w, minp)
    return s / np.maximum(n, 1)


def rstd(x, w, minp=None):
    minp = w if minp is None else minp
    s, n = rsum(x, w, minp)
    s2, _ = rsum(x * x, w, minp)
    nn = np.maximum(n, 2)
    var = (s2 - s * s / nn) / (nn - 1)
    return np.sqrt(np.maximum(var, 0.0))


def _rwin(x, w, fn):
    out = np.full_like(x, np.nan)
    if x.shape[0] >= w:
        out[w - 1:] = fn.reduce(sliding_window_view(x, w, axis=0), axis=-1)
    return out


def rmax(x, w):
    return _rwin(x, w, np.fmax)


def rmin(x, w):
    return _rwin(x, w, np.fmin)


def rcorr(x, y, w, minp):
    m = ~np.isnan(x) & ~np.isnan(y)
    x = np.where(m, x, np.nan)
    y = np.where(m, y, np.nan)
    sx, n = rsum(x, w, minp)
    sy, _ = rsum(y, w, minp)
    sxx, _ = rsum(x * x, w, minp)
    syy, _ = rsum(y * y, w, minp)
    sxy, _ = rsum(x * y, w, minp)
    nn = np.maximum(n, 2)
    cov = sxy - sx * sy / nn
    vx = sxx - sx * sx / nn
    vy = syy - sy * sy / nn
    den = np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = cov / den
    out[den < 1e-12] = np.nan
    return out


def rank_rows(x):
    """Row-wise ranks 0..n-1 over non-NaN entries; NaN stays NaN. Ties broken by position."""
    N = x.shape[1]
    order = np.argsort(x, axis=1, kind="stable")
    ranks = np.empty(order.shape, dtype=np.float32)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(N, dtype=np.float32), order.shape), axis=1)
    ranks[np.isnan(x)] = np.nan
    return ranks


def row_corr(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    n = m.sum(1)
    a0 = np.where(m, a, 0.0)
    b0 = np.where(m, b, 0.0)
    nn = np.maximum(n, 1)
    da = np.where(m, a - (a0.sum(1) / nn)[:, None], 0.0)
    db = np.where(m, b - (b0.sum(1) / nn)[:, None], 0.0)
    den = np.sqrt((da * da).sum(1) * (db * db).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = (da * db).sum(1) / den
    ic[n < 50] = np.nan
    return ic, n


def nw_t(x, lag):
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(s / n)


# ----------------------------------------------------------------------------- base arrays
class Base:
    pass


def build_base(end=DISCOVERY_END, min_adv=3e7, min_age_days=180):
    dates, codes, P = panel.load()
    T = int((dates <= np.datetime64(end)).sum())
    dates = dates[:T]
    P = {k: v[:T] for k, v in P.items()}
    B = Base()
    B.dates, B.codes, B.P, B.end = dates, codes, P, end
    close, op = P["close"], P["open"]
    B.valid = ~np.isnan(close)
    qc = ffill(P["qfq_close"])
    B.qc, B.qo = qc, P["qfq_open"]
    prev = shift(qc, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(B.valid, qc / prev - 1.0, np.nan)
        B.on = np.where(B.valid, B.qo / prev - 1.0, np.nan)
        B.idr = np.where(B.valid, qc / B.qo - 1.0, np.nan)
    # IPO days (first 5 valid sessions, only for listings inside the panel) carry no price limit
    first = np.where(B.valid.any(0), B.valid.argmax(0), 0)
    for j in np.where(first > 0)[0]:
        idx = np.where(B.valid[:, j])[0][:5]
        r[idx, j] = np.nan
        B.on[idx, j] = np.nan
        B.idr[idx, j] = np.nan
    B.r = r

    # universe pieces known at the close of t ---------------------------------------------
    listing = pl.read_parquet(os.path.join(DATA, "listing.parquet"))
    ld = dict(zip(listing["code"].to_list(), listing["list_date"].to_list()))
    cls = dict(zip(listing["code"].to_list(), listing["share_class"].to_list()))
    list_np = np.array([np.datetime64(ld.get(c), "D") if ld.get(c) else np.datetime64("1990-01-01") for c in codes])
    age_days = (dates[:, None] - list_np[None, :]).astype("timedelta64[D]").astype(np.int64)
    B.age_ok = age_days >= min_age_days
    B.is_a = np.array([cls.get(c, "A") == "A" for c in codes])

    st = np.zeros((T, len(codes)), dtype=bool)
    sti = pl.read_parquet(os.path.join(DATA, "st_daily_intervals.parquet"))
    cidx = {c: i for i, c in enumerate(codes)}
    for c, s, e in zip(sti["code"].to_list(), sti["st_start_date"].to_list(), sti["st_end_date_exclusive"].to_list()):
        if c not in cidx:
            continue
        a = np.searchsorted(dates, np.datetime64(s, "D"))
        b = T if e is None else np.searchsorted(dates, np.datetime64(e, "D"))
        st[a:b, cidx[c]] = True
    B.st = st
    B.adv20 = rmean(P["money"], 20, 15)

    hl, ll = P["high_limit"], P["low_limit"]
    B.at_hl_open = np.isclose(op, hl, atol=0.005) | (op > hl)
    B.at_ll_open = np.isclose(op, ll, atol=0.005) | (op < ll)
    B.lim_up_close = B.valid & (np.isclose(close, hl, atol=0.005))
    B.lim_dn_close = B.valid & (np.isclose(close, ll, atol=0.005))
    B.can_buy_open = ~np.isnan(op) & ~B.at_hl_open
    B.can_sell_open = ~np.isnan(op) & ~B.at_ll_open

    # signal-day universe (decision at close t): not ST, aged, A-share, liquid, traded today,
    # and executable at next open (can buy at t+1 open)
    entry_ok = shift(B.can_buy_open.astype(np.float32), -1) == 1
    B.U = (B.valid & ~st & B.age_ok & B.is_a[None, :] & (B.adv20 >= min_adv) & entry_ok)
    B.U[-1] = False

    # executable forward returns: buy open t+1, sell first available open at/after t+1+h
    qo_b = bfill(B.qo)
    entry = shift(B.qo, -1)
    B.fwd = {}
    for h in (1, 5, 10, 20):
        ex = shift(qo_b, -(1 + h))
        with np.errstate(invalid="ignore", divide="ignore"):
            B.fwd[h] = np.where(np.isnan(entry) | np.isnan(ex), np.nan, ex / entry - 1.0)
    return B


# ----------------------------------------------------------------------------- factor library
def build_factors(B):
    P, r = B.P, B.r
    F = {}
    valid = B.valid.astype(np.float64)
    r0 = np.nan_to_num(r)
    lr0 = np.log1p(np.clip(r0, -0.95, None))
    cum = np.cumsum(lr0, axis=0)

    def cov(k):
        return rsum(valid, k, 1)[1] >= 0.7 * k

    for k in (1, 2, 3, 5, 10, 20, 40, 60, 120, 250):
        f = cum - shift(cum, k)
        f[~cov(k)] = np.nan
        f[:k] = np.nan
        F[f"ret{k}"] = f
    for k, s in ((120, 20), (250, 20), (60, 5), (20, 5)):
        f = shift(cum, s) - shift(cum, k)
        f[~cov(k)] = np.nan
        F[f"mom{k}_{s}"] = f

    mret = np.nanmean(r, axis=1)[:, None] * np.ones_like(r)
    for k in (5, 10, 20, 60, 120):
        F[f"vol{k}"] = rstd(r, k, max(3, int(0.7 * k)))
    for k in (20, 60):
        neg = np.minimum(r, 0.0)
        F[f"dvol{k}"] = np.sqrt(rmean(neg * neg, k, int(0.7 * k)))
        # idiosyncratic vol: residual sd after removing equal-weight market beta
        c = rcorr(r, mret, k, int(0.7 * k))
        F[f"ivol{k}"] = F[f"vol{k}"] * np.sqrt(np.maximum(1 - c * c, 0))
        F[f"beta{k}"] = c * F[f"vol{k}"] / np.maximum(rstd(mret, k, int(0.7 * k)), 1e-9)
    for k in (5, 20, 60):
        F[f"maxr{k}"] = rmax(r, k)
    for k in (20, 60):
        F[f"minr{k}"] = rmin(r, k)
        m1, n = rsum(r, k, int(0.7 * k))
        m2, _ = rsum(r * r, k, int(0.7 * k))
        m3, _ = rsum(r ** 3, k, int(0.7 * k))
        nn = np.maximum(n, 1)
        mu, var = m1 / nn, m2 / nn - (m1 / nn) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            sk = (m3 / nn - 3 * mu * m2 / nn + 2 * mu ** 3) / var ** 1.5
        sk[var < 1e-10] = np.nan
        F[f"skew{k}"] = sk
    prev_qc = shift(B.qc, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rng = (P["qfq_high"] - P["qfq_low"]) / prev_qc
    for k in (5, 20):
        F[f"range{k}"] = rmean(rng, k, int(0.7 * k))

    turn_d = P["volume"] / (P["circulating_cap"] * 1e4)
    for k in (5, 20, 60):
        F[f"lmoney{k}"] = np.log(np.maximum(rmean(P["money"], k, int(0.7 * k)), 1.0))
        F[f"turn{k}"] = rmean(turn_d, k, int(0.7 * k))
    F["turnr5_60"] = F["turn5"] / F["turn60"]
    F["turnr20_60"] = F["turn20"] / F["turn60"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ami = np.abs(r) / P["money"] * 1e8
    ami[P["money"] <= 0] = np.nan
    for k in (20, 60):
        F[f"amihud{k}"] = rmean(ami, k, int(0.7 * k))
    F["lmcap"] = np.log(P["market_cap"])
    F["lcmcap"] = np.log(P["circulating_market_cap"])
    F["lprice"] = np.log(P["close"])

    for k in (5, 20, 60):
        F[f"overnight{k}"] = rsum(B.on, k, int(0.7 * k))[0]
        F[f"intraday{k}"] = rsum(B.idr, k, int(0.7 * k))[0]
    hi, lo, cl = P["high"], P["low"], P["close"]
    with np.errstate(invalid="ignore", divide="ignore"):
        cpos = np.where(hi > lo, (cl - lo) / (hi - lo), np.nan)
        vwap = P["money"] / P["volume"]
        vdev = cl / vwap - 1.0
    for k in (5, 20):
        F[f"closepos{k}"] = rmean(cpos, k, int(0.7 * k))
        F[f"vwapdev{k}"] = rmean(vdev, k, int(0.7 * k))

    with np.errstate(invalid="ignore", divide="ignore"):
        dlv = np.log(P["volume"]) - np.log(shift(P["volume"], 1))
    dlv[~np.isfinite(dlv)] = np.nan
    F["corr_r_dvol20"] = rcorr(r, dlv, 20, 14)
    F["corr_r_turn20"] = rcorr(r, turn_d, 20, 14)
    sgn = np.sign(r) * P["money"]
    for k in (5, 20):
        F[f"moneyflow{k}"] = rsum(sgn, k, int(0.7 * k))[0] / rsum(P["money"], k, int(0.7 * k))[0]

    qc = B.qc
    F["dist_high250"] = qc / rmax(qc, 250) - 1
    F["dist_high60"] = qc / rmax(qc, 60) - 1
    F["dist_low60"] = qc / rmin(qc, 60) - 1
    hi250, lo250 = rmax(qc, 250), rmin(qc, 250)
    F["pos250"] = (qc - lo250) / np.maximum(hi250 - lo250, 1e-9)
    F["nlimup20"] = rsum(B.lim_up_close.astype(np.float64), 20, 1)[0]
    F["nlimdn20"] = rsum(B.lim_dn_close.astype(np.float64), 20, 1)[0]
    # roe/roa move around disclosure dates but the dataset carries no PIT guarantee -> extra 2d lag
    F["roe_l"] = shift(ffill(P["roe"]), 2)
    F["roa_l"] = shift(ffill(P["roa"]), 2)

    for k, v in F.items():
        v[~np.isfinite(v)] = np.nan
        F[k] = v.astype(np.float32)
    return F


# ----------------------------------------------------------------------------- IC screen
def screen(B, F, horizons=(1, 5, 10, 20)):
    """Daily Rank-IC series for every (factor, horizon), computed inside the signal universe."""
    fr = {h: rank_rows(np.where(B.U, B.fwd[h], np.nan)) for h in horizons}
    ics = {}
    rng = np.random.default_rng(0)
    for name, f in F.items():
        x = np.where(B.U, f, np.nan)
        if name.startswith("nlim"):  # heavy ties: random tie-break
            x = x + rng.random(x.shape, dtype=np.float32) * 1e-3
        rk = rank_rows(x)
        for h in horizons:
            ics[(name, h)] = row_corr(rk, fr[h])[0]
    return ics


def summarize(B, ics, split="2022-01-01", embargo=25, h_list=(1, 5, 10, 20)):
    d = B.dates
    s = int(np.searchsorted(d, np.datetime64(split)))
    tr = np.zeros(len(d), bool)
    tr[:max(s - embargo, 0)] = True
    va = np.zeros(len(d), bool)
    va[s:] = True
    rows = []
    for (name, h), ic in ics.items():
        a, b = ic[tr], ic[va]
        rows.append(dict(factor=name, h=h,
                         ic_tr=np.nanmean(a), t_tr=nw_t(a, h), ic_va=np.nanmean(b), t_va=nw_t(b, h),
                         n_tr=int((~np.isnan(a)).sum()), n_va=int((~np.isnan(b)).sum())))
    return pl.DataFrame(rows)
