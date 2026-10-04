"""SSE Composite (000001.XSHG) turning-point toolkit: zigzag labels, volume / MACD / RSI signals,
Fibonacci measurements, and signal-vs-turning-point evaluation.

Conventions
-----------
* Indicators use data up to and including the close of day t (causal). A signal fires at the close.
* Turning points come from a zigzag on closes: a top is confirmed once the close is `theta` below the
  running high since the last bottom (and vice versa). The labels are ex post and used only to grade
  signals; the confirmation day itself is causal and doubles as the plain filter-rule benchmark.
* Research sample 1997-01..2024-12 (10% price limits from 1996-12-16; earlier data are thin). The
  recent 2025-01..2026-09 window is graded separately as an out-of-sample check.
* Indicator settings are the textbook ones and are not tuned: MACD 12/26/9, Wilder RSI 6/14,
  Fibonacci 0.236/0.382/0.5/0.618/0.786.
"""
import os
import math
import numpy as np
import polars as pl

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sse_index_v1_20261004", "sse_daily.parquet")
IS_START, IS_END, OOS_START = "1997-01-01", "2024-12-31", "2025-01-01"
FIB = (0.236, 0.382, 0.5, 0.618, 0.786)
PLACEBO = (0.30, 0.44, 0.56, 0.70, 0.85)


class Series:
    pass


def load(end=None):
    df = pl.read_parquet(PATH)
    if end:
        df = df.filter(pl.col("date") <= pl.lit(end).str.to_date())
    s = Series()
    s.date = df["date"].to_numpy().astype("datetime64[D]")
    for c in ("open", "high", "low", "close", "volume"):
        setattr(s, c, df[c].to_numpy().astype(float))
    return s


def weekly(s):
    """Weekly bars (ISO week); each bar is stamped with the index of its last trading day."""
    wk = (s.date.astype("datetime64[D]") - np.datetime64("1970-01-05")).astype(int) // 7   # weeks starting Monday
    last = np.r_[np.where(np.diff(wk) != 0)[0], len(wk) - 1]
    first = np.r_[0, last[:-1] + 1]
    w = Series()
    w.idx = last
    w.close = s.close[last]
    w.high = np.array([s.high[a:b + 1].max() for a, b in zip(first, last)])
    w.low = np.array([s.low[a:b + 1].min() for a, b in zip(first, last)])
    w.volume = np.array([s.volume[a:b + 1].sum() for a, b in zip(first, last)])
    return w


# ----------------------------------------------------------------------------- indicators
def ema(x, alpha):
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1 - alpha) * out[t - 1]
    return out


def sma(x, n):
    c = np.cumsum(np.r_[0.0, x])
    out = np.full(len(x), np.nan)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def rmax(x, n):
    out = np.full(len(x), np.nan)
    for t in range(n - 1, len(x)):
        out[t] = x[t - n + 1:t + 1].max()
    return out


def rmin(x, n):
    return -rmax(-x, n)


def macd(close, fast=12, slow=26, sig=9):
    dif = ema(close, 2 / (fast + 1)) - ema(close, 2 / (slow + 1))
    dea = ema(dif, 2 / (sig + 1))
    return dif, dea, 2 * (dif - dea)


def rsi(close, n):
    """Wilder RSI, identical to 通达信 SMA(MAX(C-REF(C,1),0),N,1)/SMA(ABS(C-REF(C,1)),N,1)*100."""
    d = np.r_[0.0, np.diff(close)]
    g = ema(np.maximum(d, 0), 1 / n)
    l = ema(np.maximum(-d, 0), 1 / n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * g / (g + l)


def cross_up(a, b):
    a, b = np.broadcast_to(a, np.shape(a)), np.broadcast_to(b, np.shape(a))
    out = np.zeros(len(a), bool)
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def cross_dn(a, b):
    return cross_up(-np.asarray(a, float), -np.broadcast_to(b, np.shape(a)))


def macd_divergence(close, dif, dea):
    """Classic divergence at the cross. Between a death cross and the next golden cross lies a trough;
    at that golden cross, bottom divergence = trough close lower than the previous trough's AND trough
    DIF higher than the previous trough's, with DIF < 0. Top divergence mirrors it at death crosses."""
    gc, dc = cross_up(dif, dea), cross_dn(dif, dea)
    bot, top = np.zeros(len(close), bool), np.zeros(len(close), bool)
    last_dc = last_gc = None
    prev_trough = prev_peak = None
    for t in range(len(close)):
        if gc[t]:
            if last_dc is not None:
                seg = slice(last_dc, t + 1)
                tr = (close[seg].min(), dif[seg].min())
                if prev_trough is not None and tr[0] < prev_trough[0] and tr[1] > prev_trough[1] and dif[t] < 0:
                    bot[t] = True
                prev_trough = tr
            last_gc = t
        if dc[t]:
            if last_gc is not None:
                seg = slice(last_gc, t + 1)
                pk = (close[seg].max(), dif[seg].max())
                if prev_peak is not None and pk[0] > prev_peak[0] and pk[1] < prev_peak[1] and dif[t] > 0:
                    top[t] = True
                prev_peak = pk
            last_dc = t
    return bot, top


# ----------------------------------------------------------------------------- turning points
def zigzag(close, theta):
    """Alternating turning points on closes. Returns (idx, kind 'T'/'B', confirm_idx) per point;
    the last, still unconfirmed extreme is not included."""
    hi = lo = 0
    trend = 0
    pts = []
    for t in range(1, len(close)):
        c = close[t]
        if trend == 0:
            hi = t if c > close[hi] else hi
            lo = t if c < close[lo] else lo
            if c >= close[lo] * (1 + theta):
                pts.append((lo, "B", t)); trend, hi = 1, t
            elif c <= close[hi] * (1 - theta):
                pts.append((hi, "T", t)); trend, lo = -1, t
        elif trend == 1:
            if c > close[hi]:
                hi = t
            elif c <= close[hi] * (1 - theta):
                pts.append((hi, "T", t)); trend, lo = -1, t
        else:
            if c < close[lo]:
                lo = t
            elif c >= close[lo] * (1 + theta):
                pts.append((lo, "B", t)); trend, hi = 1, t
    return pts


def confirm_flags(pts, n):
    up, dn = np.zeros(n, bool), np.zeros(n, bool)
    for i, k, c in pts:
        (up if k == "B" else dn)[c] = True
    return up, dn


# ----------------------------------------------------------------------------- grading
def debounce(e, cool):
    """Keep a fire only if the same signal has not fired in the previous `cool` days."""
    out = np.zeros_like(e)
    last = -10 ** 9
    for t in np.where(e)[0]:
        if t - last > cool:
            out[t] = True
        last = t
    return out


def first_passage(close, up, dn, maxd=250):
    """+1 if close reaches c*(1+up) before c*(1-dn) within maxd days, -1 for the reverse, 0 neither, NaN if
    the data end first."""
    n = len(close)
    out = np.full(n, np.nan)
    for t in range(n):
        hi, lo = close[t] * (1 + up), close[t] * (1 - dn)
        seg = close[t + 1:t + 1 + maxd]
        a = np.where(seg >= hi)[0]
        b = np.where(seg <= lo)[0]
        ia = a[0] if len(a) else 10 ** 9
        ib = b[0] if len(b) else 10 ** 9
        if ia == ib == 10 ** 9:
            out[t] = 0.0 if t + maxd < n else np.nan
        else:
            out[t] = 1.0 if ia < ib else -1.0
    return out


def fwd_log(close, h):
    out = np.full(len(close), np.nan)
    out[:-h] = np.log(close[h:] / close[:-h])
    return out


def near(idx_events, idx_tps, w):
    """For each event: is some turning point within +-w days?  For each TP: first event in [tp-w, tp+w]."""
    tps = np.asarray(idx_tps)
    ev = np.asarray(idx_events)
    hit_ev = np.array([np.any(np.abs(tps - e) <= w) for e in ev], bool) if len(tps) else np.zeros(len(ev), bool)
    first = []
    for p in tps:
        c = ev[(ev >= p - w) & (ev <= p + w)]
        first.append(c[0] if len(c) else -1)
    return hit_ev, np.array(first)


def rand_p(x_all, mask_pool, n, obs, reps=20000, seed=0):
    """Share of random same-size draws (from days in mask_pool) whose mean is >= obs (one-sided)."""
    pool = x_all[mask_pool & ~np.isnan(x_all)]
    if n == 0 or len(pool) == 0 or not np.isfinite(obs):
        return np.nan
    rng = np.random.default_rng(seed)
    m = pool[rng.integers(0, len(pool), size=(reps, n))].mean(1)
    return float((m >= obs).mean())


def kde_mass(x, a, b, h):
    """Expected share of observations in [a, b] under a Gaussian KDE (bandwidth h) of x."""
    phi = lambda z: 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    return float(np.mean(phi((b - x) / h) - phi((a - x) / h)))
