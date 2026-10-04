"""Risk-state signals known at the close of t (trailing data only) and an exposure ramp."""
import numpy as np
import lib, funding_proxies as fp


def states(B, U):
    fd = fp.build()
    assert np.array_equal(fd["date"].to_numpy().astype("datetime64[D]"), B.dates), "date misalignment"
    T = len(B.dates)
    c = fd["small30_money_share"].rolling_mean(20).to_numpy()
    q90 = np.full(T, np.nan)
    for t in range(250, T):
        w = c[t - 250:t]
        w = w[~np.isnan(w)]
        if len(w) >= 200:
            q90[t] = np.quantile(w, 0.9)
    crowded = ~np.isnan(q90) & (c > q90)
    mr = np.array([np.nanmean(B.r[t][U[t - 1] & ~np.isnan(B.r[t])]) if t > 0 and U[t - 1].any() else 0.0 for t in range(T)])
    idx = np.cumprod(1 + np.nan_to_num(mr))
    ma = lib.rmean(idx[:, None], 120)[:, 0]
    below_ma = idx < ma
    return dict(below_ma=below_ma, crowded=crowded)


def ramp(target, step, every=5, start=1.0):
    """Move exposure toward `target` by at most `step`, only every `every` trading days."""
    e = np.empty(len(target))
    cur = start
    for t in range(len(target)):
        if t % every == 0:
            cur += float(np.clip(target[t] - cur, -step, step))
            cur = round(cur, 10)
        e[t] = cur
    return e
