"""Long-only daily backtest with A-share execution frictions.

Timeline for execution day t (signal known at close of t-1):
  1. positions earn the overnight return (prev close -> open t)
  2. on rebalance days trade to target at the open of t:
       * cannot BUY a name opening at limit-up / not trading
       * cannot SELL a name opening at limit-down / suspended
       * buys are scaled down if blocked sells leave too little cash
       * costs: commission both sides, stamp duty on sells, slippage both sides
  3. positions earn the intraday return (open t -> close t)
Prices are qfq, so ex-dividend/split gaps do not show up as losses.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Costs:
    commission: float = 0.00025          # each side
    slippage: float = 0.0010             # each side, vs open auction price
    stamp_old: float = 0.0010            # sells before 2023-08-28
    stamp_new: float = 0.0005            # sells from 2023-08-28
    stamp_change: str = "2023-08-28"

    def scaled(self, k):
        return Costs(self.commission * k, self.slippage * k, self.stamp_old, self.stamp_new, self.stamp_change)


def zrank(x, U):
    """Cross-sectional rank -> [-1, 1] inside the universe, NaN elsewhere (rows independent)."""
    from lib import rank_rows
    r = rank_rows(np.where(U, x, np.nan))
    n = np.sum(~np.isnan(r), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (2.0 * r / np.maximum(n - 1, 1) - 1.0).astype(np.float32)


def run(B, score, start, end, n_hold=50, rebal=5, buffer=1.5, costs=Costs(), U=None, return_weights=False,
        exposure=None, regime=None, exposure_trade="reselect"):
    """score[t] is known at the close of t. Returns dict with daily net returns, turnover, etc.
    exposure[t] (optional, known at the close of t) is the target gross weight for trades at t+1;
    a change in exposure forces a rebalance on the next open. A change in `regime[t]` (any label array,
    known at the close of t) also forces a rebalance, e.g. when the score switches between two models.
    exposure_trade: "reselect" re-runs name selection when exposure changes; "rescale" keeps the
    current names and scales every position to the new exposure (used for gradual de-risking); "names"
    holds round(exposure*n_hold) names at 1/n_hold each and, between rebalances, sells the worst-scored
    holdings or buys the best-ranked new names as whole positions (executable with board lots)."""
    U = B.U if U is None else U
    dates = B.dates
    t0 = int(np.searchsorted(dates, np.datetime64(start)))
    t1 = int(np.searchsorted(dates, np.datetime64(end), side="right"))
    T, N = score.shape
    stamp_day = np.searchsorted(dates, np.datetime64(costs.stamp_change))
    on = np.nan_to_num(B.on)
    idr = np.nan_to_num(B.idr)
    w = np.zeros(N)
    rets, tov_buy, tov_sell, n_pos, held_blocked = [], [], [], [], []
    wlog = [] if return_weights else None
    bench = []
    last_expo = None
    last_reg = None
    for t in range(max(t0, 1), t1):
        # 1. overnight
        g = float(w @ on[t])
        w = w * (1.0 + on[t]) / (1.0 + g)
        w[np.isnan(w)] = 0.0
        nav_mult = 1.0 + g
        sb = ss = 0.0
        # 2. rebalance
        expo = 1.0 if exposure is None else float(exposure[t - 1])
        reg = None if regime is None else regime[t - 1]
        scheduled = (t - max(t0, 1)) % rebal == 0
        expo_chg = last_expo is not None and expo != last_expo
        reg_chg = last_reg is not None and reg != last_reg
        target = None
        n_sel = int(round(expo * n_hold)) if exposure_trade == "names" else n_hold
        held = np.where(w > 1e-9)[0]
        if scheduled or reg_chg or (expo_chg and exposure_trade == "reselect"):
            last_expo = expo
            last_reg = reg
            s = score[t - 1]
            elig = U[t - 1] & ~np.isnan(s)
            idx = np.where(elig)[0]
            if len(idx) >= n_hold:
                order = idx[np.argsort(-s[idx], kind="stable")]
                rank = np.full(N, np.inf)
                rank[order] = np.arange(len(order))
                keep = np.where((w > 0) & (rank <= buffer * n_hold))[0]
                keep = keep[np.argsort(rank[keep])][:n_sel]
                chosen = list(keep)
                cs = set(chosen)
                for j in order:
                    if len(chosen) >= n_sel:
                        break
                    if j not in cs:
                        chosen.append(j)
                        cs.add(j)
                target = np.zeros(N)
                target[chosen] = (1.0 / n_hold) if exposure_trade == "names" else (expo / n_hold)
        elif exposure_trade == "names" and len(held) != n_sel:
            last_expo = expo
            s = score[t - 1]
            target = w.copy()
            if len(held) > n_sel:
                sc = np.where(np.isnan(s[held]), -np.inf, s[held])
                worst = held[np.argsort(sc, kind="stable")]
                worst = worst[B.can_sell_open[t][worst]]
                target[worst[:len(held) - n_sel]] = 0.0
            else:
                elig = U[t - 1] & ~np.isnan(s) & (w <= 1e-9) & B.can_buy_open[t]
                idx = np.where(elig)[0]
                best = idx[np.argsort(-s[idx], kind="stable")][:n_sel - len(held)]
                target[best] = 1.0 / n_hold
        elif expo_chg and exposure_trade == "rescale":
            # keep the same names, scale every position to the new gross exposure
            last_expo = expo
            tot = w.sum()
            if tot > 1e-12:
                target = w * (expo / tot)
        if target is not None:
            d = target - w
            buy = (d > 1e-12) & B.can_buy_open[t]
            sell = (d < -1e-12) & B.can_sell_open[t]
            dd = np.zeros(N)
            dd[sell] = d[sell]
            sell_tot = -dd.sum()
            cash = 1.0 - w.sum()
            avail = cash + sell_tot
            want = d[buy].sum()
            scale = min(1.0, avail / want) if want > 1e-12 else 0.0
            dd[buy] = d[buy] * scale
            w = np.maximum(w + dd, 0.0)
            sb, ss = dd[dd > 0].sum(), -dd[dd < 0].sum()
            stamp = costs.stamp_new if t >= stamp_day else costs.stamp_old
            cost = sb * (costs.commission + costs.slippage) + ss * (costs.commission + costs.slippage + stamp)
            nav_mult *= (1.0 - cost)
            w = w * (1.0 - cost)  # costs paid out of the portfolio
        # 3. intraday
        g2 = float(w @ idr[t])
        w = w * (1.0 + idr[t]) / (1.0 + g2)
        nav_mult *= (1.0 + g2)
        rets.append(nav_mult - 1.0)
        tov_buy.append(sb)
        tov_sell.append(ss)
        n_pos.append(int((w > 1e-9).sum()))
        if wlog is not None:
            wlog.append(w.copy())
        m = U[t - 1] & ~np.isnan(B.r[t])
        bench.append(float(np.nanmean(B.r[t][m])) if m.any() else 0.0)
        # renormalise weights after cost leakage so w stays a fraction of NAV
        tot = w.sum()
        if tot > 1.0:
            w = w / tot
    out = dict(dates=dates[max(t0, 1):t1], ret=np.array(rets), bench=np.array(bench),
               turn_buy=np.array(tov_buy), turn_sell=np.array(tov_sell), n_pos=np.array(n_pos))
    if wlog is not None:
        out["weights"] = np.array(wlog)
    return out


def metrics(ret, bench=None, turn=None, per=252):
    ret = np.asarray(ret)
    n = len(ret)
    if n == 0:
        return {}
    nav = np.cumprod(1 + ret)
    ann = nav[-1] ** (per / n) - 1
    vol = ret.std(ddof=1) * np.sqrt(per)
    dd = nav / np.maximum.accumulate(nav) - 1
    out = dict(ann=ann, vol=vol, sharpe=(ret.mean() * per) / vol if vol > 0 else np.nan, mdd=dd.min(),
               calmar=ann / abs(dd.min()) if dd.min() < 0 else np.nan, days=n)
    if bench is not None:
        b = np.asarray(bench)
        bn = np.cumprod(1 + b)
        out["bench_ann"] = bn[-1] ** (per / n) - 1
        ex = ret - b
        out["excess_ann"] = out["ann"] - out["bench_ann"]
        out["ir"] = ex.mean() * per / (ex.std(ddof=1) * np.sqrt(per)) if ex.std() > 0 else np.nan
    if turn is not None:
        out["turn_ann"] = float(np.sum(turn) / n * per)  # sum of buys per year (one-way, x NAV)
    return out


def by_period(res, periods):
    """periods: dict name -> (start, end) strings (inclusive)."""
    d = res["dates"]
    rows = {}
    for name, (a, b) in periods.items():
        m = (d >= np.datetime64(a)) & (d <= np.datetime64(b))
        if m.sum() < 20:
            continue
        rows[name] = metrics(res["ret"][m], res["bench"][m], res["turn_buy"][m])
    return rows


def yearly(res):
    d = res["dates"]
    yrs = d.astype("datetime64[Y]").astype(int) + 1970
    out = {}
    for y in sorted(set(yrs)):
        m = yrs == y
        out[int(y)] = (float(np.prod(1 + res["ret"][m]) - 1), float(np.prod(1 + res["bench"][m]) - 1))
    return out


def fmt(m):
    return (f"ann {m['ann']:+7.1%}  vol {m['vol']:5.1%}  sharpe {m['sharpe']:5.2f}  mdd {m['mdd']:6.1%}"
            f"  | bench {m.get('bench_ann', float('nan')):+6.1%}  excess {m.get('excess_ann', float('nan')):+6.1%}"
            f"  | buy-turn/yr {m.get('turn_ann', float('nan')):5.1f}x")
