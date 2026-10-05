"""Open-time portfolio decision and order sizing.

`decide` is research/backtest.py `run(..., exposure_trade="names")` for one day, applied to the account's
real holdings instead of simulated weights. `size_orders` turns the decision into share quantities with the
backtest's trading rules (no buy at a limit-up open, no sell at a limit-down open or while suspended, skip
partial trades smaller than one board lot, scale buys down when blocked sells leave too little cash) plus
what a real account needs: whole lots and enough cash for fees.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Decision:
    kind: str                       # rebalance | reduce | add | none | skip
    target: np.ndarray | None       # target weight per code (fraction of NAV at the open)
    n_sel: int                      # names the exposure calls for
    note: str = ""


@dataclass
class Order:
    code: str
    side: str                       # buy | sell
    qty: float
    price: float                    # reference price used for sizing (the open, or the close for a preliminary plan)
    reason: str
    name: str = ""
    limit: float | None = None
    target_weight: float = 0.0

    @property
    def value(self):
        return self.qty * self.price


@dataclass
class Plan:
    day: str
    decision: Decision
    orders: list = field(default_factory=list)
    nav: float = 0.0
    exposure: float = 1.0
    scheduled: bool = False
    notes: list = field(default_factory=list)

    def turnover(self):
        return sum(o.value for o in self.orders) / self.nav if self.nav > 0 else 0.0


def decide(w, score, can_sell, expo, n_hold, buffer, scheduled):
    """w: holdings as weights of NAV at the open; score: signal-day score, NaN outside the universe (the
    universe already requires the name to be buyable at this open)."""
    N = len(w)
    n_sel = int(round(expo * n_hold))
    held = np.where(w > 1e-9)[0]
    if scheduled:
        idx = np.where(~np.isnan(score))[0]
        if len(idx) < n_hold:
            return Decision("skip", None, n_sel, f"only {len(idx)} eligible names (< {n_hold})")
        order = idx[np.argsort(-score[idx], kind="stable")]
        rank = np.full(N, np.inf)
        rank[order] = np.arange(len(order))
        keep = np.where((w > 0) & (rank <= buffer * n_hold))[0]
        keep = keep[np.argsort(rank[keep])][:n_sel]
        chosen, cs = list(keep), set(keep.tolist())
        for j in order:
            if len(chosen) >= n_sel:
                break
            if j not in cs:
                chosen.append(j)
                cs.add(j)
        target = np.zeros(N)
        target[chosen] = 1.0 / n_hold
        return Decision("rebalance", target, n_sel, f"{len(keep)} kept, {len(chosen) - len(keep)} new")
    if len(held) == n_sel:
        return Decision("none", None, n_sel)
    target = w.copy()
    if len(held) > n_sel:
        sc = np.where(np.isnan(score[held]), -np.inf, score[held])
        worst = held[np.argsort(sc, kind="stable")]
        worst = worst[can_sell[worst]]
        target[worst[:len(held) - n_sel]] = 0.0
        return Decision("reduce", target, n_sel, f"{len(held)} held -> {n_sel}")
    idx = np.where(~np.isnan(score) & (w <= 1e-9))[0]
    best = idx[np.argsort(-score[idx], kind="stable")][:n_sel - len(held)]
    target[best] = 1.0 / n_hold
    return Decision("add", target, n_sel, f"{len(held)} held -> {n_sel}")


def _lots(q, lot):
    return np.floor(q / lot + 0.5) * lot


def size_orders(dec, w, shares, cash, nav, px, can_buy, can_sell, costs, day, codes, fractional=False,
                cash_buffer=0.0):
    """Orders (sells first) that move the holdings toward dec.target at prices px."""
    if dec.target is None:
        return []
    target = dec.target
    d = target - w
    lotv = costs.lot * np.where(np.isnan(px), np.inf, px)
    d = np.where((target > 1e-12) & (np.abs(d) * nav < lotv), 0.0, d)     # full exits are always allowed
    buy = (d > 1e-12) & can_buy & np.isfinite(px)
    sell = (d < -1e-12) & can_sell & np.isfinite(px)
    avail = cash / nav - d[sell].sum()
    want = d[buy].sum()
    scale = min(1.0, avail / want) if want > 1e-12 else 0.0
    stamp = costs.stamp(day)
    orders, proceeds = [], 0.0
    for j in np.where(sell)[0]:
        if target[j] <= 1e-12:
            q, why = shares[j], "exit"
        else:
            q = -d[j] * nav / px[j]
            q = q if fractional else min(_lots(q, costs.lot), shares[j])
            why = "trim"
        if q > 1e-9:
            v = q * px[j]
            proceeds += v * (1 - costs.slippage - stamp) - costs.fee(v)
            orders.append(Order(codes[j], "sell", float(q), float(px[j]), why, target_weight=float(target[j])))
    bj = np.where(buy)[0]
    want_v = d[bj] * scale * nav
    q = want_v / px[bj]
    if not fractional:
        q = _lots(q, costs.lot)
    budget = cash + proceeds - cash_buffer * nav

    def spend(q):
        v = q * px[bj]
        return float((v * (1 + costs.slippage)).sum() + sum(costs.fee(x) for x in v if x > 0))

    if fractional:
        for _ in range(4):                          # fees are not linear (minimum fee): a few fixed-point steps
            s = spend(q)
            if s <= budget + 1e-9 or s <= 0:
                break
            q = q * max(budget, 0.0) / s
    else:
        over = spend(q) - budget
        while over > 1e-9 and (q > 0).any():
            # take one lot from the buy left closest to its target afterwards (not always the dearest lot)
            k = int(np.argmin(np.where(q > 0, np.abs((q - costs.lot) * px[bj] - want_v), np.inf)))
            q[k] -= costs.lot
            over = spend(q) - budget
    for k, j in enumerate(bj):
        if q[k] > 1e-9:
            orders.append(Order(codes[j], "buy", float(q[k]), float(px[j]), "new" if w[j] <= 1e-9 else "top-up",
                                target_weight=float(target[j])))
    return orders


def weights_at(shares, cash, px, last_close):
    """Holdings as weights of NAV, valuing each name at px (the open) or its last close when px is missing."""
    p = np.where(np.isfinite(px), px, last_close)
    val = np.where(shares > 0, shares * np.nan_to_num(p), 0.0)
    nav = float(val.sum() + cash)
    return val / nav if nav > 0 else val, nav
