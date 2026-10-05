"""Pre-trade and account risk checks. A "block" stops the day's orders (or, for a halted account, its buys)
until a person looks; "warn" goes into the report."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Check:
    name: str
    level: str          # ok | warn | block
    message: str


def data_checks(market, s, held, limits, expected_day=None):
    out = []
    d = str(market.dates[s])
    if expected_day is not None and d != str(expected_day):
        out.append(Check("data_date", "block", f"latest data is {d}, expected {expected_day}"))
    v = market.B.valid
    if s > 0:
        cov = v[s].sum() / max(v[s - 1].sum(), 1)
        lvl = "ok" if cov >= limits.min_coverage else "block"
        out.append(Check("coverage", lvl, f"{int(v[s].sum())} names traded vs {int(v[s - 1].sum())} the day before"))
    nu = int(market.U_base[s].sum())
    out.append(Check("universe", "ok" if nu >= limits.min_universe else "block", f"{nu} names in the universe"))
    cidx = market.cidx
    susp = [c for c in held if not v[s, cidx[c]]]
    if susp:
        out.append(Check("suspended", "warn", f"held but not traded on {d}: {', '.join(susp)}"))
    return out


def account_checks(days, limits, halted):
    out = []
    if halted:
        out.append(Check("halted", "block", f"account halted since {halted}: buys are blocked until `resume`"))
    if days:
        last = days[-1]
        dd, r = last["drawdown"] or 0.0, last["ret"] or 0.0
        if dd <= limits.drawdown_halt:
            out.append(Check("drawdown", "block", f"drawdown {dd:.1%} at or beyond the halt line {limits.drawdown_halt:.0%}"))
        elif dd <= limits.drawdown_warn:
            out.append(Check("drawdown", "warn", f"drawdown {dd:.1%} (halt at {limits.drawdown_halt:.0%})"))
        if r <= limits.daily_loss_warn:
            out.append(Check("daily_loss", "warn", f"day return {r:.1%}"))
    return out


def order_checks(plan, book, limits, n_hold, lot, fractional, industry=None):
    out, orders, nav = [], plan.orders, plan.nav
    if not orders:
        return out
    if len(orders) > limits.max_orders:
        out.append(Check("n_orders", "block", f"{len(orders)} orders > {limits.max_orders}"))
    buys = sum(o.value for o in orders if o.side == "buy") / nav
    sells = sum(o.value for o in orders if o.side == "sell") / nav
    one_way = max(buys, sells)
    if plan.decision.kind == "rebalance":
        cap = limits.turnover_rebalance
    else:
        cap = abs(plan.decision.n_sel - len(book.pos)) / n_hold * 1.5 + 0.02
    lvl = "ok" if one_way <= cap + 1e-9 else "block"
    out.append(Check("turnover", lvl, f"buys {buys:.1%}, sells {sells:.1%} of NAV (limit {cap:.0%})"))
    bad = [o.code for o in orders if not (np.isfinite(o.price) and o.price > 0 and o.qty > 0)]
    if not fractional:
        bad += [o.code for o in orders if o.side == "buy" and abs(o.qty / lot - round(o.qty / lot)) > 1e-9]
    bad += [o.code for o in orders if o.side == "sell" and o.qty > book.pos.get(o.code, [0])[0] + 1e-6]
    if bad:
        out.append(Check("order_shape", "block", f"invalid quantity or price: {sorted(set(bad))}"))
    heavy = [o.code for o in orders if o.side == "buy" and o.target_weight > limits.max_name_weight]
    if heavy:
        out.append(Check("name_weight", "block", f"target weight above {limits.max_name_weight:.0%}: {heavy}"))
    spend = sum(o.value for o in orders if o.side == "buy")
    if spend > book.cash + sells * nav + 1e-6 and not fractional:
        out.append(Check("cash", "block", f"buys {spend:,.0f} exceed cash + sells {book.cash + sells * nav:,.0f}"))
    if industry:
        w = {}
        for o in orders:
            if o.side == "buy" or o.target_weight > 0:
                ind = industry.get(o.code, "未知")
                w[ind] = w.get(ind, 0.0) + o.target_weight
        if w:
            top, tw = max(w.items(), key=lambda kv: kv[1])
            if tw > limits.max_industry_weight:
                out.append(Check("industry", "warn", f"{top} {tw:.0%} of NAV among traded names"))
    return out


def worst(checks):
    lv = {"ok": 0, "warn": 1, "block": 2}
    return max((c.level for c in checks), key=lv.get, default="ok")
