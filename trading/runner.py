"""Daily orchestration: open-time plans, paper execution, end-of-day marking, preliminary plans."""
import numpy as np

from . import risk
from .brokers.base import Book
from .brokers.paper import PaperBroker
from .engine import Decision, Plan, decide, size_orders, weights_at
from .ledger import Ledger


def ensure_ledger(acct, L):
    """Write the account's metadata once. Cash starts at the capital for paper accounts; live accounts get
    holdings and cash only from the broker (import-holdings / QMT), which this never overwrites."""
    if L.meta("account") is None:
        L.set_meta("account", acct.name)
        L.set_meta("broker", acct.broker)
        L.set_meta("strategy", acct.strategy.__dict__)
        L.set_meta("capital", acct.capital)
        L.set_meta("start", acct.start)
    if L.get("cash") is None:
        L.set("cash", float(acct.capital) if acct.broker == "paper" else 0.0)
    L.commit()


def make_plan(acct, ledger, row, day, expo, scheduled, book, info, stage):
    """Decision, orders and checks for one trade day from a signal row and open-time arrays.
    info: observed after the call auction (stage "final") or assumed at the close (stage "preliminary")."""
    s = acct.strategy
    cidx = {c: i for i, c in enumerate(row.codes)}
    n = len(row.codes)
    score, U = row.score(info["can_buy"], s.score)
    shares = book.shares_array(cidx, n)
    held_close = book.close_array(cidx, n)
    w, nav = weights_at(shares, book.cash, info["open"], np.where(np.isnan(held_close), row.close, held_close))
    if nav <= 0:
        plan = Plan(day=day, decision=Decision("skip", None, 0, "no capital: import holdings and cash first"),
                    nav=0.0, exposure=expo, scheduled=scheduled)
        plan.universe, plan.score, plan.checks = int(U.sum()), score, []
        return plan
    dec = decide(w, score, info["can_sell"], expo, s.n_hold, s.buffer, scheduled)
    orders = size_orders(dec, w, shares, book.cash, nav, info["open"], info["can_buy"], info["can_sell"],
                         acct.costs, day, row.codes, fractional=acct.execution.fractional,
                         cash_buffer=acct.execution.cash_buffer)
    plan = Plan(day=day, decision=dec, orders=orders, nav=nav, exposure=expo, scheduled=scheduled)
    plan.universe = int(U.sum())
    plan.score = score
    halted = ledger.get("halted")
    checks = risk.order_checks(plan, book, acct.risk, s.n_hold, acct.costs.lot, acct.execution.fractional)
    checks += risk.account_checks([], acct.risk, halted)
    if halted:
        plan.orders = [o for o in plan.orders if o.side == "sell"]
        plan.notes.append("halted: buys removed")
    blocked = [c for c in checks if c.level == "block" and c.name != "halted"]
    if blocked and stage == "final":
        plan.notes.append("blocked: " + "; ".join(c.message for c in blocked))
        plan.orders = []
    plan.checks = checks
    return plan


class AccountRun:
    """One account bound to the shared market data for a session."""

    def __init__(self, acct, market, cal, log=print, fast=False, ledger_path=None):
        self.acct, self.market, self.cal, self.log = acct, market, cal, log
        self.s = acct.strategy
        self.ledger = Ledger(ledger_path or acct.home / "ledger.sqlite", fast=fast)
        ensure_ledger(acct, self.ledger)
        self.ex = market.exposure(self.s, cal)

    # ------------------------------------------------------------------ planning
    def first_day(self):
        return self.ledger.never_invested()

    def plan(self, t, book, info, row, stage):
        """Plan for trade-day index t (len(dates) = the day after the panel)."""
        m = self.market
        day = str(m.dates[t]) if t < len(m.dates) else str(self.cal.next_day(row.day))
        scheduled = m.scheduled(self.s, t) or self.first_day()
        return make_plan(self.acct, self.ledger, row, day, float(self.ex[t - 1]), scheduled, book, info, stage)

    def preliminary_info(self, row):
        """Open-time arrays assumed at the close: every name that traded today is buyable, every held name
        sellable, at today's close."""
        n = len(row.codes)
        return dict(open=np.where(row.valid, row.close, np.nan), can_buy=row.valid.copy(), can_sell=np.ones(n, bool),
                    pre_close=row.close)

    # ------------------------------------------------------------------ paper trading
    def paper_day(self, t, broker):
        """One historical trading day of a paper account: corporate actions, open-time plan and fills, close."""
        m, L = self.market, self.ledger
        day = str(m.dates[t])
        book = Book.from_ledger(L)
        info = m.open_info(t)
        for n in broker.corporate_actions(book, m.cidx, info["pre_close"]):
            L.event(day, "info", "corporate_action", n)
        row = m.signal_row(t - 1)
        plan = self.plan(t, book, info, row, "final")
        names = getattr(self, "names", {})
        for o in plan.orders:
            o.name = names.get(o.code, "")
        fills, status = broker.execute(book, plan.orders, day, m.cidx, info)
        ids = L.add_orders(day, "final", plan.orders, "new")
        for oid, st in zip(ids, status):
            L.set_order_status(oid, st)
        for f in fills:
            f["order_id"] = ids[f.pop("order")]
            L.add_fill(f)
        for c in plan.checks:
            if c.level != "ok":
                L.event(day, c.level, c.name, c.message)
        if fills:
            L.set("first_trade", L.get("first_trade") or day)
        traded = sum(abs(f["qty"] * f["price"]) for f in fills)
        self.close_day(t, book, traded, plan)
        return plan, fills

    def close_day(self, t, book, traded=0.0, plan=None):
        m, L = self.market, self.ledger
        day = str(m.dates[t])
        close = m.B.P["close"][t]
        mv, nav = book.mark({c: close[m.cidx[c]] for c in book.pos})
        if nav <= 0:
            L.event(day, "warn", "no_capital", "nothing to mark: import holdings and cash")
            L.commit()
            return None
        prev = L.last_row()
        base = prev["nav"] if prev else (self.acct.capital or nav)
        flow = float(L.get("pending_flow", 0.0) or 0.0)            # deposits (+) / withdrawals (-) since the last close
        base += flow
        peak = max((prev["peak"] + flow) if prev else base, nav)
        row = dict(date=day, nav=nav, cash=book.cash, mv=mv, ret=nav / base - 1 if base else 0.0,
                   bench_ret=m.bench_return(t), exposure=float(self.ex[t - 1]), n_pos=len(book.pos), peak=peak,
                   drawdown=nav / peak - 1, turnover=traded / nav if nav else 0.0,
                   note=(plan.decision.kind if plan else ""))
        positions = [(c, s, book.last_close.get(c, k), s * book.last_close.get(c, k), k) for c, (s, k) in book.pos.items()]
        L.record_day(row, positions)
        L.set("pending_flow", 0.0)
        book.save(L)
        if row["drawdown"] <= self.acct.risk.drawdown_halt and not L.get("halted"):
            L.set("halted", day)
            L.event(day, "block", "halt", f"drawdown {row['drawdown']:.1%}: buys blocked until `resume`")
        L.commit()
        return row

    def catch_up(self, until=None):
        """Paper account: process every trading day from the account start (or the last processed day) on."""
        m, L = self.market, self.ledger
        broker = PaperBroker(self.acct.costs)
        last = L.last_day()
        t = int(np.searchsorted(m.dates, np.datetime64(last), side="right")) if last else \
            max(int(np.searchsorted(m.dates, np.datetime64(self.acct.start))), 1)
        t1 = len(m.dates) if until is None else int(np.searchsorted(m.dates, np.datetime64(until), side="right"))
        n = 0
        for tt in range(t, t1):
            self.paper_day(tt, broker)
            n += 1
        return n
