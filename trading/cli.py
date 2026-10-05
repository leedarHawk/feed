"""Command line: python -m trading <command> ...  (see trading/README.md for the daily runbook)."""
import argparse
import datetime as dt
import json
import sys

from . import risk
from .brokers.base import Book
from .config import HOME, all_accounts, load_account
from .data import Store

log = print


def _market(store, tie_break="stable"):
    from .strategy import Market
    from .tradecal import Calendar
    fs = store.activate(log).final_spec
    m = Market(fs.COMMON["adv_floor"], fs.COMMON["min_age_days"], tie_break=tie_break, log=log)
    return m, Calendar(m.dates)


def _accounts(names):
    if names:
        return [load_account(n) for n in names]
    return [a for a in all_accounts() if a.enabled]


def _st_named(names):
    return {c for c, n in names.items() if n and ("ST" in n.upper() or "退" in n)}


# ---------------------------------------------------------------------------- data
def cmd_ingest(a):
    from .tradecal import Calendar
    store = Store(HOME)
    for f in a.files:
        dates, _, _ = store.activate(log).panel.load()
        day, warns = store.ingest_prices(f, Calendar(dates), log)
        for w in warns:
            log(f"  warning: {w}")


def cmd_ingest_ref(a):
    Store(HOME).ingest_reference(a.table, a.file)
    log(f"{a.table} replaced from {a.file}")


# ---------------------------------------------------------------------------- accounts
def cmd_import_holdings(a):
    from .brokers.manual import read_holdings
    from .ledger import Ledger
    from .runner import ensure_ledger
    acct = load_account(a.account)
    book = read_holdings(a.file, cash=a.cash)
    L = Ledger(acct.home / "ledger.sqlite")
    ensure_ledger(acct, L)
    old = Book.from_ledger(L)
    book.last_close = {c: old.last_close[c] for c in book.pos if c in old.last_close}
    book.save(L)
    day = a.date or str(Store(HOME).last_date())
    L.set("holdings_date", day)
    L.set("pending_flow", float(a.flow or 0.0))
    L.event(day, "info", "import", f"{len(book.pos)} holdings, cash {book.cash:,.0f} from {a.file}")
    if book.external:
        L.event(day, "warn", "external", "not part of the strategy, ignored: " +
                ", ".join(f"{c} x{q:g}" for c, q in book.external))
    L.commit()
    log(f"{acct.name}: {len(book.pos)} holdings and cash {book.cash:,.0f} recorded for {day}")


def cmd_run(a):
    """After the close (new data ingested): mark every account, then plan the next trading day."""
    from .brokers.manual import write_order_sheet, limit_price
    from .report import industry_map, render, temperature
    from .runner import AccountRun
    store = Store(HOME)
    market, cal = _market(store)
    T = len(market.dates)
    s, last_day = T - 1, str(market.dates[-1])
    next_day = str(cal.next_day(last_day))
    names = store.stock_names(last_day)
    st_named = _st_named(names)
    industries = industry_map(last_day)
    temp = None if a.no_temperature else temperature()
    for acct in _accounts(a.account):
        log(f"== {acct.name} ({acct.broker}, {acct.strategy.version})")
        ar = AccountRun(acct, market, cal, log=log)
        ar.names = names
        L = ar.ledger
        extra = []
        if acct.broker == "paper":
            n = ar.catch_up()
            log(f"   paper: {n} trading day(s) simulated")
        elif L.last_day() != last_day:
            if acct.broker == "qmt":
                from .brokers.qmt import QmtBroker
                live = QmtBroker(acct.execution, live=False, log=log).book()
                if live.external:
                    extra.append("QMT 账户里有策略以外的持仓，未计入：" + "，".join(f"{c}×{q:g}" for c, q in live.external))
                old = Book.from_ledger(L)
                live.last_close = old.last_close
                live.save(L)
                L.set("holdings_date", last_day)
            if L.get("holdings_date") is None:
                extra.append("尚未导入券商持仓和现金：先运行 import-holdings，本次不记净值")
            else:
                if L.get("holdings_date") != last_day:
                    extra.append(f"持仓未按 {last_day} 收盘对账（用 {L.get('holdings_date')} 导入的持仓估值）：请运行 import-holdings")
                ar.close_day(s, Book.from_ledger(L))
        book = Book.from_ledger(L)
        row = market.signal_row(s)
        dropped = [c for c in st_named if row.U_base[market.cidx[c]]] if st_named else []
        for c in st_named:
            if c in market.cidx:
                row.U_base[market.cidx[c]] = False
        if dropped:
            extra.append(f"名称带 ST/退 但 ST 区间表未覆盖，已排除：{', '.join(sorted(dropped))}（请更新 st_daily_intervals）")
        snap = acct.home / "snapshots"
        snap.mkdir(parents=True, exist_ok=True)
        row.save(snap / f"{last_day}.npz")
        scheduled = bool(market.scheduled(acct.strategy, T) or ar.first_day())
        meta = dict(signal_day=last_day, next_day=next_day, exposure=float(ar.ex[s]), scheduled=scheduled)
        (snap / f"{last_day}.json").write_text(json.dumps(meta))
        plan = ar.plan(T, book, ar.preliminary_info(row), row, "preliminary")
        for o in plan.orders:
            o.name = names.get(o.code, "")
            if acct.broker == "manual":
                o.limit = limit_price(o.side, o.price, acct.execution.limit_band)
        checks = risk.data_checks(market, s, list(book.pos), acct.risk)
        if next_day < dt.date.today().isoformat():
            checks.append(risk.Check("data_date", "warn", f"data ends {last_day} but {next_day} is already past: "
                                     "ingest the missing days before trading"))
        checks += risk.account_checks(L.days(), acct.risk, L.get("halted"))
        checks += [c for c in plan.checks if c.name != "halted"]
        L.db.execute("DELETE FROM orders WHERE date=? AND stage='preliminary'", (next_day,))
        L.add_orders(next_day, "preliminary", plan.orders, "planned")
        for w in cal.warnings:
            extra.append(w)
        L.commit()
        sheet = write_order_sheet(acct.home / "orders" / f"{next_day}_preliminary.csv", next_day, plan.orders,
                                  acct.execution.limit_band)
        md = render(acct, L, plan, checks, names, industries, next_day, temp, extra)
        rep = acct.home / "reports" / f"{last_day}.md"
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(md, encoding="utf-8")
        lvl = risk.worst(checks)
        log(f"   {plan.decision.kind}: {len(plan.orders)} orders for {next_day}, checks {lvl}; report {rep}; sheet {sheet}")
        L.close()


def cmd_open(a):
    """QMT accounts, 09:25:30 Beijing time: decide with the call-auction result and send the orders."""
    from .brokers.qmt import QmtBroker
    from .ledger import Ledger
    from .runner import make_plan
    from .strategy import SignalRow
    acct = load_account(a.account)
    if acct.broker != "qmt":
        sys.exit(f"{acct.name} is a {acct.broker} account; `open` is for QMT accounts")
    snaps = sorted((acct.home / "snapshots").glob("*.json"))
    if not snaps:
        sys.exit("no snapshot: run `python -m trading run` after the previous close")
    meta = json.loads(snaps[-1].read_text())
    today = a.day or dt.date.today().isoformat()
    if meta["next_day"] != today:
        sys.exit(f"latest snapshot plans {meta['next_day']}, not {today}: run `python -m trading run` first")
    row = SignalRow.load(snaps[-1].with_suffix(".npz"))
    L = Ledger(acct.home / "ledger.sqlite")
    broker = QmtBroker(acct.execution, live=a.live, log=log)
    book = broker.book()
    old = Book.from_ledger(L)
    book.last_close = old.last_close
    info = broker.auction_info(row.codes)
    scheduled = meta["scheduled"] or L.never_invested()
    plan = make_plan(acct, L, row, today, meta["exposure"], scheduled, book, info, "final")
    for c in plan.checks:
        if c.level != "ok":
            log(f"   {c.level} {c.name}: {c.message}")
            L.event(today, c.level, c.name, c.message)
    cidx = {c: i for i, c in enumerate(row.codes)}
    ids = broker.submit(plan.orders, info, cidx)
    oids = L.add_orders(today, "final", plan.orders, "sent" if broker.live else "dry-run")
    for oid, bid in zip(oids, ids):
        if bid:
            L.set_order_status(oid, "sent", bid)
    if broker.live and plan.orders:
        L.set("first_trade", L.get("first_trade") or today)
    L.commit()
    log(f"{acct.name}: {plan.decision.kind}, {len(plan.orders)} orders {'sent' if broker.live else '(dry run)'}")


def cmd_chase(a):
    from .brokers.qmt import QmtBroker
    acct = load_account(a.account)
    left = QmtBroker(acct.execution, live=a.live, log=log).chase(rounds=a.rounds)
    log(f"{left} order(s) still open")


def cmd_status(a):
    from .ledger import Ledger
    for acct in _accounts(a.account):
        L = Ledger(acct.home / "ledger.sqlite")
        days = L.days()
        log(f"== {acct.name} ({acct.broker}, {acct.strategy.version})")
        if days:
            nav0 = acct.capital or days[0]["nav"]
            d = days[-1]
            log(f"   {d['date']}: NAV {d['nav']:,.0f} ({d['nav'] / nav0 - 1:+.2%} since {days[0]['date']}), "
                f"drawdown {d['drawdown']:.2%}, cash {d['cash']:,.0f}, {d['n_pos']} positions")
        else:
            log("   no trading days yet")
        if L.get("halted"):
            log(f"   HALTED since {L.get('halted')}")
        for e in L.events()[-5:]:
            log(f"   {e[1]} {e[2]} {e[3]}: {e[4]}")
        L.close()


def cmd_resume(a):
    from .ledger import Ledger
    acct = load_account(a.account)
    L = Ledger(acct.home / "ledger.sqlite")
    L.event(dt.date.today().isoformat(), "warn", "resume", f"halt lifted: {a.note}")
    L.set("halted", None)
    L.commit()
    log(f"{acct.name}: halt lifted")


def cmd_replay(a):
    from . import replay as rp
    store = Store(HOME)
    market, cal = _market(store, tie_break="research")
    out = HOME / "replays"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for version in a.version:
        for o in a.offset:
            for frac in ([True, False] if a.both else [a.fractional]):
                r = rp.replay(market, cal, version, o, a.start, a.end, a.capital, frac, out_dir=out, compare=not a.no_compare)
                tag = f"{version} | offset {o} | {'fractional shares' if frac else 'board lots'}"
                n, val, cost = r["trades"]
                log(f"{tag}: {n} fills, traded {val or 0:,.0f}, fees+stamp {cost or 0:,.0f}")
                log(f"  system   whole run  {rp.fmt(r['system'])}")
                for k, m in r["system_periods"].items():
                    log(f"  system   {k:9s} {rp.fmt(m)}")
                if "research" in r:
                    log(f"  research whole run  {rp.fmt(r['research'])}")
                    for k, m in r["research_periods"].items():
                        log(f"  research {k:9s} {rp.fmt(m)}")
                    log(f"  same holdings at the close on {r['same_names_days']:.1%} of days (mean Jaccard "
                        f"{r['mean_jaccard']:.4f}); daily return corr {r['ret_corr']:.6f}; max daily gap "
                        f"{r['max_abs_daily_diff']:.3%}")
                row = dict(version=version, offset=o, shares="fractional" if frac else "lots")
                for src in ("system", "research"):
                    for k, m in [("all", r.get(src))] + list(r.get(f"{src}_periods", {}).items()):
                        if m:
                            for f in ("ann", "sharpe", "mdd", "bench_ann"):
                                row[f"{src}_{k}_{f}"] = round(float(m[f]), 5)
                row.update(same_names_days=r.get("same_names_days"), ret_corr=r.get("ret_corr"),
                           max_daily_gap=r.get("max_abs_daily_diff"))
                rows.append(row)
    import polars as pl
    path = out / "replay_summary.csv"
    pl.DataFrame(rows).write_csv(path)
    log(f"summary: {path}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m trading", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("ingest", help="add daily price files (same columns as data/stock_price_v1)")
    q.add_argument("files", nargs="+")
    q.set_defaults(fn=cmd_ingest)
    q = sub.add_parser("ingest-ref", help="replace a reference table (listing, st_daily_intervals, dividends, ...)")
    q.add_argument("table")
    q.add_argument("file")
    q.set_defaults(fn=cmd_ingest_ref)
    q = sub.add_parser("import-holdings", help="manual accounts: record the broker's holdings export after the close")
    q.add_argument("--account", required=True)
    q.add_argument("file")
    q.add_argument("--cash", type=float)
    q.add_argument("--date")
    q.add_argument("--flow", type=float, help="net deposit (+) or withdrawal (-) since the last close")
    q.set_defaults(fn=cmd_import_holdings)
    q = sub.add_parser("run", help="after the close: mark accounts, simulate paper days, plan the next day")
    q.add_argument("--account", action="append")
    q.add_argument("--no-temperature", action="store_true")
    q.set_defaults(fn=cmd_run)
    q = sub.add_parser("open", help="QMT accounts at 09:25:30: decide with the auction result and send orders")
    q.add_argument("--account", required=True)
    q.add_argument("--live", action="store_true")
    q.add_argument("--day")
    q.set_defaults(fn=cmd_open)
    q = sub.add_parser("qmt-chase", help="QMT accounts: re-price unfilled orders")
    q.add_argument("--account", required=True)
    q.add_argument("--live", action="store_true")
    q.add_argument("--rounds", type=int, default=3)
    q.set_defaults(fn=cmd_chase)
    q = sub.add_parser("status")
    q.add_argument("--account", action="append")
    q.set_defaults(fn=cmd_status)
    q = sub.add_parser("resume", help="lift a drawdown halt")
    q.add_argument("--account", required=True)
    q.add_argument("--note", required=True)
    q.set_defaults(fn=cmd_resume)
    q = sub.add_parser("replay", help="run frozen versions through the system and compare with the research backtest")
    q.add_argument("--version", action="append", required=True)
    q.add_argument("--offset", type=int, action="append")
    q.add_argument("--start", default="2020-04-01")
    q.add_argument("--end")
    q.add_argument("--capital", type=float, default=5e5)
    q.add_argument("--fractional", action="store_true")
    q.add_argument("--both", action="store_true", help="fractional and board-lot runs")
    q.add_argument("--no-compare", action="store_true")
    q.set_defaults(fn=cmd_replay)
    a = p.parse_args(argv)
    if getattr(a, "offset", "x") is None:
        a.offset = [0]
    a.fn(a)


if __name__ == "__main__":
    main()
