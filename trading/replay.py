"""Replay a frozen version through the trading system day by day (paper broker) and compare it with the
research backtest (research/backtest.py run) on the same data, schedule and costs.

fractional=True  : shares may be fractional -> differences come only from accounting details (fees paid from
                   cash instead of pro rata, pre_close chain instead of rounded qfq prices)
fractional=False : whole board lots, minimum fees, cash never negative -> what a real account would have done
"""
import tempfile
import warnings
from pathlib import Path

import numpy as np

from .config import Account, Execution, strategy_from_spec
from .runner import AccountRun


def research_scores(market, kind):
    """The research score matrix (signals.mainboard_scores with the research universe) from the market's data."""
    R = market.R
    bt, sg = R.backtest, R.signals
    U = market.U_research
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        comp_r = bt.zrank(sg.composite(None, {k: market.F[k] for k in market.signs}, market.signs, U=U), U)
    small = bt.zrank(-market.F["lmoney20"], U)
    k30 = comp_r > -0.4
    if kind == "V0":
        return np.where(k30, small, np.nan)
    return np.where(k30, 0.5 * small + 0.5 * bt.zrank(market.dy, U), np.nan)


def metrics(ret, bench=None):
    ret = np.asarray(ret)
    nav = np.cumprod(1 + ret)
    n = len(ret)
    ann = nav[-1] ** (252 / n) - 1
    vol = ret.std(ddof=1) * np.sqrt(252)
    out = dict(ann=ann, vol=vol, sharpe=ret.mean() * 252 / vol, mdd=(nav / np.maximum.accumulate(nav) - 1).min(), days=n)
    if bench is not None:
        out["bench_ann"] = np.prod(1 + np.asarray(bench)) ** (252 / n) - 1
    return out


PERIODS = {"2020-04..2024-12": ("2020-04-01", "2024-12-31"), "2025-01..2026-09": ("2025-01-01", "2026-09-30")}


def by_period(dates, ret, bench):
    d = np.array(dates, dtype="datetime64[D]")
    out = {}
    for k, (a, b) in PERIODS.items():
        m = (d >= np.datetime64(a)) & (d <= np.datetime64(b))
        if m.sum() >= 20:
            out[k] = metrics(np.asarray(ret)[m], np.asarray(bench)[m])
    return out


def replay(market, cal, version, offset=0, start="2020-04-01", end=None, capital=5e5, fractional=False,
           out_dir=None, compare=True, log=print):
    strat = strategy_from_spec(version, anchor=start, offset=offset)
    tag = f"{version.split()[1] if len(version.split()) > 1 else version}_o{offset}_{'frac' if fractional else 'lots'}"
    acct = Account(name=f"replay_{tag}", broker="paper", capital=capital, start=start, strategy=strat,
                   execution=Execution(fractional=fractional))
    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="replay_"))
    path = out_dir / f"replay_{tag}.sqlite"
    path.unlink(missing_ok=True)
    run = AccountRun(acct, market, cal, log=log, fast=True, ledger_path=path)
    run.catch_up(until=end)
    days = run.ledger.days()
    sys_ret = np.array([d["ret"] for d in days])
    res = dict(version=version, offset=offset, fractional=fractional, system=metrics(sys_ret, [d["bench_ret"] for d in days]),
               ledger=str(path), dates=[d["date"] for d in days], sys_ret=sys_ret,
               trades=run.ledger.db.execute("SELECT count(*), sum(qty * price), sum(fee + stamp) FROM fills").fetchone())
    res["system_periods"] = by_period(res["dates"], sys_ret, [d["bench_ret"] for d in days])
    if compare:
        R = market.R
        sc = research_scores(market, strat.score)
        r = R.backtest.run(market.B, sc, start, end or str(market.dates[-1]), n_hold=strat.n_hold, rebal=strat.rebal,
                           buffer=strat.buffer, U=market.U_research, costs=R.backtest.Costs(capital=capital),
                           exposure=run.ex, exposure_trade="names", rebal_offset=offset, return_weights=True)
        assert [str(d) for d in r["dates"]] == res["dates"], "date mismatch between system and research replay"
        res["research"] = metrics(r["ret"], r["bench"])
        res["research_periods"] = by_period(res["dates"], r["ret"], r["bench"])
        res["research_ret"] = r["ret"]
        # same names held at each close?
        held_bt = r["weights"] > 1e-9
        same, jac = 0, []
        for i, d in enumerate(res["dates"]):
            sys_names = {c for c, in run.ledger.db.execute("SELECT code FROM positions WHERE date=?", (d,))}
            bt_names = {market.codes[j] for j in np.where(held_bt[i])[0]}
            same += sys_names == bt_names
            u = sys_names | bt_names
            jac.append(len(sys_names & bt_names) / len(u) if u else 1.0)
        res["same_names_days"] = same / len(res["dates"])
        res["mean_jaccard"] = float(np.mean(jac))
        res["ret_corr"] = float(np.corrcoef(sys_ret, r["ret"])[0, 1])
        res["max_abs_daily_diff"] = float(np.max(np.abs(sys_ret - r["ret"])))
    run.ledger.close()
    return res


def fmt(m):
    s = f"ann {m['ann']:+6.1%}  vol {m['vol']:5.1%}  sharpe {m['sharpe']:4.2f}  mdd {m['mdd']:6.1%}"
    if "bench_ann" in m:
        s += f"  | universe EW {m['bench_ann']:+6.1%}"
    return s
