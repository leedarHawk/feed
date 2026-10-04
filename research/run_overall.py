"""Whole-book ("整体") de-risking vs selling whole names, with realistic execution.
CONFIGS FIXED BEFORE THE FIRST RUN (committed before execution). In-sample 2020-04..2024-12.
Base: version A (hand score, top 100, rebalance 20d, buffer 3x). Risk rule: C1 (below 120d MA).

  execution : proportional costs only | capital 3M CNY | capital 10M CNY
              (capital runs: 5 CNY minimum commission per order; partial trades under one board lot skipped)
  configs   : C0 none
              C1 one-step rescale (reference)
              C1 whole book -5%/day (all holdings trimmed together, 1.0 -> 0.5 in 10 days; user's example)
              C1 whole book -2%/day
              C1 2 names/day (sell the worst-scored whole positions; reference)
"""
import numpy as np, polars as pl
import lib, backtest as bt, signals, risk_rules as rr
B = lib.build_base(end="2024-12-31"); F = lib.build_factors(B); U = lib.make_U(B, 1e7)
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
S = signals.tilted(B, F, signals.train_signs(ics, z["dates"], allf), U, 1.0)
st = rr.states(B, U)
T = len(B.dates)
tgt = np.where(st["below_ma"], 0.5, 1.0)
CONF = [("C0 none", None, "reselect"),
        ("C1 one-step", tgt, "rescale"),
        ("C1 整体 -5%/day", rr.ramp(tgt, 0.05, every=1), "rescale"),
        ("C1 整体 -2%/day", rr.ramp(tgt, 0.02, every=1), "rescale"),
        ("C1 2 names/day", rr.ramp(tgt, 0.02, every=1), "names")]
EXEC = {"proportional": bt.Costs(), "capital 3M": bt.Costs(capital=3e6), "capital 10M": bt.Costs(capital=1e7)}
A, Z = "2020-04-01", "2024-12-31"
rows = []
for name, ex, how in CONF:
    for en, c in EXEC.items():
        res = bt.run(B, S, A, Z, n_hold=100, rebal=20, buffer=3.0, U=U, costs=c, exposure=ex, exposure_trade=how)
        m = bt.metrics(res["ret"], res["bench"], res["turn_buy"])
        rows.append(dict(config=name, execution=en, ann=m["ann"], vol=m["vol"], sharpe=m["sharpe"], mdd=m["mdd"],
                         calmar=m["calmar"], turn=m["turn_ann"], **{f"y{y}": a for y, (a, b) in bt.yearly(res).items()}))
        print(f"  {name:28s} {en:12s} ann {m['ann']:+.1%} mdd {m['mdd']:+.1%} sharpe {m['sharpe']:.2f}", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/overall_derisk_2020_2024.csv")
pl.Config.set_tbl_rows(20); pl.Config.set_tbl_cols(16); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(240)
print(R.with_columns([pl.col(c).round(3) for c in R.columns if c not in ("config", "execution")]))
