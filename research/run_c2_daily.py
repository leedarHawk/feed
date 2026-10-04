"""Small-cap crowding rule (C2) with daily whole-book de-risking, realistic execution.
CONFIGS FIXED BEFORE THE FIRST RUN (committed before execution). In-sample 2020-04..2024-12.
Base: version A (hand score, top 100, rebalance 20d, buffer 3x). Risk-off target 0.5.

  execution : proportional costs only | capital 3M CNY | capital 10M CNY (5 CNY min commission, board lots)
  configs   : C0 none
              C2 one-step | C2 0.1/week (reproduces the earlier table) | C2 whole book -5%/day
              C2 whole book -2%/day | C2 2 names/day
              C5 (MA120 or crowding) one-step | C5 whole book -5%/day
Caveat carried from earlier: the crowding flag does not predict small-minus-large returns outside the
2023-12/2024-01 episode; C2's edge rests mainly on early 2022 and early 2024.
"""
import numpy as np, polars as pl
import lib, backtest as bt, signals, risk_rules as rr
B = lib.build_base(end="2024-12-31"); F = lib.build_factors(B); U = lib.make_U(B, 1e7)
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
S = signals.tilted(B, F, signals.train_signs(ics, z["dates"], allf), U, 1.0)
st = rr.states(B, U)
t2 = np.where(st["crowded"], 0.5, 1.0)
t5 = np.where(st["crowded"] | st["below_ma"], 0.5, 1.0)
CONF = [("C0 none", None, "reselect"),
        ("C2 one-step", t2, "rescale"),
        ("C2 0.1/week", rr.ramp(t2, 0.1, every=5), "rescale"),
        ("C2 整体 -5%/day", rr.ramp(t2, 0.05, every=1), "rescale"),
        ("C2 整体 -2%/day", rr.ramp(t2, 0.02, every=1), "rescale"),
        ("C2 2 names/day", rr.ramp(t2, 0.02, every=1), "names"),
        ("C5 one-step", t5, "rescale"),
        ("C5 整体 -5%/day", rr.ramp(t5, 0.05, every=1), "rescale")]
EXEC = {"proportional": bt.Costs(), "capital 3M": bt.Costs(capital=3e6), "capital 10M": bt.Costs(capital=1e7)}
A, Z = "2020-04-01", "2024-12-31"


def dd_in(res, a, b):
    d = res["dates"]; m = (d >= np.datetime64(a)) & (d <= np.datetime64(b))
    nav = np.cumprod(1 + res["ret"][m]); return float((nav / np.maximum.accumulate(nav) - 1).min())


rows = []
for name, ex, how in CONF:
    for en, c in EXEC.items():
        res = bt.run(B, S, A, Z, n_hold=100, rebal=20, buffer=3.0, U=U, costs=c, exposure=ex, exposure_trade=how)
        m = bt.metrics(res["ret"], res["bench"], res["turn_buy"])
        r20 = bt.metrics(bt.run(B, S, A, Z, n_hold=100, rebal=20, buffer=3.0, U=U, costs=bt.Costs(slippage=0.002, capital=c.capital),
                                exposure=ex, exposure_trade=how)["ret"])["ann"] if en == "proportional" else np.nan
        rows.append(dict(config=name, execution=en, ann=m["ann"], ann_20bp=r20, sharpe=m["sharpe"], mdd=m["mdd"],
                         calmar=m["calmar"], turn=m["turn_ann"], dd_2024H1=dd_in(res, "2024-01-01", "2024-06-30"),
                         **{f"y{y}": a for y, (a, b) in bt.yearly(res).items()}))
        print(f"  {name:16s} {en:12s} ann {m['ann']:+.1%} mdd {m['mdd']:+.1%} sharpe {m['sharpe']:.2f}", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/c2_daily_2020_2024.csv")
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_cols(16); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(240)
print(R.with_columns([pl.col(c).round(3) for c in R.columns if c not in ("config", "execution")]))
