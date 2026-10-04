"""Robustness of V9 before freezing (no new ideas, only perturbations). Configs fixed before the run.
Base: main board, non-ST, listed >= 1y, ADV >= 10M; drop worst 30% by comp; rank 0.5 small + 0.5 dividend
yield; 50 holdings; rebalance 20d; keep until rank > 150; crowding overlay whole book -5%/day.
In-sample 2020-04..2024-12.

  1. one-at-a-time: junk cut 0.2/0.3/0.4 | dy weight 0.3/0.5/0.7 | holdings 40/50/60 | rebalance 10/20/30 |
     ADV floor 5M/10M/20M
  2. rebalance-date luck: schedule shifted by 0/5/10/15 trading days (rebalance 20d)
  3. SW level-1 industry mix of the V9 book (average weight and the largest single-industry weight)
"""
import os, numpy as np, polars as pl
import lib, backtest as bt, signals, risk_rules as rr
from backtest import zrank
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, z["dates"], allf)
B = lib.build_base(end="2024-12-31", min_age_days=365); F = lib.build_factors(B)
pre = np.array([c[:3] for c in B.codes]); main = (~np.isin(pre, ["688", "689", "300", "301"]))[None, :]
dy_raw = signals.dividend_yield_ttm(B)
A, Z = "2020-04-01", "2024-12-31"
cache = {}


def parts(adv):
    if adv not in cache:
        U = lib.make_U(B, adv) & main
        st = rr.states(B, U)
        cache[adv] = dict(U=U, comp=zrank(signals.composite(B, F, signs, U=U), U), small=zrank(-F["lmoney20"], U),
                          dy=zrank(dy_raw, U), c2=rr.ramp(np.where(st["crowded"], 0.5, 1.0), 0.05, every=1))
    return cache[adv]


def run(cut=0.3, w=0.5, n=50, reb=20, adv=1e7, offset=0, weights=False):
    p = parts(adv)
    S = np.where(p["comp"] > 2 * cut - 1, (1 - w) * p["small"] + w * p["dy"], np.nan)
    res = bt.run(B, S, A, Z, n_hold=n, rebal=reb, buffer=3.0, U=p["U"], exposure=p["c2"], exposure_trade="rescale",
                 rebal_offset=offset, return_weights=weights)
    m = bt.metrics(res["ret"], res["bench"], res["turn_buy"])
    return res, m


base_res, base_m = run(weights=True)
print(f"BASE V9: ann {base_m['ann']:+.1%} mdd {base_m['mdd']:+.1%} sharpe {base_m['sharpe']:.2f}  (expect +32.0%)", flush=True)
rows = []
GRID = [("junk cut", "cut", (0.2, 0.3, 0.4)), ("dy weight", "w", (0.3, 0.5, 0.7)), ("holdings", "n", (40, 50, 60)),
        ("rebalance", "reb", (10, 20, 30)), ("ADV floor", "adv", (5e6, 1e7, 2e7))]
for label, key, vals in GRID:
    for v in vals:
        _, m = run(**{key: v})
        rows.append(dict(test=label, value=str(v if key != "adv" else f"{v/1e6:.0f}M"), ann=m["ann"], mdd=m["mdd"], sharpe=m["sharpe"], turn=m["turn_ann"]))
        print(f"  {label:10s} {rows[-1]['value']:>5s}: ann {m['ann']:+.1%} mdd {m['mdd']:+.1%} sharpe {m['sharpe']:.2f}", flush=True)
for off in (0, 5, 10, 15):
    _, m = run(offset=off)
    rows.append(dict(test="rebalance offset", value=str(off), ann=m["ann"], mdd=m["mdd"], sharpe=m["sharpe"], turn=m["turn_ann"]))
    print(f"  offset {off:>2d}d: ann {m['ann']:+.1%} mdd {m['mdd']:+.1%} sharpe {m['sharpe']:.2f}", flush=True)
pl.DataFrame(rows).write_csv("results/robust_v9_2020_2024.csv")

# industry mix (SW L1, membership usable from effective_from)
ii = pl.read_parquet(os.path.join(lib.DATA, "industry_intervals.parquet")).filter(pl.col("classification_system") == "sw_l1")
cidx = {c: i for i, c in enumerate(B.codes)}
names = sorted(ii["industry_name"].unique().to_list()); nid = {n: i for i, n in enumerate(names)}
T, N = len(B.dates), len(B.codes)
IND = np.full((T, N), -1)
for c, nm, a, b in zip(ii["code"].to_list(), ii["industry_name"].to_list(), ii["effective_from"].to_list(), ii["effective_to_exclusive"].to_list()):
    if c in cidx:
        ia = int(np.searchsorted(B.dates, np.datetime64(a, "D"))); ib = T if b is None else int(np.searchsorted(B.dates, np.datetime64(b, "D")))
        IND[ia:ib, cidx[c]] = nid[nm]
W = base_res["weights"]; t0 = int(np.searchsorted(B.dates, base_res["dates"][0]))
ind = IND[t0:t0 + len(W)]
gross = W.sum(1, keepdims=True)
share = np.zeros((len(W), len(names)))
for g in range(len(names)):
    share[:, g] = (W * (ind == g)).sum(1) / np.maximum(gross[:, 0], 1e-9)
avg = share.mean(0); mx = share.max(0)
order = np.argsort(-avg)[:8]
print("\nV9 book, SW L1 industry weights (average / max over time):")
print("   " + "  ".join(f"{names[g].rstrip('I')}: {avg[g]:.0%}/{mx[g]:.0%}" for g in order))
print(f"   largest single-industry weight on any day: {share.max():.0%};  days with any industry > 25%: {(share.max(1) > 0.25).mean():.0%}")
