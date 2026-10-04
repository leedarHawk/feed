"""Staggered rebalancing: four equal sleeves of V9 / J1 whose 20-day schedules are offset by 0/5/10/15
trading days (sleeve returns averaged daily). Removes dependence on one rebalance date. In-sample."""
import numpy as np
import lib, backtest as bt, signals, risk_rules as rr
from backtest import zrank
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
B = lib.build_base(end="2024-12-31", min_age_days=365); F = lib.build_factors(B)
pre = np.array([c[:3] for c in B.codes]); U = lib.make_U(B, 1e7) & (~np.isin(pre, ["688", "689", "300", "301"]))[None, :]
S = signals.mainboard_scores(B, F, signals.train_signs(ics, z["dates"], allf), U)["V9"]
c2 = rr.ramp(np.where(rr.states(B, U)["crowded"], 0.5, 1.0), 0.05, every=1)
d = B.dates; T = len(d); jan = np.ones(T)
for y in range(2019, 2025):
    i = int(np.searchsorted(d, np.datetime64(f"{y}-01-31"), side="right")) - 1
    if i >= 10 and str(d[i])[:7] == f"{y}-01":
        jan[i - 10:i] = 0.0
for name, ex in (("V9", c2), ("J1 (V9 + Jan exit)", c2 * jan)):
    for cl, c in (("10bp", bt.Costs()), ("20bp", bt.Costs(slippage=0.002))):
        sleeves = [bt.run(B, S, "2020-04-01", "2024-12-31", n_hold=50, rebal=20, buffer=3.0, U=U, costs=c, exposure=ex,
                          exposure_trade="rescale", rebal_offset=o) for o in (0, 5, 10, 15)]
        anns = [bt.metrics(s["ret"])["ann"] for s in sleeves]
        r = np.mean([s["ret"] for s in sleeves], axis=0)
        m = bt.metrics(r)
        yrs = {}
        for y in range(2020, 2025):
            msk = sleeves[0]["dates"].astype("datetime64[Y]").astype(int) + 1970 == y
            yrs[y] = np.prod(1 + r[msk]) - 1
        print(f"{name:20s} {cl}: single sleeves {' '.join(f'{a:+.1%}' for a in anns)} | 4-sleeve book ann {m['ann']:+.1%} "
              f"mdd {m['mdd']:+.1%} sharpe {m['sharpe']:.2f} | " + " ".join(f"{y}:{v:+.0%}" for y, v in yrs.items()), flush=True)
