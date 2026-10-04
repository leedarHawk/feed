"""J1 at 500k CNY capital (user's size). 5 CNY minimum commission per order and board-lot minimum for
partial trades (fractional lots otherwise allowed - lot rounding of new positions is not modelled).
ADV floor 5M (capacity is not binding at this size). Each config is run on the 4 rebalance schedules
(offset 0/5/10/15) separately, because 4 staggered sleeves of 50 names are not feasible at 500k.
In-sample 2020-04..2024-12.

  n=50 whole book -5%/day (rescale)        | n=50 sell whole names, 0.04/day (2 names/day)
  n=30 whole book -5%/day (rescale)        | n=30 sell whole names, 0.05/day (~1.5 names/day)
  n=20 sell whole names, 0.05/day (1 name/day)
  reference: n=50 rescale with proportional costs only
"""
import numpy as np, polars as pl
import lib, backtest as bt, signals, risk_rules as rr
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
B = lib.build_base(end="2024-12-31", min_age_days=365); F = lib.build_factors(B)
pre = np.array([c[:3] for c in B.codes]); U = lib.make_U(B, 5e6) & (~np.isin(pre, ["688", "689", "300", "301"]))[None, :]
S = signals.mainboard_scores(B, F, signals.train_signs(ics, z["dates"], allf), U)["V9"]
crowd = np.where(rr.states(B, U)["crowded"], 0.5, 1.0)
d = B.dates; T = len(d); jan = np.ones(T)
for y in range(2019, 2025):
    i = int(np.searchsorted(d, np.datetime64(f"{y}-01-31"), side="right")) - 1
    if i >= 10 and str(d[i])[:7] == f"{y}-01":
        jan[i - 10:i] = 0.0
CAP = bt.Costs(capital=5e5)
CONF = [("n=50 整体每天降5% (比例成本参考)", 50, "rescale", 0.05, bt.Costs()),
        ("n=50 整体每天降5%", 50, "rescale", 0.05, CAP),
        ("n=50 每天整只卖2只", 50, "names", 0.04, CAP),
        ("n=30 整体每天降5%", 30, "rescale", 0.05, CAP),
        ("n=30 每天整只卖", 30, "names", 0.05, CAP),
        ("n=20 每天整只卖1只", 20, "names", 0.05, CAP)]
rows = []
for name, n, how, step, cost in CONF:
    ex = rr.ramp(crowd, step, every=1) * jan
    anns, mdds, shs, res_list = [], [], [], []
    for o in (0, 5, 10, 15):
        r = bt.run(B, S, "2020-04-01", "2024-12-31", n_hold=n, rebal=20, buffer=3.0, U=U, costs=cost, exposure=ex,
                   exposure_trade=how, rebal_offset=o, return_weights=(o == 0))
        m = bt.metrics(r["ret"], r["bench"], r["turn_buy"])
        anns.append(m["ann"]); mdds.append(m["mdd"]); shs.append(m["sharpe"]); res_list.append(r)
    W = res_list[0]["weights"]; t0 = int(np.searchsorted(d, res_list[0]["dates"][0])); held = W > 1e-9
    px = B.P["close"][t0:t0 + len(W)][held]
    pos_cny = W[held] * 5e5
    lot_share = float(np.median(100 * px / np.maximum(pos_cny, 1)))
    yrs = {y: float(np.mean([np.prod(1 + rr_["ret"][rr_["dates"].astype("datetime64[Y]").astype(int) + 1970 == y]) - 1 for rr_ in res_list])) for y in range(2020, 2025)}
    rows.append(dict(config=name, ann_mean=np.mean(anns), ann_min=min(anns), ann_max=max(anns), mdd_worst=min(mdds), sharpe_mean=np.mean(shs),
                     pos_cny_median=float(np.median(pos_cny)), one_lot_as_share_of_position=lot_share, **{f"y{y}": v for y, v in yrs.items()}))
    print(f"  {name:28s} ann mean {np.mean(anns):+.1%} (range {min(anns):+.1%}..{max(anns):+.1%}) worst mdd {min(mdds):+.1%} sharpe {np.mean(shs):.2f} | "
          f"position ~{np.median(pos_cny)/1e3:.0f}k CNY, 1 lot = {lot_share:.0%} of a position | " + " ".join(f"{y}:{v:+.0%}" for y, v in yrs.items()), flush=True)
pl.DataFrame(rows).write_csv("results/capital_500k_j1_2020_2024.csv")
