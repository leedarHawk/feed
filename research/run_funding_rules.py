"""Funding-condition risk rules on top of version A (hand two-stage score, top 100, rebalance 20d,
buffer 3x, ADV>=10M). RULES AND PARAMETERS FIXED BEFORE THE FIRST RUN (committed before execution).
All signals use data up to the close of t only (trailing windows, no full-sample percentiles) and act
on the next open. 2019-2024 has already informed the research, so all results are in-sample.

  C0 none
  C1 universe-EW index < its 120d MA                                   -> half exposure
  C2 crowding: 20d mean of bottom-30%-cap turnover share > trailing 250d 90th pct -> half exposure
  C3 crowding (as C2)                                                  -> switch to the no-tilt score
  C4 stress: any day with >=100 limit-downs in the last 20 days        -> half exposure
  C5 C1 or C2                                                          -> half exposure
"""
import numpy as np, polars as pl
import lib, backtest as bt, signals, funding_proxies as fp
B = lib.build_base(end="2024-12-31"); F = lib.build_factors(B); U = lib.make_U(B, 1e7)
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, z["dates"], allf)
S_tilt = signals.tilted(B, F, signs, U, 1.0)
S_flat = signals.tilted(B, F, signs, U, 0.0)

fd = fp.build()
assert np.array_equal(fd["date"].to_numpy().astype("datetime64[D]"), B.dates), "date misalignment"
c = fd["small30_money_share"].rolling_mean(20).to_numpy()
T = len(c)
q90 = np.full(T, np.nan)
for t in range(250, T):
    w = c[t - 250:t]
    w = w[~np.isnan(w)]
    if len(w) >= 200:
        q90[t] = np.quantile(w, 0.9)
crowded = ~np.isnan(q90) & (c > q90)
ld = fd["n_limit_down"].to_numpy()
stress = np.array([(ld[max(0, t - 19):t + 1] >= 100).any() for t in range(T)])
mr = np.array([np.nanmean(B.r[t][U[t - 1] & ~np.isnan(B.r[t])]) if t > 0 and U[t - 1].any() else 0.0 for t in range(T)])
idx = np.cumprod(1 + np.nan_to_num(mr)); ma = lib.rmean(idx[:, None], 120)[:, 0]
below_ma = idx < ma

half = lambda flag: np.where(flag, 0.5, 1.0)
CONF = {
    "C0 none":               dict(S=S_tilt, exposure=None, regime=None),
    "C1 MA120->half":        dict(S=S_tilt, exposure=half(below_ma), regime=None),
    "C2 crowd->half":        dict(S=S_tilt, exposure=half(crowded), regime=None),
    "C3 crowd->no-tilt":     dict(S=np.where(crowded[:, None], S_flat, S_tilt), exposure=None, regime=crowded.astype(int)),
    "C4 stress->half":       dict(S=S_tilt, exposure=half(stress), regime=None),
    "C5 MA120|crowd->half":  dict(S=S_tilt, exposure=half(below_ma | crowded), regime=None),
}
A, Z = "2020-04-01", "2024-12-31"
a0 = int(np.searchsorted(B.dates, np.datetime64(A)))
print("share of days flagged 2020-04..2024-12:  below MA120 %.0f%%   crowded %.0f%%   stress %.0f%%" % (
    100 * below_ma[a0:].mean(), 100 * crowded[a0:].mean(), 100 * stress[a0:].mean()))
# crowding episodes (start..end of consecutive flagged runs)
ep, s0 = [], None
for t in range(a0, T):
    if crowded[t] and s0 is None: s0 = t
    if (not crowded[t] or t == T - 1) and s0 is not None:
        ep.append(f"{B.dates[s0]}..{B.dates[t - 1 if not crowded[t] else t]}({t - s0}d)"); s0 = None
print("crowding episodes:", ", ".join(ep))
rows = []
for name, cf in CONF.items():
    row = {"rule": name}
    for cl, cost in (("", bt.Costs()), ("_20bp", bt.Costs(slippage=0.002))):
        res = bt.run(B, cf["S"], A, Z, n_hold=100, rebal=20, buffer=3.0, U=U, costs=cost, exposure=cf["exposure"], regime=cf["regime"])
        m = bt.metrics(res["ret"], res["bench"], res["turn_buy"])
        row.update({f"ann{cl}": m["ann"], f"mdd{cl}": m["mdd"]})
        if cl == "":
            row.update(vol=m["vol"], sharpe=m["sharpe"], calmar=m["calmar"], turn=m["turn_ann"],
                       **{f"y{y}": a for y, (a, b) in bt.yearly(res).items()})
    rows.append(row)
R = pl.DataFrame(rows)
R.write_csv("results/funding_rules_2020_2024.csv")
pl.Config.set_tbl_rows(10); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(250)
print(R.with_columns([pl.col(c).round(3) for c in R.columns if c != "rule"]))
