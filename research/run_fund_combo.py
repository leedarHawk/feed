"""Adding fundamental/holder/forecast signals to the V9 ranking + January-only exit.
CONFIGS FIXED BEFORE THE FIRST RUN (committed before execution). Main-board universe, 50 holdings,
rebalance 20d, buffer 3x, crowding overlay whole book -5%/day. In-sample 2020-04..2024-12.

  V9  drop worst 30% by comp, rank by 0.5 small + 0.5 dividend yield              (current best)
  F1  ... rank by 0.5 small + 0.5 broad value (mean z of dy, ep, bp, sp)
  F2  ... rank by (small + dy + falling shareholder count) / 3
  F3  ... rank by (small + dy + forecast sign) / 3     (no recent forecast = neutral)
  J1  V9 + flat for the last 10 trading days before Jan 31 (annual loss pre-announcements), back on d+1
"""
import numpy as np, polars as pl
import lib, backtest as bt, signals, risk_rules as rr, fund_data as fd
from backtest import zrank
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, z["dates"], allf)
B = lib.build_base(end="2024-12-31", min_age_days=365); F = lib.build_factors(B)
pre = np.array([c[:3] for c in B.codes])
U = lib.make_U(B, 1e7) & (~np.isin(pre, ["688", "689", "300", "301"]))[None, :]
T, N = U.shape
comp_r = zrank(signals.composite(B, F, signs, U=U), U)
small = zrank(-F["lmoney20"], U)
dy = zrank(signals.dividend_yield_ttm(B), U)
v = fd.valuation(B.dates, B.codes)
with np.errstate(divide="ignore", invalid="ignore"):
    ep, bp, sp = (zrank(np.where(np.isfinite(1.0 / v[c]), 1.0 / v[c], np.nan), U) for c in ("pe_ratio", "pb_ratio", "ps_ratio"))
value_broad = np.nanmean(np.stack([dy, ep, bp, sp]), axis=0)
rng = np.random.default_rng(1)
hold = fd.holders(B.dates, B.codes)["holder_chg"]
holder = zrank(-(hold + rng.random(hold.shape) * 1e-9), U)
fcs = fd.forecasts(B.dates, B.codes)["fc_sign"]
fc = zrank(fcs + rng.random(fcs.shape) * 1e-6, U)
k30 = comp_r > -0.4
fill0 = lambda x: np.where(np.isnan(x), 0.0, x)               # missing component -> neutral
SC = {"V9 小盘+股息率": np.where(k30, 0.5 * small + 0.5 * dy, np.nan),
      "F1 小盘+宽价值": np.where(k30, 0.5 * small + 0.5 * value_broad, np.nan),
      "F2 小盘+股息率+户数减少": np.where(k30, (small + dy + fill0(holder)) / 3, np.nan),
      "F3 小盘+股息率+业绩预告": np.where(k30, (small + dy + fill0(fc)) / 3, np.nan)}
st = rr.states(B, U)
c2 = rr.ramp(np.where(st["crowded"], 0.5, 1.0), 0.05, every=1)
d = B.dates
jan = np.ones(T)
for y in range(2019, 2025):
    i = int(np.searchsorted(d, np.datetime64(f"{y}-01-31"), side="right")) - 1
    if i >= 10 and str(d[i])[:7] == f"{y}-01":
        jan[i - 10:i] = 0.0
CONF = [(k, s_, c2) for k, s_ in SC.items()] + [("J1 V9+1月清仓", SC["V9 小盘+股息率"], c2 * jan)]
A, Z = "2020-04-01", "2024-12-31"
rows = []
for name, S, ex in CONF:
    row = dict(config=name)
    for cl, c in (("", bt.Costs()), ("_20bp", bt.Costs(slippage=0.002)), ("_3M", bt.Costs(capital=3e6))):
        res = bt.run(B, S, A, Z, n_hold=50, rebal=20, buffer=3.0, U=U, costs=c, exposure=ex, exposure_trade="rescale",
                     return_weights=(cl == ""))
        m = bt.metrics(res["ret"], res["bench"], res["turn_buy"])
        row[f"ann{cl}"] = m["ann"]
        if cl == "":
            W = res["weights"]; t_ = int(np.searchsorted(d, res["dates"][0])); held = W > 1e-9
            row.update(sharpe=m["sharpe"], mdd=m["mdd"], calmar=m["calmar"], turn=m["turn_ann"],
                       mcap_med=float(np.nanmedian(B.P["market_cap"][t_:t_ + len(W)][held])),
                       **{f"y{y}": a for y, (a, _) in bt.yearly(res).items()})
    rows.append(row)
    print(f"  {name:24s} ann {row['ann']:+.1%} mdd {row['mdd']:+.1%} sharpe {row['sharpe']:.2f}", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/fund_combo_mainboard_2020_2024.csv")
pl.Config.set_tbl_rows(10); pl.Config.set_tbl_cols(16); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(240)
print(R.with_columns([pl.col(c).round(3) for c in R.columns if c != "config"]))
