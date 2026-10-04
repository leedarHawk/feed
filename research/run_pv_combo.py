"""Let the price-volume signals rank stocks, not only filter them. Main-board universe (main board,
non-ST, listed >= 365d, ADV >= 10M). Configs fixed before the first run. In-sample 2020-04..2024-12.
Version A settings (top 100, rebalance 20d, keep until rank > 300); overlays whole book -5%/day.

  comp  = family composite (lottery, turnover, limit-up, intraday, reversal; train-IC signs)
  small = low 20d turnover value;  vm = mean(low vwapdev20, low maxr60);  ml = walk-forward LightGBM
  V0 drop worst 30% by comp, rank by small                         (current)
  V1 drop worst 30%, rank by 0.5 small + 0.5 comp
  V2 drop worst 30%, rank by 0.5 small + 0.5 vm
  V3 drop worst 30%, rank by (small + comp + vm) / 3
  V4 drop worst 50% by comp, rank by small
  V5 drop worst 30%, rank by 0.5 small + 0.5 ml
  overlays: C0 none | C2 crowding
"""
import numpy as np, polars as pl
import lib, backtest as bt, signals, risk_rules as rr
from backtest import zrank
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, z["dates"], allf)
B = lib.build_base(end="2024-12-31", min_age_days=365); F = lib.build_factors(B)
pre = np.array([c[:3] for c in B.codes])
U = lib.make_U(B, 1e7) & (~np.isin(pre, ["688", "689", "300", "301"]))[None, :]
comp_r = zrank(signals.composite(B, F, signs, U=U), U)
small = zrank(-F["lmoney20"], U)
vm = np.nanmean(np.stack([zrank(-F["vwapdev20"], U), zrank(-F["maxr60"], U)]), axis=0)
pred = np.load("/tmp/feed_cache/ml_pred_2020_2024.npy")
assert pred.shape == U.shape
ml = zrank(pred, U)
k30, k50 = comp_r > -0.4, comp_r > 0.0
SC = {"V0 现行: 剔30%+小盘": np.where(k30, small, np.nan),
      "V1 剔30%+小盘/综合": np.where(k30, 0.5 * small + 0.5 * comp_r, np.nan),
      "V2 剔30%+小盘/VWAP+MAX": np.where(k30, 0.5 * small + 0.5 * vm, np.nan),
      "V3 剔30%+三者等权": np.where(k30, (small + comp_r + vm) / 3, np.nan),
      "V4 剔50%+小盘": np.where(k50, small, np.nan),
      "V5 剔30%+小盘/LightGBM": np.where(k30, 0.5 * small + 0.5 * ml, np.nan)}
st = rr.states(B, U)
OV = {"C0": None, "C2": rr.ramp(np.where(st["crowded"], 0.5, 1.0), 0.05, every=1)}
A, Z = "2020-04-01", "2024-12-31"
rows = []
for sname, S in SC.items():
    for oname, ex in OV.items():
        row = dict(score=sname, overlay=oname)
        for cl, c in (("", bt.Costs()), ("_20bp", bt.Costs(slippage=0.002))):
            res = bt.run(B, S, A, Z, n_hold=100, rebal=20, buffer=3.0, U=U, costs=c, exposure=ex,
                         exposure_trade="reselect" if ex is None else "rescale", return_weights=(cl == ""))
            m = bt.metrics(res["ret"], res["bench"], res["turn_buy"])
            row[f"ann{cl}"] = m["ann"]
            if cl == "":
                W = res["weights"]; t_ = int(np.searchsorted(B.dates, res["dates"][0])); held = W > 1e-9
                row.update(sharpe=m["sharpe"], mdd=m["mdd"], calmar=m["calmar"], turn=m["turn_ann"],
                           mcap_med=float(np.nanmedian(B.P["market_cap"][t_:t_ + len(W)][held])),
                           adv_med_M=float(np.nanmedian(B.adv20[t_:t_ + len(W)][held]) / 1e6),
                           **{f"y{y}": a for y, (a, _) in bt.yearly(res).items()})
        rows.append(row)
        print(f"  {sname:26s} {oname} ann {row['ann']:+.1%} mdd {row['mdd']:+.1%} sharpe {row['sharpe']:.2f}", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/pv_combo_mainboard_2020_2024.csv")
pl.Config.set_tbl_rows(14); pl.Config.set_tbl_cols(18); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(250)
print(R.with_columns([pl.col(c).round(3) for c in R.columns if c not in ("score", "overlay")]))
