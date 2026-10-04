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
  V6 drop worst 30%, rank by 0.5 small + 0.5 oversold (low RSI14)          (user request: RSI)
  V7 current V0, additionally excluding overbought names (RSI14 > 70)       (user request: RSI)
  V8 drop worst 30%, rank by 0.5 small + 0.5 quality (ROE, lagged 2 days; no PIT guarantee)
  V9 drop worst 30%, rank by 0.5 small + 0.5 value (TTM dividend yield, PIT by implementation notice)
  V10 drop worst 30%, rank by equal weights of small, comp, vm, oversold RSI14, quality, value
  (no PE/PB: the data has no earnings or book value; growth columns are empty)
  holdings: 50 and 100. Overlay: C2 for every variant, plus C0 for V0 as the reference.
  RSI(n) = 100 * mean(gains) / (mean(gains) + mean(losses)) over n days of qfq returns.
  Also reported: Rank IC of RSI6 / RSI14 (train 2019-2021, valid 2022-2023) and their correlation
  with the existing ret5 / ret20 reversal factors.
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


def rsi(n):
    g = lib.rmean(np.maximum(B.r, 0.0), n, int(0.7 * n))
    l = lib.rmean(np.maximum(-B.r, 0.0), n, int(0.7 * n))
    with np.errstate(invalid="ignore", divide="ignore"):
        x = 100.0 * g / (g + l)
    x[~np.isfinite(x)] = np.nan
    return x


RSI = {6: rsi(6), 14: rsi(14)}
d = B.dates
tr = (d >= np.datetime64("2019-04-01")) & (d < np.datetime64("2021-12-01"))
va = (d >= np.datetime64("2022-01-01")) & (d <= np.datetime64("2023-12-29"))
fr = {h: lib.rank_rows(np.where(U, B.fwd[h], np.nan)) for h in (5, 20)}
print("RSI single-factor check (main-board universe; negative IC = low RSI does better):")
for n, x in RSI.items():
    rk = lib.rank_rows(np.where(U, x, np.nan))
    cors = {k: float(np.nanmean(lib.row_corr(rk, lib.rank_rows(np.where(U, F[k], np.nan)))[0])) for k in ("ret5", "ret20")}
    for h in (5, 20):
        ic = lib.row_corr(rk, fr[h])[0]
        print(f"   RSI{n:<2d} h={h:<2d} IC train {np.nanmean(ic[tr]):+.3f} (t {lib.nw_t(ic[tr], h):+.1f})  valid {np.nanmean(ic[va]):+.3f} (t {lib.nw_t(ic[va], h):+.1f})"
              f"   | corr with ret5 {cors['ret5']:+.2f}, ret20 {cors['ret20']:+.2f}", flush=True)
import os
dv = pl.read_parquet(os.path.join(lib.DATA, "dividends.parquet")).filter(
    (pl.col("status") == "实施方案") & pl.col("implementation_pub_date").is_not_null())
cidx = {c: i for i, c in enumerate(B.codes)}
T, N = U.shape
E = np.zeros((T, N))
for c, pub, cash in zip(dv["code"].to_list(), dv["implementation_pub_date"].to_list(), dv["cash_total_10k_cny"].to_list()):
    if c in cidx and cash is not None:
        k = int(np.searchsorted(B.dates, np.datetime64(pub, "D"), side="right"))
        if k < T:
            E[k, cidx[c]] += cash
cs = np.cumsum(E, axis=0); ttm = cs - lib.shift(cs, 243); ttm[:243] = cs[:243]
rng = np.random.default_rng(0)
dy = ttm / 1e4 / B.P["market_cap"] + rng.random((T, N)) * 1e-9          # random tie-break for non-payers
qual = zrank(F["roe_l"], U)
value = zrank(dy, U)
SC = {"V0 现行: 剔30%+小盘": np.where(k30, small, np.nan),
      "V1 剔30%+小盘/综合": np.where(k30, 0.5 * small + 0.5 * comp_r, np.nan),
      "V2 剔30%+小盘/VWAP+MAX": np.where(k30, 0.5 * small + 0.5 * vm, np.nan),
      "V3 剔30%+三者等权": np.where(k30, (small + comp_r + vm) / 3, np.nan),
      "V4 剔50%+小盘": np.where(k50, small, np.nan),
      "V5 剔30%+小盘/LightGBM": np.where(k30, 0.5 * small + 0.5 * ml, np.nan),
      "V6 剔30%+小盘/RSI超卖": np.where(k30, 0.5 * small + 0.5 * zrank(-RSI[14], U), np.nan),
      "V7 现行+剔除RSI>70": np.where(k30 & ~(RSI[14] > 70), small, np.nan),
      "V8 剔30%+小盘/质量ROE": np.where(k30, 0.5 * small + 0.5 * qual, np.nan),
      "V9 剔30%+小盘/价值股息率": np.where(k30, 0.5 * small + 0.5 * value, np.nan),
      "V10 剔30%+六项等权": np.where(k30, np.nanmean(np.stack([small, comp_r, vm, zrank(-RSI[14], U), qual, value]), axis=0), np.nan)}
st = rr.states(B, U)
OV = {"C0": None, "C2": rr.ramp(np.where(st["crowded"], 0.5, 1.0), 0.05, every=1)}
A, Z = "2020-04-01", "2024-12-31"
rows = []
for n_hold in (50, 100):
  for sname, S in SC.items():
    for oname, ex in OV.items():
        if oname == "C0" and not sname.startswith("V0"):
            continue
        row = dict(n=n_hold, score=sname, overlay=oname)
        for cl, c in (("", bt.Costs()), ("_20bp", bt.Costs(slippage=0.002))):
            res = bt.run(B, S, A, Z, n_hold=n_hold, rebal=20, buffer=3.0, U=U, costs=c, exposure=ex,
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
        print(f"  n={n_hold:<3d} {sname:26s} {oname} ann {row['ann']:+.1%} mdd {row['mdd']:+.1%} sharpe {row['sharpe']:.2f}", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/pv_combo_mainboard_2020_2024.csv")
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_cols(18); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(250)
print(R.with_columns([pl.col(c).round(3) for c in R.columns if c not in ("n", "score", "overlay")]))
