import time, numpy as np
import lib, backtest as bt, signals, ml
t0 = time.time()
B = lib.build_base(); F = lib.build_factors(B)
U = lib.make_U(B, 1e7)
pred, imp = ml.walk_forward(B, F, U, "2020-04-01", log=lambda s: print(s, flush=True))
np.save("/tmp/feed_cache/ml_pred_discovery.npy", pred)
print(f"walk-forward done {time.time()-t0:.0f}s")
print("top feature importance (gain share):", {k: round(v, 3) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:12]})
# out-of-sample rank IC of the prediction (h=5), inside U
d = B.dates; o = int(np.searchsorted(d, np.datetime64("2020-04-01")))
rk_p = lib.rank_rows(np.where(U, pred, np.nan)); rk_f = lib.rank_rows(np.where(U, B.fwd[5], np.nan))
ic = lib.row_corr(rk_p, rk_f)[0][o:]
print(f"OOS rank-IC h=5: mean {np.nanmean(ic):.4f}  NW-t {lib.nw_t(ic, 5):.1f}   (hand-built composite for reference below)")
OOS = ("2020-04-01", "2023-12-29")
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, B.dates, allf)
S_hand = signals.tilted(B, F, signs, U, 1.0)
PER = {"2020-04..2021-12": ("2020-04-01","2021-12-31"), "2022-2023": ("2022-01-01","2023-12-29"), "ALL OOS": OOS}
for label, S in (("ML walk-forward", pred), ("hand composite (spec A)", S_hand)):
    for n_hold, rebal in ((100, 5), (50, 5), (100, 10)):
        res = bt.run(B, S, OOS[0], OOS[1], n_hold=n_hold, rebal=rebal, U=U)
        print(f"\n[{label}] top{n_hold} reb{rebal}")
        for k, m in bt.by_period(res, PER).items(): print(f"   {k:18s} {bt.fmt(m)}")
        print("   by year:", {y: f"{a:+.0%}/{b:+.0%}" for y,(a,b) in bt.yearly(res).items()})
