import numpy as np, lib, backtest as bt, signals
B = lib.build_base(); F = lib.build_factors(B)
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, B.dates, allf)
S = signals.composite(B, F, signs)
def decile_table(score, h, a, b, label):
    d = B.dates; m = (d >= np.datetime64(a)) & (d <= np.datetime64(b))
    sc = np.where(B.U, score, np.nan)[m]; fw = B.fwd[h][m]
    rk = lib.rank_rows(sc); n = np.sum(~np.isnan(rk), axis=1, keepdims=True)
    q = np.floor(rk / np.maximum(n, 1) * 10)
    mu_all = np.nanmean(np.where(~np.isnan(sc), fw, np.nan), axis=1)
    rows = []
    for k in range(10):
        mk = (q == k) & ~np.isnan(fw)
        rows.append(np.nanmean(np.where(mk, fw, np.nan), axis=1) - mu_all)   # excess over universe mean, per day
    ex = np.array(rows)  # [10, days]
    per = 252 / h
    print(f"{label} h={h}: mean excess return per holding period, annualised x{per:.0f} (no costs), decile 1(worst)..10(best):")
    print("   ", " ".join(f"{np.nanmean(e)*per:+6.1%}" for e in ex))
decile_table(S, 5, "2019-04-01", "2021-12-31", "TRAIN")
decile_table(S, 5, "2022-01-01", "2023-12-15", "VALID")
decile_table(S, 20, "2019-04-01", "2021-12-31", "TRAIN")
decile_table(S, 20, "2022-01-01", "2023-11-15", "VALID")
# what are the top-50 like?
d = B.dates; t = np.searchsorted(d, np.datetime64("2021-06-30"))
s = np.where(B.U, S, np.nan)[t]; top = np.argsort(-np.nan_to_num(s, nan=-9))[:50]
mc = B.P["market_cap"][t]; uni = B.U[t]
print(f"\nTop-50 on 2021-06-30: median mkt cap {np.nanmedian(mc[top]):.0f}亿  vs universe median {np.nanmedian(mc[uni]):.0f}亿 | median 20d ADV {np.nanmedian(B.adv20[t][top])/1e6:.0f}M vs {np.nanmedian(B.adv20[t][uni])/1e6:.0f}M")
print("  top-50 vol20 median %.3f vs universe %.3f; turn20 %.4f vs %.4f" % (np.nanmedian(F['vol10'][t][top]), np.nanmedian(F['vol10'][t][uni]), np.nanmedian(F['turn20'][t][top]), np.nanmedian(F['turn20'][t][uni])))
