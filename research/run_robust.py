import numpy as np, lib, backtest as bt, signals
B = lib.build_base(); F = lib.build_factors(B)
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, B.dates, allf)
TR = ("2019-04-01","2021-12-31"); VA = ("2022-01-01","2023-12-29"); PER = {"TRAIN": TR, "VALID": VA}
SPEC = dict(n_hold=100, rebal=5)          # train-selected: adv 10M, tilt 1.0, n 100, rebalance 5d

def go(label, adv=1e7, costs=bt.Costs(), U_extra=None, **kw):
    U = lib.make_U(B, adv)
    if U_extra is not None: U = U & U_extra
    S = signals.tilted(B, F, signs, U, 1.0)
    res = bt.run(B, S, TR[0], VA[1], U=U, **{**SPEC, **kw}, costs=costs)
    p = bt.by_period(res, PER)
    print(f"{label:34s} TRAIN ann {p['TRAIN']['ann']:+6.1%} sh {p['TRAIN']['sharpe']:4.2f} mdd {p['TRAIN']['mdd']:6.1%} | VALID ann {p['VALID']['ann']:+6.1%} sh {p['VALID']['sharpe']:4.2f} mdd {p['VALID']['mdd']:6.1%}", flush=True)
    return res, U

res, U = go("BASELINE (slip 10bp, ADV>=10M)")
print("   by year (strategy / universe EW):", {y: f"{a:+.0%}/{b:+.0%}" for y,(a,b) in bt.yearly(res).items()})
print("\n-- cost sensitivity (slippage per side) --")
for k in (2, 3, 5): go(f"slippage x{k} ({10*k}bp/side)", costs=bt.Costs(slippage=0.001*k))
print("\n-- liquidity floor --")
for adv in (3e6, 5e6, 3e7): go(f"ADV floor {adv/1e6:.0f}M", adv=adv)
print("\n-- exclude STAR(688)/ChiNext-registration(301): 10% limit boards only --")
cod = np.array([c[:3] for c in B.codes]); main_only = ~np.isin(cod, ["688", "301"])
go("exclude 688 & 301", U_extra=np.broadcast_to(main_only, B.U.shape))
print("\n-- holdings count / frozen-name exposure (names with no price for >=5 days) --")
r2, U2 = go("baseline w/ weights", return_weights=True)
W = r2["weights"]; t0 = int(np.searchsorted(B.dates, r2["dates"][0]))
inv = (~B.valid).astype(int)
cs = np.cumsum(inv, axis=0); run5 = (cs - np.vstack([np.zeros((5, inv.shape[1]), int), cs[:-5]]) >= 5)
fro = (W * run5[t0:t0 + len(W)]).sum(1)
print(f"   avg positions {r2['n_pos'].mean():.0f};  weight frozen in suspended/delisted names: mean {fro.mean():.2%}, max {fro.max():.2%}, days>3%: {(fro>0.03).sum()}")
