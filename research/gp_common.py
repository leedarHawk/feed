"""Shared pieces for the GP selection and final test: V0 setup exactly as final_spec.py CTRL V0-50,
loading gpminer factor matrices, the GP composite, the overlay score and a Python IC that mirrors
gpminer's (used for the composite, which is not a single formula)."""
import json, os, subprocess
import numpy as np
from scipy.stats import rankdata
import lib, backtest as bt, signals, risk_rules as rr, final_spec as fs

EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpminer", "target", "release", "gpminer")
C = fs.COMMON
V0 = fs.VERSIONS["CTRL V0-50 (no dividend)"]


class Setup:
    pass


def setup(end):
    S = Setup()
    B = lib.build_base(end=end, min_age_days=C["min_age_days"])
    F = lib.build_factors(B)
    pre = np.array([c[:3] for c in B.codes])
    U = lib.make_U(B, C["adv_floor"]) & (~np.isin(pre, ["688", "689", "300", "301"]))[None, :]
    z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)      # TRAIN-period ICs (frozen signs)
    ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
    allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
    signs = signals.train_signs(ics, z["dates"], allf)
    comp_r = bt.zrank(signals.composite(B, F, signs, U=U), U)
    small = bt.zrank(-F["lmoney20"], U)
    crowd = np.where(rr.states(B, U)["crowded"], 0.5, 1.0)
    d = B.dates
    jan = np.ones(len(d))
    for y in range(2019, 2027):
        i = int(np.searchsorted(d, np.datetime64(f"{y}-01-31"), side="right")) - 1
        if i >= 10 and str(d[i])[:7] == f"{y}-01":
            jan[i - 10:i] = 0.0
    S.B, S.F, S.U, S.small, S.k30 = B, F, U, small, comp_r > -0.4
    S.M = U & S.k30 & (small >= 0.4)
    S.ex = rr.ramp(crowd, V0["step"], every=1) * jan
    return S


def eval_formulas(data_dir, formulas, out_json, write_dir=None):
    path = out_json + ".formulas.txt"
    open(path, "w").write("\n".join(formulas) + "\n")
    args = [EXE, "eval", "--data", data_dir, "--formulas", path, "--out", out_json]
    if write_dir:
        args += ["--write-factors", write_dir]
    subprocess.run(args, check=True, stderr=subprocess.DEVNULL)
    return json.load(open(out_json))


def load_factor(S, data_dir, path):
    """gpminer output ([n][t] over the export's columns) -> [T, N] aligned with S.B.codes."""
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    assert meta["T"] == len(S.B.dates) and meta["dates"][-1] == str(S.B.dates[-1])
    f = np.fromfile(path, dtype=np.float32).reshape(meta["N"], meta["T"]).T
    cidx = {c: i for i, c in enumerate(S.B.codes)}
    out = np.full((len(S.B.dates), len(S.B.codes)), np.nan, dtype=np.float32)
    out[:, [cidx[c] for c in meta["codes"]]] = f
    return out


def zrank_avg(x, U):
    """Like backtest.zrank but ties share their average rank (backtest.zrank breaks ties by column
    position, which for sparse factors encodes stock-code order)."""
    out = np.full(x.shape, np.nan, dtype=np.float32)
    for t in range(x.shape[0]):
        m = U[t] & np.isfinite(x[t])
        n = int(m.sum())
        if n >= 2:
            out[t, m] = 2.0 * (rankdata(x[t, m]) - 1) / (n - 1) - 1.0
    return out


def composite(S, facs, signs):
    z = np.stack([s * zrank_avg(f, S.U) for f, s in zip(facs, signs)])
    with np.errstate(invalid="ignore"):
        g = np.nanmean(z, axis=0)
    return zrank_avg(g, S.U)


def overlay(S, gp, lam):
    g = np.where(np.isnan(gp), 0.0, gp)
    return np.where(S.k30, (1 - lam) * S.small + lam * g, np.nan)


def run_bt(S, score, start, end, costs, periods):
    """Mean / worst metrics over the 4 rebalance schedules (CTRL V0-50 settings)."""
    per = {k: [] for k in periods}
    for o in C["offsets"]:
        r = bt.run(S.B, score, start, end, n_hold=V0["n_hold"], rebal=C["rebal"], buffer=C["buffer"], U=S.U,
                   costs=costs, exposure=S.ex, exposure_trade="names", rebal_offset=o)
        for k, m in bt.by_period(r, periods).items():
            per[k].append(m)
    return {k: dict(ann=float(np.mean([m["ann"] for m in ms])), ann_min=float(min(m["ann"] for m in ms)),
                    mdd_worst=float(min(m["mdd"] for m in ms)), sharpe=float(np.mean([m["sharpe"] for m in ms])),
                    excess=float(np.mean([m["excess_ann"] for m in ms])))
            for k, ms in per.items() if ms}


def py_ic(S, f, t_list, H=20):
    """Same IC as gpminer: inside M, missing factor -> middle rank, OLS removal of the small rank.
    A day on which the factor does not vary inside M gets IC 0 (it ranks nothing)."""
    fwd, out = S.B.fwd[H], []
    for t in t_list:
        idx = S.M[t] & np.isfinite(fwd[t]) & np.isfinite(S.small[t])
        m = int(idx.sum())
        if m < 50:
            out.append(np.nan)
            continue
        fr = (rankdata(fwd[t, idx]) - 1) / (m - 1); fr -= fr.mean()
        sr = (rankdata(S.small[t, idx]) - 1) / (m - 1); sr -= sr.mean()
        v = f[t, idx].astype(np.float64); rf = np.full(m, 0.5); fin = np.isfinite(v)
        if fin.sum() < 2:
            out.append(0.0)
            continue
        rf[fin] = (rankdata(v[fin]) - 1) / (fin.sum() - 1); rf -= rf.mean()
        e = rf - (rf @ sr) / (sr @ sr) * sr
        ss = e @ e
        out.append(0.0 if ss < 1e-10 else float(e @ fr / np.sqrt(ss * (fr @ fr))))
    return np.array(out)
