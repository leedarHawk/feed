"""Composite scores built from family-level rank averages. Signs come from TRAIN IC only."""
import numpy as np
import lib, backtest

FAMILIES = {
    "lottery":   ["ivol20", "maxr20", "vol10", "skew20"],
    "turnover":  ["turn5", "turnr5_60"],
    "limitup":   ["nlimup20"],
    "intraday":  ["intraday20", "overnight20"],
    "reversal":  ["ret5", "ret20", "ret60"],
}


def train_signs(ics, dates, factors, split="2022-01-01", embargo=25, h=10):
    s = int(np.searchsorted(dates, np.datetime64(split)))
    return {f: float(np.sign(np.nanmean(ics[(f, h)][:max(s - embargo, 0)]))) for f in factors}


def composite(B, F, signs, families=FAMILIES, weights=None):
    fam_scores = []
    for fam, names in families.items():
        z = [signs[n] * backtest.zrank(F[n], B.U) for n in names]
        fam_scores.append((weights or {}).get(fam, 1.0) * np.nanmean(np.stack(z), axis=0))
    return np.nanmean(np.stack(fam_scores), axis=0)


def tilted(B, F, signs, U, tilt, junk_cut=0.3, base_families=FAMILIES):
    """Stage 1: drop the worst `junk_cut` share by the family composite (anti-lottery etc.).
    Stage 2: rank survivors by (1-tilt)*composite + tilt*small-liquidity score (low 20d turnover value)."""
    z = [signs[n] * backtest.zrank(F[n], U) for fams in base_families.values() for n in fams[:1]]
    comp = np.nanmean(np.stack([np.nanmean(np.stack([signs[n] * backtest.zrank(F[n], U) for n in names]), axis=0)
                                for names in base_families.values()]), axis=0)
    comp_r = backtest.zrank(comp, U)                       # [-1, 1]
    small = backtest.zrank(-F["lmoney20"], U)              # +1 = least liquid in universe
    score = (1 - tilt) * comp_r + tilt * small
    score = np.where(comp_r > (2 * junk_cut - 1), score, np.nan)   # junk filter
    return score
