"""Walk-forward LightGBM score: expanding window, quarterly refit, embargo for label overlap.
Hyper-parameters are fixed up front (no tuning on any evaluation window)."""
import numpy as np
import lightgbm as lgb
import backtest

PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=2000,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0,
              n_jobs=4, verbose=-1, random_state=7)
EXCLUDE = {"roe_l", "roa_l"}          # no PIT guarantee, unstable sign across train/valid


def walk_forward(B, F, U, start_pred, h=5, retrain_every=63, embargo=25, stride=3, log=print):
    names = [n for n in F if n not in EXCLUDE]
    zf = {n: backtest.zrank(F[n], U) for n in names}
    y = backtest.zrank(B.fwd[h], U)
    T = U.shape[0]
    pred = np.full(U.shape, np.nan, dtype=np.float32)
    imp = np.zeros(len(names))
    a0 = int(np.searchsorted(B.dates, np.datetime64(start_pred)))
    fits = 0
    for a in range(a0, T, retrain_every):
        b = min(a + retrain_every, T)
        days = np.arange(60, a - embargo, stride)
        m = U[days] & ~np.isnan(y[days])
        X = np.column_stack([zf[n][days][m] for n in names])
        Y = y[days][m]
        model = lgb.LGBMRegressor(**PARAMS).fit(X, Y)
        imp += model.booster_.feature_importance("gain")
        rows = U[a:b]
        Xp = np.column_stack([zf[n][a:b][rows] for n in names])
        p = np.full(rows.shape, np.nan, dtype=np.float32)
        p[rows] = model.predict(Xp)
        pred[a:b] = p
        fits += 1
        log(f"  fit {fits}: train rows {len(Y):,} (days<{B.dates[a - embargo]}), predict {B.dates[a]}..{B.dates[b - 1]}")
    imp = imp / imp.sum()
    return pred, dict(zip(names, imp))
