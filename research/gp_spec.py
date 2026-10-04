"""PRE-REGISTERED GP FACTOR MINING (frozen before any mining run; mining data stops at 2024-12-31).

Goal
  Find price-volume formulas that add excess return to the V0 small-cap strategy (CTRL V0-50 J1 in
  final_spec.py), i.e. that rank stocks well INSIDE the pool V0 picks from, beyond V0's own ranking.

Data and splits (research cache, FEED_LAST_YEAR unset -> panel ends 2024-12-31; 2025+ is not loaded)
  warm-up : 2019
  train   : signal dates 2020-01-02 .. 2023-12, forward window inside 2020-2023     -> GP fitness
  valid   : signal dates 2024, forward window inside 2024                           -> selection filter
  test    : signal dates 2025-01-02 .. 2026-09 (forward window <= 2026-09-30)       -> one shot, after freezing
  Caveat: 2025-2026 prices were already used once (final_spec.py test of the V0/V9 strategies), so the
  test is clean for the mined formulas, lambda and the overlay, but V0's own 2025-2026 result is known.

Mining universe M and target (gp_export.py)
  M       : V0 universe (main board, non-ST, listed >= 365 days, ADV20 >= 5M CNY, buyable next open),
            after V0's junk filter (family composite rank > -0.4), least liquid 30% of the universe
            (small score zrank(-lmoney20) >= 0.4); median ~750 stocks per day in train
  target  : forward 20-trading-day return, buy next open, sell first available open at/after t+21
  IC      : daily Spearman IC inside M of the factor rank (missing -> middle rank) after an OLS removal
            of the small-score rank, against the forward-return rank

Search space (gpminer)
  terminals: ret, oret (overnight), iret (intraday), hl, ch, cl (same-day high/low/close ratios),
             vwapc, turn, lmoney, lcap, lprice, uplim, dnlim, cumr (sum of past log returns)
  operators: abs slog sign cs_rank cs_z | add sub mul div | ts_mean ts_std ts_delta ts_delay ts_max
             ts_min ts_rank ts_z ts_decay (windows 3/5/10/20/40/60) | ts_corr (5/10/20/40/60)
  rolling windows need >= 60% valid days; cs_* run inside the tradable universe of each day
  fitness  : |mean IC / sd IC| on train x stability (4/4 train years same sign 1.0, 3/4 0.6, else 0.2)
             - 0.004 per node above 7; coverage inside M must be >= 90%
  GP       : ramped half-and-half init (depth 2-4), tournament 5, elitism 10, crossover 0.6,
             subtree mutation 0.2, point mutation 0.15, shrink 0.05, max depth 6, max 20 nodes;
             hall of fame (cap 200) for fitness >= 0.15 and >= 3/4 stable years, near-duplicates
             (|corr| >= 0.7 of size-neutral ranks on every 8th train day) keep only the better one
  budget   : 4 independent runs (seeds 1-4) x population 400 x 30 generations, all reported

Selection (train + valid only; run_gp_select.py)
  1. pool = union of the 4 halls of fame, deduplicated at |corr| >= 0.7 (better train NW-t kept)
  2. keep if: train Newey-West t (lag 20) |t| >= 3.0; all 4 train years same sign as the train mean;
     2024 mean IC same sign with NW t >= 1.0 in that direction
  3. greedy by train |NW t|: add if |corr| < 0.5 with every factor already chosen; at most 8
  4. none left -> stop, report "nothing passes", no overlay test
  composite gp = zrank( mean_k sign_k * zrank(f_k, U) , U )
  overlay      : score = (1 - lam) * small + lam * gp inside V0's junk-filtered pool (V0 = lam 0)
  lam          : from {0.15, 0.30, 0.50}, the best mean Sharpe over the 4 rebalance schedules on
                 2020-04-01 .. 2024-12-31 (10bp), research cache only. In-sample by construction.

Final test (one shot, FEED_LAST_YEAR=2026; run_gp_final.py)
  - test-period IC of each chosen factor and of the composite (same IC definition as mining)
  - backtest exactly as final_spec.py CTRL V0-50 (no dividend, January exit, 50 names, step 0.04,
    4 schedules, 500k CNY, 10bp and 20bp) for V0 and for V0+GP(lam), periods: 2020-04..2024-12,
    2025-01..2026-09, 2025, 2026 YTD
  pass (both required): composite test IC has the train sign with NW t >= 1.5; AND V0+GP beats V0 on
    2025-01..2026-09 annualised return (schedule mean, 10bp) with worst max drawdown at most 3 points
    deeper. Results are reported in full either way.
"""
SEEDS = (1, 2, 3, 4)
GP_ARGS = dict(pop=400, gens=30, hof=200, **{"hof-min": 0.15, "max-depth": 6, "max-nodes": 20, "dup-corr": 0.7})
SELECT = dict(t_train=3.0, t_valid=1.0, dedup_corr=0.7, max_corr=0.5, max_factors=8, nw_lag=20)
LAMBDAS = (0.15, 0.30, 0.50)
TEST_PASS = dict(t_test=1.5, mdd_slack=0.03)
