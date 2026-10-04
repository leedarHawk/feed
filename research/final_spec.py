"""PRE-REGISTERED FINAL TEST (frozen before the sealed 2025-2026 data is checked out or read).

Common to all versions
  universe   : main board only (exclude 688/689/300/301), non-ST, listed >= 365 calendar days,
               20-day average turnover value >= 5M CNY, not in a no-price-limit regime, buyable at next open
  composite  : 5 price-volume families (lottery: ivol20, maxr20, vol10, skew20 | turnover: turn5, turnr5_60 |
               limit-up: nlimup20 | intraday: intraday20, overnight20 | reversal: ret5, ret20, ret60),
               signs from TRAIN (2019-2021) Rank IC, family-averaged ranks; drop the worst 30% each day
  ranking    : V9 = 0.5 * rank(low 20d turnover value) + 0.5 * rank(TTM dividend yield, PIT by implementation notice)
               V0 = rank(low 20d turnover value) only
  portfolio  : equal weight, rebalance every 20 trading days, keep a holding until its rank > 3 x holdings
  crowding   : 20d mean of bottom-30%-by-cap turnover share > its trailing 250-day 90th percentile -> target 50%;
               exposure moves toward target by `step` per day by selling the worst-scored / buying the best-ranked
               whole positions ("names" mode); same speed when re-risking
  january    : (J versions) flat for the last 10 trading days up to Jan 31, back on the next trading day
  execution  : signal at close, trade at next open; no buy at limit-up open, no sell at limit-down open or when
               suspended; commission 0.025% (min 5 CNY/order), stamp duty, slippage 0.10% per side;
               capital 500,000 CNY; partial trades below one board lot skipped
  schedules  : each version run on 4 rebalance schedules (offset 0/5/10/15 trading days); headline = mean

Evaluation
  sealed period : 2025-01-02 .. 2026-09-30 (the strategy runs continuously from 2020-04-01; only the sealed part
                  is the test). Also reported: 2025, 2026 YTD, 20bp slippage, universe EW benchmark, CSI 1000, HS300.
  pass criteria (main version, schedule mean): annualised return above the universe EW benchmark AND worst max
                  drawdown no deeper than -30% over the sealed period. 21 months is short; one window cannot prove
                  or disprove a ~30%/yr strategy, so results are reported in full either way.
"""
VERSIONS = {
    "MAIN J1-50": dict(score="V9", n_hold=50, step=0.04, january=True),
    "ALT  J1-30": dict(score="V9", n_hold=30, step=0.05, january=True),
    "CTRL V9-50 (no Jan exit)": dict(score="V9", n_hold=50, step=0.04, january=False),
    "CTRL V0-50 (no dividend)": dict(score="V0", n_hold=50, step=0.04, january=True),
}
COMMON = dict(adv_floor=5e6, min_age_days=365, rebal=20, buffer=3.0, offsets=(0, 5, 10, 15), capital=5e5,
              run_start="2020-04-01", sealed_start="2025-01-01", sealed_end="2026-09-30")
