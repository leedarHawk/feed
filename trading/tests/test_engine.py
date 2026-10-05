import numpy as np

from trading.config import CostModel
from trading.engine import decide, size_orders, weights_at

NAN = np.nan
codes = np.array([f"60000{i}.XSHG" for i in range(8)])


def test_rebalance_keeps_buffered_holdings_and_fills_by_rank():
    # held: 0 (rank 3), 1 (rank 6 > buffer*n_hold = 4 -> dropped); best names 2, 3, 4, 0
    score = np.array([0.6, 0.2, 0.9, 0.8, 0.7, 0.3, 0.1, NAN])
    w = np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0])
    d = decide(w, score, np.ones(8, bool), expo=1.0, n_hold=2, buffer=2.0, scheduled=True)
    assert d.kind == "rebalance"
    assert set(np.where(d.target > 0)[0]) == {0, 2}          # 0 kept (rank 3 <= 4), then the best new name
    assert np.allclose(d.target[[0, 2]], 0.5)


def test_rebalance_skipped_when_universe_too_small():
    score = np.array([1, NAN, NAN, NAN, NAN, NAN, NAN, NAN])
    d = decide(np.zeros(8), score, np.ones(8, bool), 1.0, n_hold=2, buffer=3, scheduled=True)
    assert d.kind == "skip" and d.target is None


def test_reduce_sells_worst_sellable_and_add_buys_best_unheld():
    score = np.array([0.1, 0.5, 0.9, 0.3, 0.8, NAN, NAN, NAN])
    w = np.array([0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0])
    can_sell = np.array([False, True, True, True, True, True, True, True])
    d = decide(w, score, can_sell, expo=0.5, n_hold=4, buffer=3, scheduled=False)
    assert d.kind == "reduce" and d.n_sel == 2
    assert set(np.where((d.target == 0) & (w > 0))[0]) == {1, 3}        # worst sellable: 0 opens limit-down
    w2 = np.array([0.25, 0, 0, 0, 0, 0, 0, 0])
    d2 = decide(w2, score, np.ones(8, bool), expo=0.75, n_hold=4, buffer=3, scheduled=False)
    assert d2.kind == "add" and set(np.where((d2.target > 0) & (w2 == 0))[0]) == {2, 4}


def test_size_orders_lots_cash_and_blocks():
    c = CostModel()
    px = np.array([10.0, 20.0, 5.0, 8.0, NAN, 10, 10, 10])
    shares = np.array([5000.0, 2500.0, 0, 0, 0, 0, 0, 0])
    cash = 0.0
    w, nav = weights_at(shares, cash, px, px)
    assert nav == 100000
    target = np.array([0.0, 0.5, 0.5, 0.0, 0.5, 0, 0, 0])      # sell 0, keep 1, buy 2, (4 has no price)
    dec = type("D", (), {"target": target})()
    can = np.isfinite(px)
    orders = size_orders(dec, w, shares, cash, nav, px, can, can, c, "2026-10-08", codes)
    by = {(o.code, o.side): o for o in orders}
    assert by[(codes[0], "sell")].qty == 5000                    # full exit, whatever the lot
    buy = by[(codes[2], "buy")]
    assert buy.qty % 100 == 0
    spend = buy.qty * 5.0 * (1 + c.slippage) + c.fee(buy.qty * 5.0)
    proceeds = 5000 * 10 * (1 - c.slippage - c.stamp("2026-10-08")) - c.fee(50000)
    assert spend <= proceeds + 1e-6                               # cash never goes negative
    assert (codes[4], "buy") not in by                             # suspended / no price: not bought


def test_tiny_trades_below_one_lot_are_skipped():
    c = CostModel()
    px = np.array([10.0, 10.0, 10, 10, 10, 10, 10, 10])
    shares = np.array([5050.0, 4950.0, 0, 0, 0, 0, 0, 0])
    w, nav = weights_at(shares, 0.0, px, px)
    dec = type("D", (), {"target": np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0])})()
    assert size_orders(dec, w, shares, 0.0, nav, px, np.ones(8, bool), np.ones(8, bool), c, "2026-10-08", codes) == []


def test_blocked_sells_scale_buys_down():
    c = CostModel(min_fee=0.0, slippage=0.0, commission=0.0)
    px = np.full(8, 10.0)
    shares = np.array([5000.0, 5000.0, 0, 0, 0, 0, 0, 0])
    w, nav = weights_at(shares, 0.0, px, px)
    dec = type("D", (), {"target": np.array([0, 0, 0.5, 0.5, 0, 0, 0, 0])})()
    can_sell = np.array([True, False, True, True, True, True, True, True])     # name 1 opens limit-down
    orders = size_orders(dec, w, shares, 0.0, nav, px, np.ones(8, bool), can_sell, c, "2026-10-08", codes,
                         fractional=True)
    buys = sum(o.value for o in orders if o.side == "buy")
    assert abs(buys - 50000 * (1 - c.stamp("2026-10-08"))) < 1.0
