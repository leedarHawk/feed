"""The live strategy path against the research code on real data (slow: loads the price panel)."""
import os

import numpy as np
import pytest

from trading.config import HOME
from trading.data import Store
from trading.strategy import (FACTORS, SignalRow, dividend_yield, frozen_signs, lean_factors, mainboard_mask,
                              universe_base)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def research_panel():
    R = Store(HOME).activate()
    B = R.lib.build_base(end="2020-09-30", min_age_days=365)
    F = R.lib.build_factors(B)
    return R, B, F


def test_signs_match_the_train_ic_cache(research_panel):
    R, _, _ = research_panel
    path = "/tmp/feed_cache/ics_discovery.npz"
    if not os.path.exists(path):
        pytest.skip("research IC cache not present")
    z = np.load(path, allow_pickle=True)
    ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
    assert R.signals.train_signs(ics, z["dates"], list(FACTORS)) == frozen_signs()


def test_lean_factors_equal_research(research_panel):
    _, B, F = research_panel
    L = lean_factors(B)
    for k, v in L.items():
        assert np.array_equal(v, F[k], equal_nan=True), k


def test_dividend_yield_research_tie_break_equal(research_panel):
    R, B, _ = research_panel
    assert np.array_equal(dividend_yield(B, "research"), R.signals.dividend_yield_ttm(B), equal_nan=True)


def test_row_scores_equal_research_matrix(research_panel):
    R, B, F = research_panel
    mb = mainboard_mask(B.codes)
    U = R.lib.make_U(B, 5e6) & mb[None, :]
    Ub = universe_base(B, 5e6, mb)
    nxt = np.zeros_like(B.can_buy_open)
    nxt[:-1] = B.can_buy_open[1:]
    assert np.array_equal(Ub & nxt, U)
    signs = frozen_signs()
    SC = R.signals.mainboard_scores(B, F, signs, U)
    dy = dividend_yield(B, "research")
    for s in range(260, len(B.dates) - 1, 9):
        row = SignalRow(str(B.dates[s]), B.codes, Ub[s], {k: F[k][s] for k in FACTORS + ("lmoney20",)}, dy[s],
                        B.P["close"][s], B.valid[s], signs)
        for kind in ("V0", "V9"):
            got, _ = row.score(B.can_buy_open[s + 1], kind)
            assert np.array_equal(got, SC[kind][s], equal_nan=True), (kind, B.dates[s])
