"""Access to the frozen research modules (research/*.py) with live-data settings.

The research code is the single source of truth for the strategy. It is imported unchanged; the trading
system only points it at a different price cache (panel.CACHE) and reference-data directory (lib.DATA).
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
DATA = ROOT / "data"

_R = None


def research():
    """Import panel, lib, backtest, signals, risk_rules, funding_proxies, final_spec once and return them."""
    global _R
    if _R is not None:
        return _R
    os.environ["FEED_LAST_YEAR"] = "9999"           # panel reads this at import; the trading cache holds all years
    if str(RESEARCH) not in sys.path:
        sys.path.insert(0, str(RESEARCH))
    import panel, lib, backtest, signals, risk_rules, funding_proxies, final_spec   # noqa: E401
    panel.LAST_ALLOWED_YEAR = 9999
    _R = SimpleNamespace(panel=panel, lib=lib, backtest=backtest, signals=signals, risk_rules=risk_rules,
                         funding_proxies=funding_proxies, final_spec=final_spec)
    return _R


def activate(cache_dir, ref_dir):
    """Point the research loaders at the trading system's price cache and reference tables."""
    R = research()
    R.panel.CACHE = str(cache_dir)
    R.lib.DATA = str(ref_dir)
    return R
