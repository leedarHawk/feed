"""Account configuration (TOML). Strategy parameters are never set here: they come from the frozen
research/final_spec.py version named in [strategy].version."""
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .bridge import ROOT, research

HOME = Path(os.environ.get("FEED_TRADING_HOME", ROOT / "trading" / "var"))
CONFIG_DIR = ROOT / "trading" / "config"


@dataclass(frozen=True)
class Strategy:
    version: str            # key of final_spec.VERSIONS
    score: str              # "V0" or "V9"
    n_hold: int
    step: float             # crowding ramp speed per day
    january: bool           # flat for the last 10 trading days of January
    adv_floor: float
    min_age_days: int
    rebal: int
    buffer: float
    anchor: str             # rebalance schedule anchor (trading day index 0)
    offset: int             # 0 / 5 / 10 / 15 in the research


@dataclass(frozen=True)
class CostModel:
    commission: float = 0.00025
    min_fee: float = 5.0
    slippage: float = 0.0010
    stamp_old: float = 0.0010
    stamp_new: float = 0.0005
    stamp_change: str = "2023-08-28"
    lot: int = 100

    def stamp(self, day):
        return self.stamp_new if str(day) >= self.stamp_change else self.stamp_old

    def fee(self, value):
        return max(self.min_fee, self.commission * value) if value > 0 else 0.0


@dataclass(frozen=True)
class RiskLimits:
    drawdown_warn: float = -0.20
    drawdown_halt: float = -0.30          # the pre-registered failure line of the final test
    daily_loss_warn: float = -0.07
    turnover_rebalance: float = 1.20      # one-way order value / NAV allowed on a rebalance day
    turnover_other: float = 0.35
    max_orders: int = 150
    max_name_weight: float = 0.06
    max_industry_weight: float = 0.35
    min_universe: int = 150
    min_coverage: float = 0.97            # traded names today / yesterday


@dataclass(frozen=True)
class Execution:
    fractional: bool = False              # True only for parity replays against the research backtest
    cash_buffer: float = 0.0              # fraction of NAV kept back from buys
    limit_band: float = 0.03              # manual orders: limit price band around the reference price
    qmt_path: str = ""                    # miniQMT userdata_mini directory
    qmt_account: str = ""
    allow_live: bool = False              # QMT orders are only sent when this is true AND --live is passed


@dataclass(frozen=True)
class Account:
    name: str
    broker: str                           # paper | manual | qmt
    capital: float
    start: str                            # first trading day of the account
    strategy: Strategy
    costs: CostModel = field(default_factory=CostModel)
    risk: RiskLimits = field(default_factory=RiskLimits)
    execution: Execution = field(default_factory=Execution)
    enabled: bool = True

    @property
    def home(self):
        return HOME / "accounts" / self.name


def strategy_from_spec(version, anchor=None, offset=0):
    fs = research().final_spec
    if version not in fs.VERSIONS:
        raise KeyError(f"unknown strategy version {version!r}; frozen versions: {list(fs.VERSIONS)}")
    v, c = fs.VERSIONS[version], fs.COMMON
    return Strategy(version=version, score=v["score"], n_hold=v["n_hold"], step=v["step"], january=v["january"],
                    adv_floor=c["adv_floor"], min_age_days=c["min_age_days"], rebal=c["rebal"], buffer=c["buffer"],
                    anchor=anchor or c["run_start"], offset=int(offset))


def load_account(path_or_name):
    p = Path(path_or_name)
    if not p.exists():
        p = CONFIG_DIR / "accounts" / f"{path_or_name}.toml"
    with open(p, "rb") as f:
        raw = tomllib.load(f)
    a, s = raw["account"], raw["strategy"]
    if a["broker"] not in ("paper", "manual", "qmt"):
        raise ValueError(f"broker must be paper, manual or qmt, not {a['broker']!r}")
    strat = strategy_from_spec(s["version"], s.get("anchor"), s.get("offset", 0))
    return Account(name=a["name"], broker=a["broker"], capital=float(a.get("capital", 0.0)), start=str(a["start"]),
                   strategy=strat, costs=CostModel(**raw.get("costs", {})), risk=RiskLimits(**raw.get("risk", {})),
                   execution=Execution(**raw.get("execution", {})), enabled=bool(a.get("enabled", True)))


def all_accounts():
    return [load_account(p) for p in sorted((CONFIG_DIR / "accounts").glob("*.toml"))]
