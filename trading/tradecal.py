"""Trading calendar: realized days from the price data, projected forward with config/holidays.toml."""
import datetime as dt
import tomllib

import numpy as np

from .config import CONFIG_DIR


class CalendarError(RuntimeError):
    pass


def _d(x):
    return x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x)[:10])


class Calendar:
    def __init__(self, realized, holidays_file=CONFIG_DIR / "holidays.toml"):
        self.realized = [_d(x) for x in np.asarray(realized).astype("datetime64[D]").astype(str)]
        self._set = set(self.realized)
        with open(holidays_file, "rb") as f:
            raw = tomllib.load(f).get("years", {})
        self.closed = {_d(x) for y in raw.values() for x in y.get("closed", [])}
        self.years = {int(k): bool(v.get("confirmed", False)) for k, v in raw.items()}
        self.warnings = []

    @property
    def last_realized(self):
        return self.realized[-1]

    def _projected_open(self, d):
        if d.weekday() >= 5 or d in self.closed:
            return False
        if d.year not in self.years:
            raise CalendarError(f"no holiday schedule for {d.year} in config/holidays.toml; add it before trading")
        if not self.years[d.year]:
            msg = f"holiday schedule for {d.year} is provisional (config/holidays.toml)"
            if msg not in self.warnings:
                self.warnings.append(msg)
        return True

    def is_open(self, d):
        d = _d(d)
        if d <= self.last_realized:
            return d in self._set
        return self._projected_open(d)

    def next_day(self, d):
        d = _d(d) + dt.timedelta(days=1)
        for _ in range(40):
            if self.is_open(d):
                return d
            d += dt.timedelta(days=1)
        raise CalendarError(f"no trading day within 40 days after {d}")

    def january_exit_days(self, year, n=10):
        """The last n trading days of January `year` (trade days on which the J versions hold nothing)."""
        days, d = [], dt.date(year, 1, 31)
        while d.month == 1:
            if self.is_open(d):
                days.append(d)
                if len(days) == n:
                    break
            d -= dt.timedelta(days=1)
        return sorted(days)
