import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="run tests that load the full price panel (minutes, ~6 GB)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads the full price panel")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="needs --slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
