"""One rule for GUI-timing assertions under coverage.

Maintainer decision, 2026-11-02: tests that measure GUI-thread gaps or stall
budgets are not timing-meaningful under coverage line tracing, which multiplies
every step's time. They keep running and asserting in the normal Qt shards. In
coverage runs the test still executes its code (so the lines are covered) and
only the timing assertion is skipped, through ``assert_within_budget``.
"""

from __future__ import annotations

import os
import sys


def timing_is_meaningful() -> bool:
    """Return False while coverage is tracing this process."""
    if sys.gettrace() is not None:
        return False
    if os.environ.get("COVERAGE_PROCESS_START"):
        return False
    try:
        import coverage
    except ImportError:
        return True
    return coverage.Coverage.current() is None


def assert_within_budget(measured: float, budget: float) -> None:
    """Assert ``measured < budget`` unless coverage makes the clock meaningless."""
    if timing_is_meaningful():
        assert measured < budget, (measured, budget)
