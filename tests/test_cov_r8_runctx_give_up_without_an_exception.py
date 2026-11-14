"""Giving up on a unit when there is no exception to give up over.

``_give_up`` ends every failed unit: it records the failure and then
either skips or re-raises. Its last line is ``raise exc``, so an ``exc``
of None would raise TypeError from the error handler itself -- the run
would die of the thing meant to keep it alive. The guard above returns
instead.

Reaching it through :meth:`attempts_for` is not possible today; this file
asserts the construction-time fact that makes it so. The guard itself, and
the body-never-ran return, are driven in tests/test_runctx_policy_and_seeding.py.
"""
from __future__ import annotations

import pytest

from spacr.runctx import ON_ERROR_RETRY, ErrorPolicy


class TestTheProducingSide:
    """Why ``last`` cannot reach ``_give_up`` as None."""

    def test_a_retry_budget_of_zero_is_refused_at_construction(self):
        """Fact one: the attempt loop always runs at least once.

        ``range(1, total + 1)`` with total 0 is empty, so the loop would
        fall straight through to _give_up having set nothing.
        """
        with pytest.raises(ValueError, match="at least 1"):
            ErrorPolicy(ON_ERROR_RETRY, attempts=0)
        with pytest.raises(ValueError, match="at least 1"):
            ErrorPolicy(ON_ERROR_RETRY, attempts=-3)

        assert ErrorPolicy(ON_ERROR_RETRY, attempts=1).attempts == 1
