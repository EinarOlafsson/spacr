"""Giving up on a unit when there is no exception to give up over.

``_give_up`` ends every failed unit: it records the failure and then
either skips or re-raises. Its last line is ``raise exc``, so an ``exc``
of None would raise TypeError from the error handler itself -- the run
would die of the thing meant to keep it alive. The guard above returns
instead.

Reaching it through :meth:`attempts_for` is not possible today, and this
file says exactly why and asserts the two facts that make it so.
"""
from __future__ import annotations

import pytest

from spacr.runctx import (ON_ERROR_RETRY, ON_ERROR_SKIP, ON_ERROR_STOP,
                          ErrorPolicy)


class TestTheProducingSide:
    """Two facts stop ``last`` reaching ``_give_up`` as None."""

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

    def test_an_attempt_that_never_ran_returns_before_giving_up(self):
        """Fact two: no exception means nothing to judge.

        A ``break`` or ``continue`` past the ``with`` leaves the attempt
        with ok False and exc None. That is not a failure, so the loop
        returns rather than recording one.
        """
        policy = ErrorPolicy(ON_ERROR_STOP)
        for attempt in policy.attempts_for("plate1", stage="plate"):
            break                       # the body never ran

        assert policy.skips == []
        assert policy.retries == []


class TestGivingUpWithNothingToRaise:

    def test_it_records_the_failure_and_returns_rather_than_raising_none(self):
        """THE UNCOVERED GUARD.

        Left to fall through, the method reaches ``raise exc`` with exc
        None, and Python raises ``TypeError: exceptions must derive from
        BaseException`` from inside the error handler -- a run killed by
        its own retry policy, with the original unit's name nowhere in
        the traceback.

        Recording still happens: a unit that was given up on belongs in
        the ledger whether or not anything is left to re-raise.
        """
        policy = ErrorPolicy(ON_ERROR_STOP)

        policy._give_up("plate1", "plate", None, 0)    # must not raise

        assert policy.ledger.failures, (
            "the give-up was not recorded before returning")

    def test_the_same_in_skip_mode_adds_no_skip_record(self):
        """A skip record needs a reason, and there is no exception to
        take one from."""
        policy = ErrorPolicy(ON_ERROR_SKIP)

        policy._give_up("plate1", "plate", None, 0)

        assert policy.skips == []

    def test_with_a_real_exception_stop_mode_re_raises_it(self):
        """The live path either side of the guard."""
        policy = ErrorPolicy(ON_ERROR_STOP)
        boom = RuntimeError("the plate could not be read")

        with pytest.raises(RuntimeError, match="could not be read"):
            policy._give_up("plate1", "plate", boom, 1)

    def test_with_a_real_exception_skip_mode_records_and_continues(self):
        policy = ErrorPolicy(ON_ERROR_SKIP)
        boom = RuntimeError("the plate could not be read")

        policy._give_up("plate1", "plate", boom, 1)

        assert len(policy.skips) == 1
        record = policy.skips[0]
        assert record.unit == "plate1"
        assert record.exc_type == "RuntimeError"
        assert "could not be read" in record.reason
