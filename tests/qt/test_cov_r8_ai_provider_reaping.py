"""Reaping an AI provider's child process, and what "exited" is allowed to mean.

The rule these three helpers share is stated in `_terminate_and_reap`'s
docstring: **a failed signal is not evidence that the child died.**
Callers use the confirmation to decide whether to keep an uncertain child
registered for a later retry, because that handle is the only thing that
can unblock its reader.

So every uncertainty resolves to "still live", never to "gone". Reporting
a child dead when it is not loses the handle and leaks the process; the
opposite costs one retry.
"""
from __future__ import annotations

import subprocess

import pytest

from spacr.qt.ai import providers as P


class _Proc:
    """A stand-in Popen whose three verbs can each be told to misbehave."""

    def __init__(self, *, poll_result=None, poll_error=None,
                 kill_error=None, terminate_error=None, wait_error=None):
        self._poll_result = poll_result
        self._poll_error = poll_error
        self._kill_error = kill_error
        self._terminate_error = terminate_error
        self._wait_error = wait_error
        self.killed = False
        self.terminated = False

    def poll(self):
        if self._poll_error is not None:
            raise self._poll_error
        return self._poll_result

    def kill(self):
        self.killed = True
        if self._kill_error is not None:
            raise self._kill_error

    def terminate(self):
        self.terminated = True
        if self._terminate_error is not None:
            raise self._terminate_error

    def wait(self, timeout=None):
        if self._wait_error is not None:
            raise self._wait_error
        return 0


class TestAskingWhetherItHasExited:

    def test_an_exit_code_means_it_has_gone(self):
        assert P._process_has_exited(_Proc(poll_result=0)) is True

    def test_no_exit_code_means_it_is_still_running(self):
        assert P._process_has_exited(_Proc(poll_result=None)) is False

    def test_a_poll_that_raises_is_read_as_still_running(self):
        """THE UNCOVERED GUARD, and the direction matters.

        An unanswerable question is not permission to forget the child.
        Answering "still live" keeps its handle registered for a retry;
        answering "gone" would leak the process and the reader with it.
        """
        assert P._process_has_exited(
            _Proc(poll_error=OSError("no such process handle"))) is False


class TestKillingAndReaping:

    def test_a_clean_kill_is_confirmed(self):
        proc = _Proc(poll_result=None)
        assert P._kill_and_reap(proc) is True
        assert proc.killed is True

    def test_a_kill_that_fails_falls_back_to_asking(self):
        """THE UNCOVERED FALLBACK.

        The signal did not go, so nothing is known -- except what poll
        says. A child that had already exited is still confirmed; one
        that may be alive is not.
        """
        gone = _Proc(kill_error=PermissionError("not permitted"),
                     poll_result=0)
        assert P._kill_and_reap(gone) is True

        alive = _Proc(kill_error=PermissionError("not permitted"),
                      poll_result=None)
        assert P._kill_and_reap(alive) is False

    def test_a_wait_that_times_out_falls_back_to_asking(self):
        """THE OTHER UNCOVERED FALLBACK.

        The kill went out and the child has not been reaped inside the
        budget. A timeout is not a death certificate.
        """
        stubborn = _Proc(wait_error=subprocess.TimeoutExpired("cmd", 1),
                         poll_result=None)
        assert P._kill_and_reap(stubborn) is False

        raced = _Proc(wait_error=subprocess.TimeoutExpired("cmd", 1),
                      poll_result=0)
        assert P._kill_and_reap(raced) is True


class TestTerminatingAndReaping:
    """Returns `(requested, confirmed_exited)` -- two different questions."""

    def test_a_child_already_gone_is_not_signalled_again(self):
        proc = _Proc(poll_result=0)
        assert P._terminate_and_reap(proc) == (False, True)
        assert proc.terminated is False, (
            "a signal was sent to a child that had already exited")

    def test_a_running_child_is_asked_to_stop_and_confirmed(self):
        proc = _Proc(poll_result=None)
        requested, confirmed = P._terminate_and_reap(proc)
        assert requested is True and confirmed is True
        assert proc.terminated is True

    def test_known_running_skips_the_question_and_signals(self):
        """The caller already knows; asking again is a wasted syscall."""
        proc = _Proc(poll_result=0)
        requested, _confirmed = P._terminate_and_reap(proc,
                                                      known_running=True)
        assert proc.terminated is True
        assert requested is True

    def test_a_signal_that_fails_is_not_evidence_of_death(self):
        """The docstring's own rule, asserted.

        `requested` is False because the signal did not go, and
        `confirmed` is whatever poll can still say -- not an assumption.
        """
        alive = _Proc(poll_result=None,
                      terminate_error=PermissionError("not permitted"))
        assert P._terminate_and_reap(alive, known_running=True) == (False,
                                                                    False)

        gone = _Proc(poll_result=0,
                     terminate_error=PermissionError("not permitted"))
        assert P._terminate_and_reap(gone, known_running=True) == (False, True)
