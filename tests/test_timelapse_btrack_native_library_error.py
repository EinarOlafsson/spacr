"""btrack native-library failures reach the user as the readable ImportError.

btrack loads ``libtracker.so`` when a ``BayesianTracker`` is constructed, and
its debug wrapper re-raises the loader ``OSError`` as a bare ``Exception``
whose ``__cause__`` is that OSError. These tests drive
``spacr.timelapse._new_btrack_tracker`` with stand-in btrack modules, so they
run whether or not btrack (or its native library) works on this machine.
"""

from types import SimpleNamespace

import pytest

from spacr import timelapse

GLIBCXX = (
    "/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.32' not found"
)


def _fake_btrack(error):
    """A stand-in btrack module whose BayesianTracker raises ``error``."""

    def bayesian_tracker():
        """Raise the configured construction error."""
        raise error

    return SimpleNamespace(BayesianTracker=bayesian_tracker)


def _wrapped_os_error():
    """The bare Exception btrack's debug wrapper raises, chained to an OSError."""
    try:
        try:
            raise OSError(GLIBCXX)
        except OSError as inner:
            raise Exception from inner
    except Exception as outer:
        return outer


def test_a_wrapped_native_library_failure_becomes_the_readable_import_error():
    with pytest.raises(ImportError, match=r"spacr\[btrack\].*trackpy, Trackastra, or IoU") as info:
        timelapse._new_btrack_tracker(_fake_btrack(_wrapped_os_error()))
    assert isinstance(info.value.__cause__, OSError)
    assert "GLIBCXX" in str(info.value.__cause__)


def test_a_bare_os_error_becomes_the_readable_import_error():
    with pytest.raises(ImportError, match=r"spacr\[btrack\]"):
        timelapse._new_btrack_tracker(_fake_btrack(OSError(GLIBCXX)))


def test_the_message_matches_the_import_guard():
    assert str(timelapse._btrack_unavailable_error()) == (
        "btrack tracking requires the optional btrack dependency. "
        "Install it with `pip install 'spacr[btrack]'`, or select "
        "trackpy, Trackastra, or IoU tracking instead."
    )


def test_other_construction_errors_are_not_disguised():
    with pytest.raises(ValueError, match="bad config"):
        timelapse._new_btrack_tracker(_fake_btrack(ValueError("bad config")))


def test_a_working_tracker_is_returned_unentered():
    sentinel = object()
    fake = SimpleNamespace(BayesianTracker=lambda: sentinel)
    assert timelapse._new_btrack_tracker(fake) is sentinel
