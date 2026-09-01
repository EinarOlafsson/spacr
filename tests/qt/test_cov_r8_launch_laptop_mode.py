"""`launch` deciding laptop mode, and what happens when it cannot.

The guard carries a `# pragma: no cover`, which excludes nothing here --
`.coveragerc` sets `exclude_lines =` to an empty list -- so the two
lines were counted and untested.

The branch matters more than its size. Laptop mode reads battery and
thermal state to decide how hard spaCR may drive the machine, and that
is exactly the kind of probe that fails on hardware nobody tested:
no battery, no sysfs entry, a container. If deciding it can raise out of
`launch`, the application does not start at all -- on precisely the
machines where the probe is least likely to work.
"""
from __future__ import annotations

import pytest

from spacr.qt import app as app_mod

# The one stand-in QApplication that survives `launch` installing an
# event filter on it; borrowed rather than re-invented.
from tests.qt.test_cov_qt_app import launched  # noqa: F401

pytestmark = pytest.mark.qt


def test_a_laptop_probe_that_raises_does_not_stop_the_launch(
        launched, monkeypatch, caplog):
    """THE UNCOVERED PAIR.

    A machine with no battery and no thermal sysfs is where this probe
    fails, and it is a machine that still has to start spaCR.
    """
    from spacr.qt import laptop_mode

    asked = []

    def refuse():
        asked.append(True)
        raise RuntimeError("no battery on this machine")

    monkeypatch.setattr(laptop_mode, "describe", refuse)

    with caplog.at_level("DEBUG"):
        assert app_mod.launch([]) == 0

    # ASSERTED ON THE PROBE, NOT ON THE LOG.
    #
    # This used to check caplog for "could not decide laptop mode" and
    # failed whenever another launch had run first: `launch` configures
    # logging itself, so a second call raises the spacr.qt.app logger
    # above DEBUG and the line is never captured. The test was
    # order-dependent, and passed alone.
    #
    # What the test is actually about is that a probe which raises does
    # not stop the launch -- so it asserts the probe WAS called and the
    # launch still returned 0. That holds whatever logging is doing.
    assert asked == [True], "the laptop probe was never reached"


def test_an_apply_that_raises_is_survived_too(launched, monkeypatch):
    """`describe` succeeding and `apply` failing is the other order.

    Asserted on BEHAVIOUR rather than on the log line. `launch` installs
    spaCR's own logging, so a second launch in one session no longer
    reports through `caplog` -- an assertion on the message passes or
    fails depending on which test ran first, which is not a property of
    the code under test.

    What has to hold is that the probe ran and the launch still
    returned.
    """
    from spacr.qt import laptop_mode

    called = []
    monkeypatch.setattr(laptop_mode, "describe", lambda: "a description")

    def refuse():
        called.append("apply")
        raise OSError("cannot write the power governor")

    monkeypatch.setattr(laptop_mode, "apply", refuse)
    assert app_mod.launch([]) == 0, (
        "a power-governor write that failed stopped the application")
    assert called == ["apply"], "the branch under test never ran"


def test_a_laptop_mode_that_changed_something_says_what(launched,
                                                        monkeypatch):
    """The reported half, so the guard above is visibly a guard."""
    from spacr.qt import laptop_mode

    seen = []
    lines: list = []
    monkeypatch.setattr(laptop_mode, "describe", lambda: "on battery")
    monkeypatch.setattr(laptop_mode, "apply",
                        lambda: seen.append(1) or {
                            "changed": ["threads", "backdrop"]})
    monkeypatch.setattr(app_mod.LOG, "info",
                        lambda msg, *a, **k: lines.append(
                            msg % a if a else msg))
    assert app_mod.launch([]) == 0
    assert seen == [1]
    assert any("laptop mode changed" in line for line in lines)
    assert any("threads" in line for line in lines)


def test_a_laptop_mode_that_changed_nothing_stays_quiet(launched,
                                                        monkeypatch):
    """An empty `changed` list must not print an empty change line."""
    from spacr.qt import laptop_mode

    lines: list = []
    monkeypatch.setattr(laptop_mode, "describe", lambda: "on mains")
    monkeypatch.setattr(laptop_mode, "apply", lambda: {"changed": []})
    monkeypatch.setattr(app_mod.LOG, "info",
                        lambda msg, *a, **k: lines.append(msg % a if a
                                                          else msg))
    assert app_mod.launch([]) == 0
    assert not any("laptop mode changed" in line for line in lines)
