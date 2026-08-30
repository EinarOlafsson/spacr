"""``spacr.qt`` quietening the backdrop's logger on a machine that fights it.

:func:`spacr.qt._quiet_vispy_logging` turns three loggers down and then asks
vispy itself to do the same. Both halves are wrapped, and this file pins what
the wrapping is for: a logger that refuses to be reconfigured must not stop
the loggers after it being quietened, and a vispy that has no
``set_log_level`` -- an old one, or the module stub a machine without the
package leaves behind -- must not turn "quieten the backdrop" into an
ImportError out of application start-up.
"""
from __future__ import annotations

import logging
import sys
import types

import pytest

pytest.importorskip("PySide6")

from spacr import qt as spacr_qt

_LOGGERS = ("vispy", "vispy.gloo", "vispy.app")


@pytest.fixture(autouse=True)
def _restore_levels():
    """The three logger levels are process-wide state; put them back."""
    get_logger = logging.getLogger          # the real one, kept past any patch
    saved = {name: get_logger(name).level for name in _LOGGERS}
    yield
    for name, level in saved.items():
        get_logger(name).setLevel(level)


class _RefusesToBeConfigured:
    """A logger whose level cannot be set -- a stand-in for a handler-managed
    logger installed by something else in the process."""

    def setLevel(self, level):                      # noqa: N802 (logging name)
        raise RuntimeError("this logger's level is owned elsewhere")


def test_a_logger_that_refuses_does_not_keep_the_next_one_loud(monkeypatch):
    """The middle logger of the three throws; the third is still turned down.

    Without the ``continue`` the loop would abort on ``vispy.gloo`` and
    ``vispy.app`` -- the one that narrates a redraw -- would keep its level.
    """
    real_get_logger = logging.getLogger
    for name in _LOGGERS:
        real_get_logger(name).setLevel(logging.DEBUG)

    def refusing_get_logger(name=None):
        if name == "vispy.gloo":
            return _RefusesToBeConfigured()
        return real_get_logger(name)

    monkeypatch.setattr(logging, "getLogger", refusing_get_logger)

    spacr_qt._quiet_vispy_logging()

    assert real_get_logger("vispy").level == logging.ERROR
    assert real_get_logger("vispy.app").level == logging.ERROR, (
        "the loop stopped at the logger that refused")
    # The one that threw is untouched, which is what makes the other two
    # meaningful: the loop really did pass through a failure.
    assert real_get_logger("vispy.gloo").level == logging.DEBUG


def test_a_vispy_with_no_log_level_setter_still_leaves_the_loggers_quiet(
        monkeypatch):
    """`from vispy import set_log_level` on a vispy that has not got one is an
    ImportError, and it is swallowed -- but only after the loggers above it
    have already been turned down."""
    stub = types.ModuleType("vispy")
    monkeypatch.setitem(sys.modules, "vispy", stub)
    for name in _LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG)

    spacr_qt._quiet_vispy_logging()

    assert all(logging.getLogger(name).level == logging.ERROR
               for name in _LOGGERS)

    # And when vispy DOES have the setter it is called -- the same code path,
    # with the one thing that was missing put back.
    asked = []
    stub.set_log_level = asked.append
    spacr_qt._quiet_vispy_logging()
    assert asked == ["error"]
