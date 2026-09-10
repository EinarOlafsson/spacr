"""What the fast renderer decides before it draws anything.

Two decisions that are invisible in the finished figure and wrong only on
somebody else's machine: whether to force Qt offscreen (only when there is no
display to use) and whether a destination needs a directory made for it (only
when it names one).
"""
from __future__ import annotations

import os

import pytest

from spacr.figures import fast_render as fr


# ---------------------------------------------------------------------------
# starting Qt
# ---------------------------------------------------------------------------

class _RunningApplication:
    """Enough of a QApplication for the module -- and for pytest-qt, which
    calls ``processEvents`` on whatever ``instance()`` hands back."""

    def processEvents(self):                    # noqa: N802 (Qt naming)
        return None


class _StandInApplication:
    """A QApplication that is already running, so nothing has to be started."""

    running = _RunningApplication()

    @classmethod
    def instance(cls):
        return cls.running


def test_a_machine_with_a_display_is_not_pushed_offscreen(monkeypatch):
    """``QT_QPA_PLATFORM=offscreen`` is a default for a headless box. Setting
    it where there IS a display would send a user's own interactive figure to
    a buffer they never see, so it is conditional on the display variables."""
    pytest.importorskip("pyqtgraph")
    qtwidgets = pytest.importorskip("PySide6.QtWidgets")

    # The QApplication class is imported inside the function, so the stand-in
    # goes on the module it is imported from. `qt_application` is patched
    # because a live QApplication (pytest-qt starts one for the Qt suite)
    # would skip the whole branch under test.
    monkeypatch.setattr(qtwidgets, "QApplication", _StandInApplication)
    monkeypatch.setattr(fr, "qt_application", lambda: None)
    monkeypatch.setattr(fr, "_APPLICATION", None)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")

    ok, why = fr._pyqtgraph_ready(create=True)

    assert (ok, why) == (True, "")
    assert fr._APPLICATION is _StandInApplication.running
    assert "QT_QPA_PLATFORM" not in os.environ, (
        "a machine with a display was pushed offscreen anyway")

    # Take the display away and the same call does install the default --
    # so the absence above is a decision and not a line that never ran.
    monkeypatch.delenv("DISPLAY")
    monkeypatch.setattr(fr, "_APPLICATION", None)

    ok, why = fr._pyqtgraph_ready(create=True)

    assert (ok, why) == (True, "")
    assert os.environ["QT_QPA_PLATFORM"] == "offscreen"


# ---------------------------------------------------------------------------
# writing the file
# ---------------------------------------------------------------------------

class _Plot:
    """A scene that records where it was asked to export itself."""

    def __init__(self, written=None):
        self.asked = []
        self._written = written

    def export(self, destination):
        self.asked.append(destination)
        return self._written


def test_a_bare_filename_has_no_directory_to_make(tmp_path, monkeypatch):
    """``os.makedirs("")`` raises FileNotFoundError, so a destination in the
    working directory has to skip the call rather than pass it an empty path.

    Driven at ``_render_with_pyqtgraph`` because the public entry point runs
    every destination through :func:`figure_path`, which always returns a
    path with a directory in it -- the bare name only arrives here.
    """
    monkeypatch.chdir(tmp_path)
    plot = _Plot()

    panel = fr._render_with_pyqtgraph("volcano", None, "volcano.png",
                                      plot=plot, announce=False)

    assert panel.drawn and panel.path == "volcano.png"
    assert plot.asked == ["volcano.png"]
    assert list(tmp_path.iterdir()) == [], "an empty directory name was made"

    # A destination that does name a folder gets the folder made for it.
    nested = tmp_path / "figures" / "volcano.png"
    panel = fr._render_with_pyqtgraph("volcano", None, str(nested),
                                      plot=plot, announce=False)

    assert panel.drawn and panel.path == str(nested)
    assert (tmp_path / "figures").is_dir()


# ---------------------------------------------------------------------------
# One guard in this module cannot be made to fire, and is left standing
# rather than silenced. Written down here so the next reader does not spend
# the afternoon looking for an input that reaches it.
#
# `build_fast_plot`, the fall-through past the last `elif` in the panel
# chain (straight to `return plot` without any branch having run). The
# function opens with `if key not in FAST_PANELS: raise KeyError`, and
# FAST_PANELS has seven keys -- volcano, effect_rank, effect_distribution,
# controls, agreement, p_histogram, qq. The chain tests every one of them:
# volcano, effect_rank, effect_distribution, ("p_histogram", "qq"),
# controls, agreement. A key that survives the membership check therefore
# always matches a branch, and the fall-through arc is unreachable while
# the two lists agree. Adding a panel to FAST_PANELS without a branch here
# would make it reachable -- and would silently draw an empty scene, which
# is what the fall-through is for.
