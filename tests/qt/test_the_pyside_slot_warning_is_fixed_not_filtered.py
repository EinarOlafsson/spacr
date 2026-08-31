"""The libpyside slot warning is REPAIRED, and must not be filtered again.

Opening any module printed, once per screen built:

    libpyside: addMetaMethod: Cannot add dynamic method "_on_tick()" (2)
    to QWidget/0x7feab2c18060: No Wrapper found.

For three sessions that was read as cosmetic, because the connection it
names really does still fire. It was not cosmetic. ``QEvent.ChildAdded``
is delivered synchronously from inside the child's C++ constructor, before
Shiboken has registered the wrapper for the Python class being built, and
``_LateCaptionTranslator`` called ``event.child()`` there and HELD the
result for a turn. That minted a bare ``QWidget`` wrapper which displaced
the real one permanently, and a child with no wrapper loses its dynamic
metaobject AND its Python overrides of Qt virtuals. ``AmbientWidget``
starts its timer from ``showEvent``, so the backdrop never animated.

These tests pin the repair from both ends: the cause cannot come back, and
the warning must not be quietly filtered if it ever does.

``tools/diagnose_pyside_slot_warning.py`` Part 5 is the record, and
reproduces both the breakage and the fix without any spaCR code.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject, QTimer, Signal
from PySide6.QtWidgets import QWidget

from spacr.qt import _QT_NOISE

#: Verbatim from the macOS report, both the anonymous widget and the named
#: ones. These must now REACH the console: each one is a live report that
#: some widget has just been stripped of its wrapper.
REPORTED = (
    'libpyside: addMetaMethod: Cannot add dynamic method "_on_tick()" (2) '
    'to QWidget/0x7feab2c18060: No Wrapper found.',
    'libpyside: addMetaMethod: Cannot add dynamic method '
    '"_on_use_offered()" (2) to QWidget/"ChainingBar": No Wrapper found.',
    'libpyside: addMetaMethod: Cannot add dynamic method "refresh()" (2) '
    'to QWidget/"MeasureQCBanner": No Wrapper found.',
    'libpyside: addMetaMethod: Cannot add dynamic method '
    '"_on_measure_clicked()" (2) to QWidget/"DiameterPanel": '
    'No Wrapper found.',
)

#: Warnings this project has actually needed to read. Every one must
#: survive the filter. The first two are not hypothetical -- the thread
#: affinity line arrives immediately before a crash and is singled out
#: for a Python stack in the handler itself.
MUST_SURVIVE = (
    "QBasicTimer::start: Timers cannot be started from another thread",
    "QObject: Cannot create children for a parent that is in a different "
    "thread.",
    "QPixmap::scaled: Pixmap is a null pixmap",
    "Could not parse stylesheet of object QWidget",
    "QOpenGLShaderProgram: could not create shader program",
    "libpyside: Invalid return value in function Foo, expected bar",
)


@pytest.mark.parametrize("message", REPORTED)
def test_the_warning_is_no_longer_swallowed(message):
    """It reports a real defect, so it has to be audible.

    Filtering it was defensible only while the cause was unknown. Now that
    a poisoned child silently loses its virtual overrides, this line is the
    only notice anyone gets, and hiding it hides that.
    """
    assert not _QT_NOISE.search(message), (
        "this warning reports a widget that has lost its Shiboken wrapper "
        "-- fix the ChildAdded filter that caused it, do not filter this")


@pytest.mark.parametrize("message", MUST_SURVIVE)
def test_a_warning_that_matters_still_gets_through(message):
    """The remaining filter is a scalpel, not a mute."""
    assert not _QT_NOISE.search(message), f"wrongly swallowed: {message}"


def test_the_filter_still_only_covers_the_two_cosmetic_lines():
    """What is left is the font note and the plugin note, and nothing else."""
    assert _QT_NOISE.search("OpenType support missing for script 7")
    assert _QT_NOISE.search(
        "This plugin does not support propagateSizeHints")
    assert not _QT_NOISE.search("No Wrapper found.")
    assert not _QT_NOISE.search("libpyside: addMetaMethod: something else")


def _poisonable(qtbot):
    """A host watched by the real filter, and a child parented into it."""
    from spacr.qt.screens.app_screen import _LateCaptionTranslator

    class Child(QWidget):
        poked = Signal(int)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.shown = 0
            self.delivered = []
            self.poked.connect(self.delivered.append)

        def showEvent(self, event):                     # noqa: N802
            self.shown += 1
            super().showEvent(event)

    host = QWidget()
    qtbot.addWidget(host)
    watcher = _LateCaptionTranslator(host)
    host.installEventFilter(watcher)
    return host, watcher, Child(host)


def test_a_child_parented_into_a_watched_host_keeps_its_wrapper(qtbot):
    """The regression test for the cause, stated as what the user loses.

    Deliberately about the child's IDENTITY rather than about the absence
    of a log line: the warning is a symptom, and a future filter would make
    a symptom-based test pass while the widget stayed broken.
    """
    host, _watcher, child = _poisonable(qtbot)

    assert child.metaObject().className() == "Child", (
        "the child was stripped of its dynamic metaobject -- something "
        "called event.child() during ChildAdded and held the result")
    assert child.metaObject().indexOfSignal("poked(int)") != -1
    assert len(host.findChildren(type(child))) == 1


def test_a_watched_host_does_not_kill_its_childs_virtual_overrides(qtbot):
    """The consequence that actually cost a feature.

    A poisoned child's ``showEvent`` never runs, because Shiboken resolves
    the override through the wrapper. ``AmbientWidget`` starts its timer
    there, so the backdrop silently stopped animating.
    """
    host, _watcher, child = _poisonable(qtbot)
    host.show()
    child.show()
    qtbot.waitUntil(lambda: child.shown > 0, timeout=2000)


def test_a_watched_host_still_delivers_its_childs_own_signals(qtbot):
    """A signal on a poisoned child is a silent no-op, which is the worst
    shape a defect can take: nothing raises and nothing happens."""
    _host, _watcher, child = _poisonable(qtbot)
    child.poked.emit(7)
    assert child.delivered == [7]


def test_the_late_caption_filter_never_asks_for_the_arriving_child():
    """Stated against the source, because it is a rule about one call.

    The behavioural tests above are the real guard; this one names the
    forbidden call so that a reader who breaks it is told why rather than
    left to infer it from a metaobject assertion.
    """
    import dis

    from spacr.qt.screens.app_screen import _LateCaptionTranslator

    # Bytecode, not source: the docstring on that method has to be free to
    # name the call it is forbidding, and a source scan cannot tell the
    # prohibition from the violation.
    referenced = {instruction.argval for instruction
                  in dis.get_instructions(_LateCaptionTranslator.eventFilter)}
    assert "child" not in referenced, (
        "ChildAdded arrives from inside the child's C++ constructor; "
        "wrapping the child there and keeping it displaces its real "
        "Shiboken wrapper for the life of the process")


def test_the_diagnostic_that_found_the_cause_is_still_here():
    """The evidence stays with the decision.

    Part 5 is the reproduction, and it is what a future session will reach
    for when this warning reappears.
    """
    from pathlib import Path

    tool = (Path(__file__).resolve().parents[2]
            / "tools" / "diagnose_pyside_slot_warning.py")
    assert tool.is_file(), (
        "the diagnostic that found this is gone; it is the only "
        "reproduction of a bug that took three sessions to localise")
    body = tool.read_text()
    assert "addMetaMethod" in body
    assert "PART 5" in body, "the part that found the cause is gone"
