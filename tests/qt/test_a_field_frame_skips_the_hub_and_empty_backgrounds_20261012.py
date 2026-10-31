"""A field frame is painted past the event hub and past empty backgrounds.

With card compositing off (the default since 02540ce97) Qt paints every
widget above the animated field on every field frame: 163 on Home and 179 on
Make Masks at 2560x1440. Profiled on 2026-10-12 (offscreen, cheapest
GUI-thread CPU of a synchronous field frame, median of six interleaved
rounds), two costs of that pass paid for nothing:

* The application event hub (:mod:`spacr.qt.gil_priority`) was asked about
  every paint event, a crossing into Python each, though no watcher reads a
  paint: 0.65 ms of a 4.8 ms Home frame and 0.67 ms of a 6.8 ms Mask frame.
  The hub now steps off the application while the update request that
  paints a window is delivered.
* 173 of Home's 184 style-sheet backgrounds and 100 of Mask's 117 painted no
  pixel, and Qt opened a painter and asked the style for each: 0.48 ms on
  either screen. :class:`spacr.qt.widgets.ambient._QuietBackgrounds` clears
  ``WA_StyledBackground`` on those once the window settles; the style sheet
  gives it back on any restyle that finds a background to draw.

Both leave the picture as it was; these tests pin that, the hub's delivery
contract, and the hand-back on every change that can give a quieted
background something to paint.
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject                    # noqa: E402
from PySide6.QtWidgets import QFrame, QLabel, QWidget         # noqa: E402

from spacr.qt.gil_priority import _application_event_hub_class  # noqa: E402
from spacr.qt.widgets import ambient                          # noqa: E402
from spacr.qt.widgets.ambient import (                        # noqa: E402
    _QUIET_RECHECK_TICKS, _QUIET_SETTLE_TICKS, _QuietBackgrounds,
    _hand_backgrounds_back,
    _paints_no_background, _restyled_by_its_sheet)

STYLED = ambient.Qt.WA_StyledBackground


class _Counting(QLabel):
    """A label that counts its paints."""

    def __init__(self, *args):
        super().__init__(*args)
        self.paints = 0

    def paintEvent(self, event):                              # noqa: N802
        self.paints += 1
        super().paintEvent(event)


class _Recorder(QObject):
    """Records the types of the events it is asked about."""

    def __init__(self, answer=False):
        super().__init__()
        self.kinds = []
        self.answer = answer

    def eventFilter(self, watched, event):                    # noqa: N802
        self.kinds.append(event.type())
        return self.answer


def _calls_to(code, action, owner=None):
    """How many times ``code`` runs while ``action()`` does.

    With ``owner``, only the calls whose ``self`` is ``owner`` count. A hub
    an earlier test left on the shared application is asked about the same
    events as the one under test, and is no measure of it.
    """
    calls = [0]

    def profile(frame, event, _arg):
        if (event == "call" and frame.f_code is code
                and (owner is None or frame.f_locals.get("self") is owner)):
            calls[0] += 1

    sys.setprofile(profile)
    try:
        action()
    finally:
        sys.setprofile(None)
    return calls[0]


@pytest.fixture
def hub(qapp):
    """A hub of its own, installed on the application as the real one is."""
    made = _application_event_hub_class()(qapp)
    qapp.installEventFilter(made)
    yield made
    qapp.removeEventFilter(made)
    made.setParent(None)
    made.deleteLater()


@pytest.fixture
def painted(qtbot):
    """A shown window of labels, one of which counts its paints."""
    window = QWidget()
    qtbot.addWidget(window)
    window.resize(240, 160)
    labels = [_Counting(f"label {index}", window) for index in range(6)]
    for index, label in enumerate(labels):
        label.setGeometry(10, 10 + 22 * index, 200, 20)
    with qtbot.waitExposed(window):
        window.show()
    return window, labels


def test_a_repaint_asks_the_hub_once_not_once_a_widget(hub, painted):
    window, labels = painted
    code = type(hub).eventFilter.__code__
    assert hub._paints_unheard
    past = _calls_to(code, window.repaint, hub)
    hub._paints_unheard = False
    through = _calls_to(code, window.repaint, hub)
    assert past == 1
    assert through > len(labels)


def test_the_window_paints_once_and_its_own_filter_hears_the_request(
        hub, painted, qapp):
    window, labels = painted
    heard = _Recorder()
    window.installEventFilter(heard)
    try:
        before = labels[0].paints
        window.repaint()
        assert labels[0].paints == before + 1
        assert heard.kinds.count(QEvent.Type.UpdateRequest) == 1
    finally:
        window.removeEventFilter(heard)


def test_the_hub_is_back_on_the_application_after_a_frame(hub, painted, qapp):
    window, _labels = painted
    watcher = _Recorder()
    hub.add(watcher, (QEvent.Type.User,))
    window.repaint()
    qapp.sendEvent(window, QEvent(QEvent.Type.User))
    assert watcher.kinds == [QEvent.Type.User]


@pytest.mark.parametrize("kind", [QEvent.Type.Paint, QEvent.Type.Move,
                                  QEvent.Type.Resize, QEvent.Type.UpdateRequest,
                                  QEvent.Type.UpdateLater])
def test_a_watcher_of_what_a_repaint_sends_keeps_the_hub_on(hub, painted,
                                                             kind):
    window, _labels = painted
    watcher = _Recorder()
    hub.add(watcher, (kind,))
    assert not hub._paints_unheard
    if kind == QEvent.Type.Paint:
        window.repaint()
        assert QEvent.Type.Paint in watcher.kinds
    hub.discard(watcher)
    assert hub._paints_unheard


def test_a_hub_on_no_application_lets_qt_deliver_the_request(qapp):
    orphan = _application_event_hub_class()()
    widget = QWidget()
    try:
        assert orphan.eventFilter(
            widget, QEvent(QEvent.Type.UpdateRequest)) is False
    finally:
        orphan.deleteLater()


def test_a_request_for_something_not_a_widget_goes_the_usual_way(hub, qapp):
    watcher = _Recorder(answer=True)
    hub.add(watcher, (QEvent.Type.User,))
    assert hub._paints_unheard
    target = QObject()
    assert hub.eventFilter(target, QEvent(QEvent.Type.UpdateRequest)) is False
    assert watcher.kinds == []


SHEET = """
QWidget#QuietHost { background: rgb(12, 24, 36); }
QWidget#Clear { background: transparent; }
QLabel#Inked { background: rgba(200, 0, 0, 120); }
QLabel#Hovered { background: transparent; }
QLabel#Hovered:hover { background: rgb(0, 0, 200); }
QFrame#Disabled { background: transparent; }
QFrame#Disabled:disabled { background: rgb(0, 160, 0); }
"""


class _OwnBackground(QWidget):
    """A subclass that sets the attribute itself, as `mark_surface` does."""


@pytest.fixture
def host(qtbot, qapp):
    """A styled window over a stand-in field, and its quiet backgrounds."""
    from spacr.qt.gil_priority import _application_event_hub

    _application_event_hub(qapp)
    window = QWidget()
    qtbot.addWidget(window)
    window.setObjectName("QuietHost")
    window.setStyleSheet(SHEET)
    window.resize(320, 240)
    field = QWidget(window)
    field.setGeometry(0, 0, 320, 240)
    parts = {}
    for name, kind, at in (("Clear", QWidget, (10, 10)),
                           ("Clear", QLabel, (10, 50)),
                           ("Inked", QLabel, (10, 90)),
                           ("Hovered", QLabel, (10, 130)),
                           ("Disabled", QFrame, (170, 10)),
                           ("Clear", _OwnBackground, (170, 50))):
        widget = kind(window)
        widget.setObjectName(name)
        if isinstance(widget, QLabel):
            widget.setText(name)
        widget.setGeometry(at[0], at[1], 140, 30)
        parts[f"{name}/{kind.__name__}"] = widget
    parts["Clear/_OwnBackground"].setAttribute(STYLED, True)
    with qtbot.waitExposed(window):
        window.show()
    quiet = _QuietBackgrounds(field)
    yield window, field, quiet, parts
    _hand_backgrounds_back(quiet._quieted)


def _settle(quiet):
    for _tick in range(_QUIET_SETTLE_TICKS + 1):
        quiet.tick()
    assert quiet._settled


def _pixels(window):
    return window.grab().toImage()


def test_only_backgrounds_that_paint_nothing_are_quieted(host):
    window, field, quiet, parts = host
    assert not quiet._settled
    _settle(quiet)
    styled = {name: widget.testAttribute(STYLED)
              for name, widget in parts.items()}
    assert styled == {"Clear/QWidget": False, "Clear/QLabel": False,
                      "Inked/QLabel": True, "Hovered/QLabel": True,
                      "Disabled/QFrame": True, "Clear/_OwnBackground": True}
    assert set(quiet._quieted.values()) == {parts["Clear/QWidget"],
                                            parts["Clear/QLabel"]}
    assert window.testAttribute(STYLED)


def test_the_picture_is_the_same_quieted_or_not(host, qtbot):
    window, _field, quiet, _parts = host
    before = _pixels(window)
    _settle(quiet)
    assert quiet._quieted
    assert _pixels(window) == before
    _hand_backgrounds_back(quiet._quieted)
    assert _pixels(window) == before


def _recheck(quiet):
    for _tick in range(_QUIET_RECHECK_TICKS):
        quiet.tick()


def test_a_settled_window_is_not_looked_at_again(host, monkeypatch):
    _window, _field, quiet, _parts = host
    _settle(quiet)
    looked = []
    monkeypatch.setattr(quiet, "_quiet_the_window",
                        lambda: looked.append(1))
    for _round in range(3):
        _recheck(quiet)
    assert quiet._settled
    assert looked == []


def _inside(window, widget):
    return _pixels(window).pixelColor(widget.geometry().center()).getRgb()[:3]


def test_a_restyle_gives_the_background_back_before_it_paints(host):
    window, _field, quiet, parts = host
    _settle(quiet)
    clear = parts["Clear/QWidget"]
    clear.setStyleSheet("background: rgb(0, 200, 0);")
    assert clear.testAttribute(STYLED)
    assert _inside(window, clear) == (0, 200, 0)
    assert quiet._settled
    _recheck(quiet)
    assert not quiet._settled
    clear.setStyleSheet("background: transparent;")
    _settle(quiet)
    assert not clear.testAttribute(STYLED)
    assert id(clear) in quiet._quieted


def test_a_property_restyle_gives_the_background_back(host):
    window, _field, quiet, parts = host
    window.setStyleSheet(
        SHEET + 'QLabel#Clear[spacrQuietProbe="true"] '
                '{ background: rgb(0, 0, 220); }')
    _settle(quiet)
    label = parts["Clear/QLabel"]
    assert not label.testAttribute(STYLED)
    label.setProperty("spacrQuietProbe", True)
    label.style().unpolish(label)
    label.style().polish(label)
    assert label.testAttribute(STYLED)
    assert _inside(window, label) == (0, 0, 220)


def test_a_theme_change_gives_every_background_back_that_it_fills(host):
    window, _field, quiet, parts = host
    _settle(quiet)
    window.setStyleSheet(SHEET.replace(
        "QWidget#Clear { background: transparent; }",
        "QWidget#Clear { background: rgb(90, 0, 90); }"))
    for name in ("Clear/QWidget", "Clear/QLabel"):
        assert parts[name].testAttribute(STYLED)
        assert _inside(window, parts[name]) == (90, 0, 90)


def test_a_disabled_state_that_paints_is_never_quieted(host):
    window, _field, quiet, parts = host
    _settle(quiet)
    frame = parts["Disabled/QFrame"]
    assert frame.testAttribute(STYLED)
    frame.setEnabled(False)
    assert _inside(window, frame) == (0, 160, 0)


def test_a_parent_change_hands_the_background_back(host):
    _window, _field, quiet, parts = host
    _settle(quiet)
    label = parts["Clear/QLabel"]
    assert not label.testAttribute(STYLED)
    label.setParent(parts["Disabled/QFrame"])
    assert label.testAttribute(STYLED)
    assert id(label) not in quiet._quieted
    assert not quiet._settled


def test_a_parent_change_of_a_widget_it_did_not_quiet_changes_nothing(host):
    _window, _field, quiet, parts = host
    _settle(quiet)
    parts["Inked/QLabel"].setParent(parts["Disabled/QFrame"])
    assert quiet._settled


def test_a_quieted_widget_that_goes_away_has_the_window_looked_at_again(
        host):
    from shiboken6 import delete

    _window, _field, quiet, parts = host
    _settle(quiet)
    delete(parts["Clear/QWidget"])
    _recheck(quiet)
    assert not quiet._settled


def test_a_widget_shown_in_the_window_has_it_looked_at_again(host, qtbot):
    window, _field, quiet, _parts = host
    _settle(quiet)
    other = QWidget()
    qtbot.addWidget(other)
    other.show()
    assert quiet._settled
    late = QWidget(window)
    late.setObjectName("Clear")
    late.setGeometry(170, 130, 140, 30)
    late.show()
    assert not quiet._settled
    _settle(quiet)
    assert not late.testAttribute(STYLED)


def test_a_field_going_away_hands_every_background_back(host, qapp):
    from shiboken6 import delete

    _window, field, quiet, parts = host
    _settle(quiet)
    gone = parts["Clear/QWidget"]
    delete(gone)
    label = parts["Clear/QLabel"]
    delete(field)
    assert label.testAttribute(STYLED)


def test_a_quieted_widget_that_was_deleted_is_forgotten(host):
    from shiboken6 import delete

    _window, _field, quiet, parts = host
    _settle(quiet)
    gone = parts["Clear/QWidget"]
    key = id(gone)
    delete(gone)
    quiet._quiet_the_window()
    assert key not in quiet._quieted
    assert id(parts["Clear/QLabel"]) in quiet._quieted


def test_a_field_that_is_its_own_window_quiets_nothing(qtbot, qapp):
    alone = QWidget()
    qtbot.addWidget(alone)
    alone.setObjectName("Clear")
    alone.setStyleSheet("QWidget#Clear { background: transparent; }")
    alone.resize(50, 50)
    alone.show()
    quiet = _QuietBackgrounds(alone)
    _settle(quiet)
    assert quiet._quieted == {}


def test_a_widget_with_no_area_is_not_called_empty(qapp):
    flat = QWidget()
    flat.resize(0, 0)
    assert _paints_no_background(flat) is False


def test_the_sheet_sets_the_attribute_again_only_on_the_classes_it_lists(
        qapp):
    assert _restyled_by_its_sheet(QWidget())
    assert _restyled_by_its_sheet(QLabel())
    assert not _restyled_by_its_sheet(_OwnBackground())


def test_a_field_tick_counts_toward_quieting_the_window(qtbot, qapp):
    widget = ambient.AmbientWidget(fps=24, seed=1)
    qtbot.addWidget(widget)
    quiet = widget._quiet_backgrounds
    before = quiet._quiet
    widget._on_tick()
    assert quiet._quiet == before + 1
