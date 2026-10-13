"""AppScreen's module-level helpers when what they read has gone or is idle.

Pins the behaviour of the small helpers around the screen that are only
exercised when a collaborator has been destroyed under them (a window, a
Live switch, a timer, a widget whose C++ half is gone), and of the idle
pre-builder that builds waiting settings categories between user input:

* the hint-strip fitter hands empty text back untouched;
* ``_window_of`` / ``_live_is_on`` answer "none" / "off" rather than raise;
* ``_show_the_src_live`` survives a dead debounce timer and re-primes a
  non-live preview card whose switch is off;
* the late-caption translator skips a child that dies while it is asked;
* ``_IdlePrebuild`` stops when there is nothing to build or the screen is
  hidden or gone, waits after pointer and key input, and builds a waiting
  category step by step once the user is idle;
* ``_BuiltOnFirstUse`` answers class access with itself and ``del`` like a
  plain attribute.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QLabel, QWidget  # noqa: E402

from spacr.qt.screens import app_screen as aps  # noqa: E402

pytestmark = pytest.mark.qt


class _Gone:
    """Any call on it raises as a destroyed Qt object does."""

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Internal C++ object already deleted.")


class _DeadSwitch:
    def isChecked(self):                                     # noqa: N802
        raise RuntimeError("Internal C++ object already deleted.")


class _Switch:
    def __init__(self, checked=False):
        self.checked = checked

    def isChecked(self):                                     # noqa: N802
        return self.checked

    def setChecked(self, value):                             # noqa: N802
        self.checked = bool(value)


class _DeadTimer:
    def __init__(self):
        self.asked = 0

    def stop(self):
        self.asked += 1
        raise RuntimeError("Internal C++ object already deleted.")


# --------------------------------------------------------------------------
# the small helpers


def test_an_empty_hint_comes_back_empty(qtbot):
    label = QLabel()
    qtbot.addWidget(label)
    label.resize(300, 40)
    assert aps._fit_to_lines("", label, 4) == ""
    assert aps._fit_to_lines("   ", label, 4) == ""


def test_a_label_with_no_width_keeps_the_whole_text(qtbot):
    label = QLabel()
    qtbot.addWidget(label)
    label.resize(0, 40)
    text = "a long hint " * 40
    assert aps._fit_to_lines(text, label, 2) == " ".join(text.split())


def test_a_window_that_has_gone_is_no_window():
    screen = types.SimpleNamespace(window=_Gone())
    assert aps._window_of(screen) is None


def test_a_live_switch_that_has_gone_reads_as_off():
    screen = types.SimpleNamespace(_preview_switch=_DeadSwitch())
    assert aps._live_is_on(screen) is False
    assert aps._live_is_on(types.SimpleNamespace(
        _preview_switch=_Switch(True))) is True


def test_a_dead_debounce_timer_does_not_stop_the_src_reaching_a_card():
    timer = _DeadTimer()
    screen = types.SimpleNamespace(
        _live_src_timer=timer,
        _preview_switch=_Switch(False),
        _settings_src_path=lambda: "/data/plate",
        _preview_card_attr="_measure_preview_card",
        _preview_primed=True,
    )
    aps._show_the_src_live(screen)
    assert timer.asked == 1
    assert screen._preview_primed is False, (
        "a preview card that is off must forget its prime so it reloads "
        "the new src when opened")


def test_an_empty_setting_key_has_no_animation():
    assert aps._setting_has_an_animation("") is False
    assert aps._setting_has_an_animation(None) is False


def test_a_child_that_dies_while_asked_is_skipped(qtbot):
    class _Dying(QWidget):
        def property(self, name):                            # noqa: A003
            raise RuntimeError("Internal C++ object already deleted.")

    alive = QWidget()
    dying = _Dying()
    qtbot.addWidget(alive)
    qtbot.addWidget(dying)
    host = types.SimpleNamespace(children=lambda: [dying, alive])
    translator = aps._LateCaptionTranslator()
    seen = []
    translator._on_arrival = seen.append
    translator._on_arrivals_in(host)
    assert seen == [alive]
    assert alive.property(aps._LateCaptionTranslator._HANDLED) is True


# --------------------------------------------------------------------------
# _IdlePrebuild


class _Screen(QWidget):
    """Just the surface ``_IdlePrebuild`` reads."""

    def __init__(self, sections=(), steps=1):
        super().__init__()
        self.waiting = list(sections)
        self.steps = {s: steps for s in sections}
        self.built = []
        self.gone = False
        self.after_step = None

    def isVisible(self):                                     # noqa: N802
        if self.gone:
            raise RuntimeError("Internal C++ object already deleted.")
        return super().isVisible()

    def rendered_settings_sections(self):
        return list(self.steps)

    def _heading_is_waiting(self, section):
        return section in self.waiting

    def _run_a_step_of(self, section):
        self.built.append(section)
        self.steps[section] -= 1
        if self.steps[section] <= 0:
            self.waiting.remove(section)
        if self.after_step is not None:
            self.after_step()
        return self.steps[section] > 0


@pytest.fixture
def idle(qtbot):
    made = []

    def make(sections=(), steps=1):
        screen = _Screen(sections, steps)
        qtbot.addWidget(screen)
        prebuild = aps._IdlePrebuild(screen)
        made.append(prebuild)
        return screen, prebuild

    yield make
    for prebuild in made:
        prebuild.stop()


def test_nothing_to_build_means_it_does_not_start(idle):
    screen, prebuild = idle()
    prebuild.resume()
    assert not prebuild._timer.isActive()
    assert prebuild._watching is False


def test_a_form_that_has_gone_leaves_nothing_to_build(idle):
    screen, prebuild = idle(["Advanced"])

    def gone():
        raise RuntimeError("Internal C++ object already deleted.")

    screen.rendered_settings_sections = gone
    assert prebuild._work_left() == []
    prebuild.resume()
    assert not prebuild._timer.isActive()


def test_a_waiting_heading_starts_the_idle_wait(idle):
    screen, prebuild = idle(["Advanced"])
    prebuild.resume()
    assert prebuild._timer.isActive()
    assert prebuild._timer.isSingleShot()
    assert prebuild._timer.interval() == prebuild.IDLE_MS


def test_a_key_waits_and_keeps_watching_a_click_waits_unwatched(idle):
    screen, prebuild = idle(["Advanced"])
    key = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A,
                    Qt.KeyboardModifier.NoModifier, "a")
    prebuild._watching = True
    stopped = []
    import spacr.qt.gil_priority as gp
    real = gp._stop_watching_application_events
    try:
        gp._stop_watching_application_events = (
            lambda app, watcher: stopped.append(watcher) or True)
        assert prebuild.eventFilter(screen, key) is False
        assert prebuild._timer.isActive()
        assert prebuild._watching is True and stopped == []
        click = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(1, 1),
                            QPointF(1, 1), Qt.MouseButton.LeftButton,
                            Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
        assert prebuild.eventFilter(screen, click) is False
        assert stopped == [prebuild]
        assert prebuild._watching is False
    finally:
        gp._stop_watching_application_events = real


def test_an_event_that_is_not_input_is_ignored(idle):
    screen, prebuild = idle(["Advanced"])
    assert prebuild.eventFilter(screen, QEvent(QEvent.Type.Paint)) is False
    assert not prebuild._timer.isActive()


def test_a_hidden_screen_stops_the_build(idle):
    screen, prebuild = idle(["Advanced"])
    prebuild.resume()
    prebuild._slice()
    assert not prebuild._timer.isActive()
    assert screen.built == []


def test_a_screen_that_has_gone_stops_the_build(idle):
    screen, prebuild = idle(["Advanced"])
    prebuild.resume()
    screen.gone = True
    prebuild._slice()
    assert not prebuild._timer.isActive()
    assert screen.built == []


def test_without_an_application_the_slice_does_nothing(idle, monkeypatch):
    screen, prebuild = idle(["Advanced"])
    screen.show()
    prebuild._pointer = (1, 1)
    prebuild._pointer_now = lambda: (1, 1)
    monkeypatch.setattr(aps, "QApplication",
                        types.SimpleNamespace(instance=lambda: None))
    prebuild._slice()
    assert prebuild._watching is False
    assert screen.built == []


def test_an_idle_screen_builds_its_waiting_headings(idle, qtbot):
    screen, prebuild = idle(["Advanced", "Output"], steps=2)
    screen.show()
    prebuild._pointer = (7, 7)
    prebuild._pointer_now = lambda: (7, 7)
    prebuild._slice()
    assert prebuild._watching is True
    assert screen.built[:1] == ["Advanced"]
    assert prebuild.slices_ms and prebuild.slices_ms[0] >= 0
    qtbot.waitUntil(lambda: not screen.waiting, timeout=3000)
    assert set(screen.built) == {"Advanced", "Output"}
    assert not prebuild._timer.isActive()
    assert prebuild._watching is False


def test_a_step_that_stops_the_build_is_not_followed(idle):
    screen, prebuild = idle(["Advanced", "Output"], steps=1)
    screen.show()
    prebuild._pointer = (3, 3)
    prebuild._pointer_now = lambda: (3, 3)
    screen.after_step = prebuild.stop
    prebuild._slice()
    assert screen.built == ["Advanced"]
    assert "Output" in screen.waiting
    assert not prebuild._timer.isActive()


def test_the_pointer_moving_puts_the_build_off(idle):
    screen, prebuild = idle(["Advanced"])
    screen.show()
    prebuild._pointer = (1, 1)
    prebuild._pointer_now = lambda: (9, 9)
    prebuild._slice()
    assert screen.built == []
    assert prebuild._timer.isActive()


# --------------------------------------------------------------------------
# _BuiltOnFirstUse


def test_the_class_reads_the_descriptor_itself():
    assert isinstance(aps.AppScreen._results_panel, aps._BuiltOnFirstUse)
    assert aps.AppScreen._results_panel.name == "_results_panel"


def test_deleting_an_unset_built_attribute_raises_attribute_error():
    class _Holder:
        panel = aps._BuiltOnFirstUse("part")

    holder = _Holder()
    with pytest.raises(AttributeError, match="panel"):
        del holder.panel
    holder.panel = "built"
    assert holder.panel == "built"
    del holder.panel
    with pytest.raises(AttributeError):
        _ = holder.panel
