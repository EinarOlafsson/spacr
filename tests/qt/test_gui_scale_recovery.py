"""Scale changes survive closed listeners, callback failures and unsaved previews."""
from __future__ import annotations

import gc
import weakref

import pytest
from PySide6.QtWidgets import QLabel, QWidget, QHBoxLayout, QSplitter, QVBoxLayout

from spacr.qt import gui_scale
from spacr.qt import preferences as prefs
from spacr.qt.widgets import preview_scale as ps

from .test_the_gui_scale_fits_a_laptop import (
    _never_the_real_preferences,  # noqa: F401
    _the_layer_is_in_and_the_scale_goes_back,  # noqa: F401
    _panel,
)

pytestmark = pytest.mark.qt


def test_closed_and_failing_scale_listeners_do_not_block_live_listeners(monkeypatch):
    monkeypatch.setattr(gui_scale, "_LISTENERS", [])
    calls = []

    class Observer:
        def changed(self, scale):
            calls.append(("dead", scale))

    observer = Observer()
    reference = weakref.ref(observer)
    gui_scale.add_listener(observer.changed)
    del observer
    gc.collect()
    assert reference() is None

    def broken(scale):
        calls.append(("broken", scale))
        raise RuntimeError("closed panel")

    gui_scale.add_listener(broken)
    gui_scale.add_listener(lambda scale: calls.append(("live", scale)))
    assert gui_scale.set_gui_scale_live(0.65) == pytest.approx(0.65)
    assert calls == [("broken", 0.65), ("live", 0.65)]
    assert len(gui_scale._LISTENERS) == 2


def test_a_failed_window_refresh_does_not_stop_other_windows(qtbot):
    calls = []

    class GoodWindow(QWidget):
        def refresh_theme(self):
            calls.append("good")

    class BrokenWindow(QWidget):
        def refresh_theme(self):
            calls.append("broken")
            raise RuntimeError("unavailable drawing surface")

    good, broken = GoodWindow(), BrokenWindow()
    qtbot.addWidget(good)
    qtbot.addWidget(broken)
    assert gui_scale.refresh_the_windows() >= 1
    assert calls.count("good") == calls.count("broken") == 1


def test_closing_scale_question_restores_settings_even_if_callback_fails(qtbot, qt_theme_applied):
    answers, destroyed = [], []

    def notify(kept):
        answers.append(kept)
        raise RuntimeError("owner has closed")

    dialog = gui_scale.change_scales(gui=0.5, font=1.4, seconds=60,
                                    require_parent=False, on_done=notify)
    dialog.destroyed.connect(lambda: destroyed.append(True))
    dialog.close()
    qtbot.waitUntil(lambda: bool(destroyed))
    assert answers == [False]
    assert prefs.get_gui_scale() == pytest.approx(1.0)
    assert prefs.get_font_scale() == pytest.approx(1.0)
    assert gui_scale.current_scale() == pytest.approx(1.0)


@pytest.mark.parametrize("keep", [False, True])
def test_gesture_confirmation_uses_the_saved_previous_scales(qtbot, qt_theme_applied, keep):
    previous = (prefs.get_gui_scale(), prefs.get_font_scale())
    answers = []
    assert gui_scale.change_scales(gui=0.7, font=1.3, ask=False,
                                   on_done=answers.append) is None
    assert answers == [True]
    answers.clear()
    dialog = gui_scale.change_scales(previous=previous, seconds=60,
                                    require_parent=False, on_done=answers.append)
    if keep:
        dialog.keep_button.click()
    else:
        dialog.revert_button.click()
    qtbot.waitUntil(lambda: bool(answers))
    assert answers == [keep]
    assert prefs.get_gui_scale() == pytest.approx(0.7 if keep else previous[0])
    assert prefs.get_font_scale() == pytest.approx(1.3 if keep else previous[1])


def test_requesting_the_current_scales_does_not_open_a_question():
    answers = []
    assert gui_scale.change_scales(gui=prefs.get_gui_scale(), font=prefs.get_font_scale(),
                                   on_done=answers.append) is None
    assert answers == []


@pytest.mark.parametrize("layout_type,dimension", [(QHBoxLayout, "width"), (QVBoxLayout, "height")])
def test_fixed_spacers_inserted_into_a_layout_round_trip_without_drift(qtbot, layout_type, dimension):
    panel = QWidget()
    qtbot.addWidget(panel)
    layout = layout_type(panel)
    layout.addWidget(QLabel("Existing control"))
    layout.insertSpacing(0, 30)
    spacer = layout.itemAt(0).spacerItem()
    assert getattr(spacer.sizeHint(), dimension)() == 30
    gui_scale.set_gui_scale_live(0.5)
    assert getattr(spacer.sizeHint(), dimension)() == 15
    gui_scale.set_gui_scale_live(2.0)
    assert getattr(spacer.sizeHint(), dimension)() == 60
    gui_scale.set_gui_scale_live(1.0)
    assert getattr(spacer.sizeHint(), dimension)() == 30


def test_splitter_readback_can_be_saved_and_reapplied_while_scaled(qtbot):
    splitter = QSplitter()
    qtbot.addWidget(splitter)
    splitter.addWidget(QWidget())
    splitter.addWidget(QWidget())
    splitter.resize(800, 300)
    splitter.show()
    qtbot.waitExposed(splitter)
    splitter.setSizes([600, 200])
    actual_sizes = gui_scale._ORIGINAL[(QSplitter, "sizes")]
    before = actual_sizes(splitter)
    assert all(size > 0 for size in before)
    gui_scale.set_gui_scale_live(0.5)
    saved = splitter.sizes()
    assert saved == [2 * size for size in actual_sizes(splitter)]
    splitter.setSizes(saved)
    assert actual_sizes(splitter) == before
    gui_scale.set_gui_scale_live(1.0)
    assert splitter.sizes() == before


def test_saved_gui_scale_is_applied_to_existing_widgets(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.setMinimumWidth(200)
    prefs.set_gui_scale(0.6)
    assert gui_scale.apply_saved_gui_scale() == pytest.approx(0.6)
    getter = gui_scale._ORIGINAL[(QWidget, "minimumWidth")]
    assert getter(widget) == 120
    assert widget.minimumWidth() == 200


@pytest.mark.parametrize("stored,expected", [(None, 1.0), ("invalid", 1.0), (-4, 0.1), (9, 2.0)])
def test_invalid_saved_preview_scales_remain_usable(stored, expected):
    prefs._settings().setValue("prefs/preview_scale/recovery", stored)
    assert ps.get_preview_scale("recovery") == pytest.approx(expected)


def test_preview_updates_and_notifies_hooks_when_saving_fails(qtbot, monkeypatch):
    ps.set_preview_scale("unit_test", 1.0)
    panel, control, label, thumb, view, layout = _panel(qtbot)
    control.set_percent(50)
    calls = []

    def broken_hook(scale):
        raise RuntimeError("renderer has closed")

    control.scaler.add_hook(broken_hook)
    control.scaler.add_hook(calls.append)
    assert calls == [0.5], "a newly attached renderer receives the current scale"

    def unwritable(name, scale):
        raise PermissionError("read-only preference store")

    monkeypatch.setattr(ps, "set_preview_scale", unwritable)
    assert control.scaler.set_scale(0.75) == pytest.approx(0.75)
    assert thumb.minimumWidth() == 99
    assert view.minimumHeight() == 120
    assert calls == [0.5, 0.75]
    assert ps.get_preview_scale("unit_test") == pytest.approx(0.5)


def test_preview_refresh_tracks_changed_parent_styles_and_restores_them(qtbot):
    ps.set_preview_scale("style_recovery", 1.0)
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.setStyleSheet("QLabel {font-size:20px; color:red;}")
    panel = QWidget(parent)
    layout = QVBoxLayout(panel)
    label = QLabel("Preview", panel)
    layout.addWidget(label)
    control = ps.install_preview_scale(panel, "style_recovery", layout)
    QVBoxLayout(parent).addWidget(panel)
    parent.show()
    panel.show()
    control.scaler.set_scale(0.5, persist=False)
    qtbot.waitUntil(lambda: label.font().pixelSize() == 10)
    parent.setStyleSheet("QLabel {font-size:30px; color:blue;}")
    control.scaler.refresh()
    qtbot.waitUntil(lambda: label.font().pixelSize() == 15)
    control.scaler.set_scale(1.0, persist=False)
    qtbot.waitUntil(lambda: label.font().pixelSize() == 30)
    assert panel.styleSheet() == ""
    assert parent.styleSheet() == "QLabel {font-size:30px; color:blue;}"
