"""Room folding, the idle build of a folded screen and the Napari hand-off in Make Masks."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from spacr.qt.screens import make_masks as mm

pytestmark = pytest.mark.qt

Screen = mm.MakeMasksScreen


# -- folding panes for room ----------------------------------------------------------

def test_a_screen_without_its_splitters_offers_no_panes_to_fold():
    fake = SimpleNamespace()
    assert Screen._room_panes(fake) == []
    fake = SimpleNamespace(_body_splitter=None, _view_pane=None, _image_container=None,
                           _chrome_splitter=None)
    assert Screen._room_panes(fake) == []


def test_a_title_with_no_instruction_line_has_nothing_to_fold(qapp):
    fake = SimpleNamespace(_header=SimpleNamespace(instruction_label=None))
    assert Screen._room_folded(fake, fake, mm.TITLE_LINE_PANE) is True
    assert Screen._fold_room(fake, fake, mm.TITLE_LINE_PANE, True) is False


def test_the_instruction_line_folds_without_a_title_bar_or_a_measurement_hook(qtbot):
    header = QWidget()
    qtbot.addWidget(header)
    QVBoxLayout(header)
    label = QLabel("Choose a folder of images to start.", header)
    header.instruction_label = label
    fake = SimpleNamespace(_header=header)
    assert Screen._fold_room(fake, fake, mm.TITLE_LINE_PANE, True) is True
    assert label.isHidden()
    assert Screen._fold_room(fake, fake, mm.TITLE_LINE_PANE, True) is False    # already folded
    assert Screen._fold_room(fake, fake, mm.TITLE_LINE_PANE, False) is True
    assert not label.isHidden()


def test_the_side_column_is_left_alone_without_its_pane_and_is_not_reopened_when_open(qapp):
    fake = SimpleNamespace(_view_pane=None, _side_held_by_user=True, _side_folded_for_width=True)
    Screen._fit_the_side_column(fake)
    assert fake._side_held_by_user is True and fake._side_folded_for_width is True
    opened = []
    pane = QWidget()
    pane.is_collapsed = lambda name: False
    pane.set_collapsed = lambda name, collapsed, by_user=True: opened.append(name)
    fake = SimpleNamespace(_view_pane=pane, width=lambda: mm._COMPACT_WIDTH + 400,
                           _side_held_by_user=True, _side_folded_for_width=True)
    Screen._fit_the_side_column(fake)
    assert fake._side_held_by_user is False and fake._side_folded_for_width is False
    assert opened == []


def test_only_the_shortcut_column_notes_a_fold_and_a_missing_column_is_tolerated():
    fake = SimpleNamespace(_shortcut_panel=None, _side_folded_for_width=True,
                           _side_held_by_user=False)
    Screen._on_view_pane_toggled(fake, "Settings", True, True)
    assert fake._side_held_by_user is False and fake._side_folded_for_width is True
    Screen._on_view_pane_toggled(fake, "Shortcuts", True, True)
    assert fake._side_held_by_user is True and fake._side_folded_for_width is False


# -- the idle build of a folded screen -----------------------------------------------

def _idle(monkeypatch, **over):
    import spacr.qt.app as qt_app
    import spacr.qt.preferences as preferences
    from spacr.qt import screens as screens_package

    monkeypatch.setattr(preferences, "_screen_prewarm_allowed", lambda: (True, ""))
    monkeypatch.setattr(qt_app, "_prewarm_memory_is_short", lambda: False)
    monkeypatch.setattr(screens_package, "_a_window_is_opening_a_screen", False)
    monkeypatch.setitem(sys.modules, "spacr.qt.gc_policy", SimpleNamespace(_FREEZE_DUE=None))
    retries = []
    monkeypatch.setattr(mm.QTimer, "singleShot",
                        staticmethod(lambda ms, callback: retries.append(ms)))
    laid = []
    fake = SimpleNamespace(
        isVisible=lambda: True, _cold_imports=None, _PREBUILD_RETRY_S=0.25,
        _PREBUILD_FOLDED=("a", "b"), _fold_screens={"a": object(), "b": object()},
        _fold_dialogs={}, window=lambda: SimpleNamespace(_opening_a_screen=False),
        _lay_out_a_folded_screen=laid.append, _prebuild_a_folded_screen=lambda: None)
    fake.__dict__.update(over)
    return fake, retries, laid


def test_the_idle_build_waits_while_the_cold_imports_are_still_running(qapp, monkeypatch):
    fake, retries, laid = _idle(
        monkeypatch, _cold_imports=SimpleNamespace(is_alive=lambda: True))
    Screen._prebuild_a_folded_screen(fake)
    assert retries == [250] and laid == []


def test_the_idle_build_lays_out_one_screen_per_turn_and_stops_when_all_are_laid_out(qapp, monkeypatch):
    fake, retries, laid = _idle(monkeypatch, _laid_out_folds={"a"})
    Screen._prebuild_a_folded_screen(fake)
    assert laid == [fake._fold_screens["b"]] and fake._laid_out_folds == {"a", "b"}
    assert retries == []
    Screen._prebuild_a_folded_screen(fake)             # every screen laid out: nothing to do
    assert laid == [fake._fold_screens["b"]] and retries == []


def test_the_idle_build_gives_up_quietly_when_the_screen_is_gone(qapp):
    def gone():
        raise RuntimeError("Internal C++ object already deleted.")

    assert Screen._prebuild_a_folded_screen(SimpleNamespace(isVisible=gone)) is None


def test_a_folded_screen_with_no_layout_is_still_sized_and_one_that_vanishes_is_tolerated(qtbot):
    screen = QWidget()
    qtbot.addWidget(screen)
    warmed = []
    fake = SimpleNamespace(size=lambda: QSize(320, 200),
                           _warm_the_fields_of=warmed.append)
    Screen._lay_out_a_folded_screen(fake, screen)
    assert (screen.width(), screen.height()) == (320, 200) and warmed == [screen]

    def gone(_screen):
        raise RuntimeError("Internal C++ object already deleted.")

    fake._warm_the_fields_of = gone
    Screen._lay_out_a_folded_screen(fake, screen)          # swallowed


def test_a_popup_backdrop_is_not_warmed_with_the_animated_fields(qtbot, monkeypatch):
    from spacr.qt.widgets.ambient import AmbientWidget

    screen = QWidget()
    qtbot.addWidget(screen)
    backdrop = AmbientWidget(screen)
    backdrop.setProperty("spacrPopupBackdrop", True)
    monkeypatch.setattr(AmbientWidget, "_art_render_size",
                        lambda self, w, h: pytest.fail("rendered a popup backdrop"))
    assert backdrop.property("spacrPopupBackdrop") is True
    Screen._warm_the_fields_of(SimpleNamespace(), screen)
    assert not backdrop.isVisible()


# -- the Napari hand-off ---------------------------------------------------------------

def test_a_missing_napari_extra_is_reported_in_the_status_pane(tmp_path, monkeypatch):
    from spacr import napari_bridge

    mask = tmp_path / "mask.tif"
    mask.write_bytes(b"x")
    said = []

    def missing(handoff):
        raise napari_bridge.NapariExtraMissing("Install the napari extra first.")

    monkeypatch.setattr(napari_bridge, "load_handoff", lambda path, image: object())
    monkeypatch.setattr(napari_bridge, "open_in_napari", missing)
    fake = SimpleNamespace(mask_path=lambda: str(mask), image_path=lambda: "", say=said.append)
    assert mm.NapariBridgeScreen.open_in_napari(fake) is None
    assert said == ["Install the napari extra first."]
