"""A mask dimension names a plane, and sometimes there is no such object.

"in the measure modual there is always an integer in the cell mask dim,
nucleus mask dim and the pathogen mask dim, however all of these objects will
not allways be present so i should be able to delete the integer from the
setting so that there is nothing there."

The organelle slots always got this right -- ``organelle_mask_dim`` ships None
and its control is a box that can be emptied -- while cell, nucleus and
pathogen shipped 4, 5 and 6 and got a spin box, which has no empty state. The
number could be changed and never CLEARED, so a screen with no nucleus was
made to name a nucleus plane.

Everything below is measured off the built panel and off the code that reads
the value, not off the declaration that decides them.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

#: The three non-slot objects a fresh Measure run declares. Organelle planes
#: are checked separately on a form whose count says that slot exists: a
#: fresh run now correctly declares zero organelles and builds no slot rows.
PLANE_KEYS = ("cell_mask_dim", "nucleus_mask_dim", "pathogen_mask_dim")


def _screen(qtbot, app_key: str, current=None):
    from spacr.qt.screens.app_screen import AppScreen

    before = AppScreen.values_the_next_screen_is_built_for
    AppScreen.values_the_next_screen_is_built_for = current
    try:
        screen = AppScreen(app_key)
    finally:
        AppScreen.values_the_next_screen_is_built_for = before
    qtbot.addWidget(screen)
    # The first visibility pass is scheduled, not immediate: the rows do not
    # exist until the screen has laid them out. See `build_sections`.
    qtbot.wait(1)
    return screen, screen._settings_model


# ---------------------------------------------------------------------------
# The control
# ---------------------------------------------------------------------------

def test_every_mask_dim_offers_a_control_that_can_be_emptied(qtbot):
    """A spin box cannot express "there is no such object"."""
    from PySide6.QtWidgets import QLineEdit

    _screen_, model = _screen(qtbot, "measure")

    for key in PLANE_KEYS:
        widget = model._widgets[key]
        assert isinstance(widget, QLineEdit), (
            f"{key} is a {type(widget).__name__}, which has no empty state")

    _organelle_screen, organelle_model = _screen(
        qtbot, "measure", {"number_of_organelles": 1})
    widget = organelle_model._widgets["organelle_mask_dim"]
    assert isinstance(widget, QLineEdit), (
        "organelle_mask_dim has no empty state when its slot exists")


def test_clearing_a_mask_dim_stores_none(qtbot):
    """Typed empty, collected as None -- not as '' and not as the old number."""
    _screen_, model = _screen(qtbot, "measure")

    assert model.collect()["cell_mask_dim"] == 4
    model._widgets["cell_mask_dim"].clear()
    qtbot.wait(1)
    assert model.collect()["cell_mask_dim"] is None
    # AND THE OTHERS ARE UNTOUCHED. Emptying one object's plane says nothing
    # about another's.
    assert model.collect()["nucleus_mask_dim"] == 5
    assert model.collect()["pathogen_mask_dim"] == 6


def test_a_plane_of_zero_is_a_real_plane(qtbot):
    """Plane 0 is the first plane, not an absent object."""
    _screen_, model = _screen(qtbot, "measure")

    model._widgets["cell_mask_dim"].setText("0")
    qtbot.wait(1)
    assert model.collect()["cell_mask_dim"] == 0


def test_an_emptied_cell_mask_dim_keeps_the_cell_family_available(qtbot):
    """Cell is the explicit exception to optional-object form gating."""
    screen, model = _screen(qtbot, "measure")

    assert screen.setting_row_is_visible("cell_min_size") is True
    model._widgets["cell_mask_dim"].clear()
    model._widgets["cell_mask_dim"].editingFinished.emit()
    qtbot.wait(1)
    assert model.collect()["cell_mask_dim"] is None
    assert screen.setting_row_is_visible("cell_min_size") is True
    assert screen.setting_row_is_visible("cell_mask_dim") is True

    model._widgets["cell_mask_dim"].setText("4")
    model._widgets["cell_mask_dim"].editingFinished.emit()
    qtbot.wait(1)
    assert screen.setting_row_is_visible("cell_min_size") is True


@pytest.mark.parametrize("role", ["nucleus", "pathogen"])
def test_a_committed_optional_mask_dim_rebuilds_the_live_form(
        qapp, qtbot, role, monkeypatch):
    """Measure watches its mask planes, not Mask's similarly named channels."""
    from spacr.qt.app import MainWindow
    from spacr.qt.settings_search import (ALL, disclosure_for,
                                           remember_disclosure)

    previous = disclosure_for("measure")
    remember_disclosure("measure", ALL)
    window = MainWindow()
    qtbot.addWidget(window)
    try:
        window.show()
        window._on_nav_selected("measure")
        qapp.processEvents()
        screen = window._screens["measure"]
        switch = screen._settings_model._widgets[f"{role}_mask_dim"]
        follower = f"{role}_min_size"
        assert follower in screen._settings_model._widgets
        assert screen.setting_row_is_visible(follower) is True

        # Leaving an unchanged shaping field is a no-op, including on the
        # first screen opened through normal navigation.
        switch.editingFinished.emit()
        qapp.processEvents()
        assert window._screens["measure"] is screen

        # EMPTYING THE PLANE HIDES ITS ROWS. It does not rebuild the screen,
        # and this test used to assert that it did. Committing a plane used
        # to call `rebuild_app_screen` -- 455 ms, and a DIFFERENT screen
        # object in the window's stack -- to change which rows are visible,
        # so everything uncommitted, every scroll position and every open
        # fold went with it. `refresh_object_visibility` does the same job in
        # place.
        retired = []
        monkeypatch.setattr(
            screen, "_shutdown_settings_widgets",
            lambda: retired.append(screen))
        switch.clear()
        switch.editingFinished.emit()
        qapp.processEvents()

        assert retired == [], "emptying a plane rebuilt the whole screen"
        assert window._screens["measure"] is screen, "the screen was replaced"
        assert screen.setting_row_is_visible(follower) is False
        # THE WIDGET STAYS, holding its value. Hidden and not dropped is what
        # lets the row come back with the number the user typed, rather than
        # with the module default.
        assert follower in screen._settings_model._widgets
        assert f"{role}_mask_dim" in screen._settings_model._widgets

        switch = screen._settings_model._widgets[f"{role}_mask_dim"]
        switch.setText("4")
        switch.editingFinished.emit()
        qapp.processEvents()

        assert window._screens["measure"] is screen
        assert follower in screen._settings_model._widgets
        assert screen.setting_row_is_visible(follower) is True
    finally:
        window.close()
        remember_disclosure("measure", previous)


def test_a_shape_edit_during_a_run_leaves_the_run_alone(qapp, qtbot):
    """A form edit during a run must not cancel it through closeEvent.

    This used to be a test about DEFERRING a rebuild until the thread
    finished, because emptying a plane replaced the whole screen and closing
    the old one cancelled the run underneath it. Rows are hidden in place
    now, so there is no rebuild to defer and no closeEvent to reach the
    worker -- the danger is gone rather than scheduled.

    The property still worth pinning is the one in the summary: the run is
    untouched, and the form still answers correctly while it is going.
    """
    from spacr.qt.app import MainWindow

    class RunningThread:
        running = True

        def isRunning(self):
            return self.running

    class Worker:
        def __init__(self):
            self.cancelled = []

        def request_cancel(self, reason):
            self.cancelled.append(reason)

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window._on_nav_selected("measure")
    qapp.processEvents()
    original = window._screens["measure"]
    thread = RunningThread()
    worker = Worker()
    original._thread = thread
    original._worker = worker

    switch = original._settings_model._widgets["nucleus_mask_dim"]
    switch.clear()
    switch.editingFinished.emit()
    qapp.processEvents()

    assert window._screens["measure"] is original
    assert worker.cancelled == [], "editing the form cancelled the run"
    assert original.setting_row_is_visible("nucleus_min_size") is False
    assert original._settings_model.collect()["nucleus_mask_dim"] is None

    # AND WHEN THE RUN ENDS, still no rebuild: there is nothing queued that
    # the finishing thread could set off.
    thread.running = False
    original._clear_thread_refs()
    qapp.processEvents()

    assert window._screens["measure"] is original
    assert worker.cancelled == []


def test_the_measure_panel_can_say_the_run_has_no_masks_at_all(qtbot):
    """All three cleared is a legal thing to type, and it collects as None."""
    screen, model = _screen(qtbot, "measure")

    for key in ("cell_mask_dim", "nucleus_mask_dim", "pathogen_mask_dim"):
        model._widgets[key].clear()
    qtbot.wait(1)
    collected = model.collect()
    assert [collected[k] for k in
            ("cell_mask_dim", "nucleus_mask_dim", "pathogen_mask_dim")] \
        == [None, None, None]


# ---------------------------------------------------------------------------
# What the value means once it leaves the panel
# ---------------------------------------------------------------------------

def test_the_declaration_admits_none_for_every_plane_naming_key():
    """A channel that may be None and a dimension that may not is a
    contradiction the panel would otherwise permit."""
    from spacr.settings import expected_types

    for role in ("cell", "nucleus", "pathogen", "organelle"):
        for suffix in ("channel", "mask_dim", "chann_dim"):
            key = f"{role}_{suffix}"
            declared = expected_types.get(key)
            if declared is None:
                continue
            allowed = declared if isinstance(declared, tuple) else (declared,)
            assert type(None) in allowed, f"{key} cannot be left unset"


def test_a_settings_file_that_omits_the_key_loads():
    """Absence is not an error; it is answered with the shipped default."""
    from spacr.settings import get_measure_crop_settings

    filled = get_measure_crop_settings(settings={"src": "/tmp/spacr-nowhere"})
    assert filled["cell_mask_dim"] == 4
    # And an explicit None survives the defaults pass rather than being
    # overwritten by that same 4.
    kept = get_measure_crop_settings(
        settings={"src": "/tmp/spacr-nowhere", "cell_mask_dim": None})
    assert kept["cell_mask_dim"] is None


def test_a_cleared_mask_dim_survives_the_settings_csv(qtbot, tmp_path):
    """The whole route: empty the box, save, open the file in a new panel.

    Written and read through the same two functions the module uses --
    `spacr.utils.save_settings` and the panel's own CSV loader -- because
    "None" and "" are the values a round trip is most likely to turn back
    into a number.
    """
    import os

    from spacr.qt.screens.app_screen import AppScreen
    from spacr.utils import save_settings

    _screen_, model = _screen(qtbot, "measure")
    model._widgets["nucleus_mask_dim"].clear()
    qtbot.wait(1)

    src = tmp_path / "merged"
    src.mkdir()
    settings = dict(model.collect())
    settings["src"] = str(src)
    save_settings(settings, name="measure_crop_settings", show=False)
    written = src / "settings" / "measure_crop_settings.csv"
    assert written.is_file()

    loaded = AppScreen._load_settings_csv(str(written))
    assert loaded["nucleus_mask_dim"] is None

    reopened, reopened_model = _screen(qtbot, "measure")
    reopened.apply_settings_dict(loaded)
    qtbot.wait(1)
    assert reopened_model.collect()["nucleus_mask_dim"] is None
    # AND THE PANEL AGREES WITH THE FILE: a run with no nucleus does not
    # offer nucleus settings.
    assert reopened.setting_row_is_visible("nucleus_min_size") is False
    assert reopened.setting_row_is_visible("cell_min_size") is True
    assert os.path.exists(str(written))


def test_the_preflight_does_not_flag_an_absent_object():
    """`validate` is what a run meets first, and None is not a type error."""
    from spacr.settings import get_measure_crop_settings
    from spacr.validate import _check_types

    settings = get_measure_crop_settings(settings={"src": "/tmp/spacr-nowhere"})
    settings["nucleus_mask_dim"] = None
    flagged = [p for p in _check_types(settings, "measure")
               if p.key == "nucleus_mask_dim"]
    assert flagged == []


def test_measure_crops_own_gate_accepts_an_absent_plane():
    """The check `measure_crop` applies before it reads a field.

    Read off the source rather than by running the pipeline, because what is
    at stake is one clause -- ``or settings[key] is None`` -- and a run that
    needs a plate to reach it would not tell us whether the clause is there.
    """
    import inspect

    from spacr import measure

    source = inspect.getsource(measure.measure_crop)
    assert "isinstance(settings.get(key), int)" in source
    assert "settings.get(key) is None" in source, (
        "measure_crop rejects an unset mask plane")
