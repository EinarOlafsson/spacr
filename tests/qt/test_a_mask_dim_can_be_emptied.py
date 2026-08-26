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

#: The three that could not be emptied, and the organelle slot that always
#: could -- kept in one list so a rule that fixed three of four fails here.
PLANE_KEYS = ("cell_mask_dim", "nucleus_mask_dim", "pathogen_mask_dim",
              "organelle_mask_dim")


def _screen(qtbot, app_key: str):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key)
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


def test_an_emptied_mask_dim_takes_that_object_off_the_form(qtbot):
    """The two answers agree: no plane, no rows; the plane back, rows back."""
    screen, model = _screen(qtbot, "measure")

    assert screen.setting_row_is_visible("cell_min_size") is True
    model._widgets["cell_mask_dim"].clear()
    qtbot.wait(1)
    assert screen.setting_row_is_visible("cell_min_size") is False
    # The switch itself is never hidden -- there would be nothing left to
    # turn the object back on with.
    assert screen.setting_row_is_visible("cell_mask_dim") is True

    model._widgets["cell_mask_dim"].setText("4")
    qtbot.wait(1)
    assert screen.setting_row_is_visible("cell_min_size") is True


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
    assert ("isinstance(settings[key], int) or settings[key] is None"
            in source), "measure_crop rejects an unset mask plane"
