"""A Cellpose-checkpoint settings row offers the zoo as well as a typed path.

A model setting takes a filesystem path, which is exact and unhelpful: the user
has to already know a model exists, find it, and type it correctly. The button
beside the field is the other way in.

WHICH ROWS ACTUALLY GET IT, measured rather than assumed. Of every built-in
module, exactly one renders a key in ``_MODEL_ZOO_KEYS`` today:
``analyze_plaques`` renders ``plaque_model``. Mask offers ``pathogen_model``
in its settings dict, but the panel builds no widget for it -- the Pathogen
section is conditional on the run having a pathogen and no path through the
panel turns it on. So the button is tested where it appears, and instruction
322's file records the missing row as the separate piece of work it is.

Two things have to stay true, and both have been broken by wrappers on this
panel before:

  * the INNER FIELD is still what the panel collects from. ``_lay_out_setting_row``
    records a wrapped row losing its ``settingKey`` the moment the wrapper
    stopped being a no-op;
  * typing a path by hand still works. The button ADDS a way in.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def plaque_screen(qapp):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key="analyze_plaques")
    yield screen
    screen.deleteLater()


def _row_for(screen, key):
    """The row the panel laid out for ``key`` -- the wrapper, if there is one."""
    from PySide6.QtWidgets import QWidget

    field = screen._settings_model._widgets.get(key)
    if field is None:
        return None
    holder = field.parentWidget()
    while isinstance(holder, QWidget):
        if getattr(holder, "_spacr_field", None) is field:
            return holder
        holder = holder.parentWidget()
    return field


def test_the_model_row_carries_a_zoo_button(plaque_screen):
    from PySide6.QtWidgets import QPushButton

    row = _row_for(plaque_screen, "plaque_model")
    assert row is not None, "the plaque panel offers no plaque_model row"
    buttons = [b for b in row.findChildren(QPushButton)
               if "zoo" in b.text().lower()]
    assert buttons, (
        "no model-zoo button: "
        f"{[b.text() for b in row.findChildren(QPushButton)]}")


def test_the_wrapped_row_still_reports_its_setting_key(plaque_screen):
    """The failure this panel has already had once: a wrapped row that lost
    its settingKey read as a field belonging to no setting."""
    row = _row_for(plaque_screen, "plaque_model")
    assert row.property("settingKey") == "plaque_model"


def test_the_inner_field_is_still_what_the_panel_collects(plaque_screen):
    """Typing a path by hand must be unchanged. If wrapping broke collection,
    every pipeline that sets the model by hand would silently stop passing it."""
    assert "plaque_model" in plaque_screen._settings_model.collect()


def test_choosing_a_model_writes_the_path_into_the_field(plaque_screen,
                                                         monkeypatch):
    import spacr.qt.widgets.model_zoo_picker as picker

    monkeypatch.setattr(picker, "choose_model",
                        lambda *a, **k: "/models/cpsam_plaque_r3")
    field = plaque_screen._settings_model._widgets["plaque_model"]
    plaque_screen._choose_a_model_for(field)

    value = field.text() if hasattr(field, "text") else field.get_value()
    assert value == "/models/cpsam_plaque_r3"


def test_a_cancelled_picker_leaves_the_field_alone(plaque_screen, monkeypatch):
    """Cancel must not erase a path the user already typed."""
    import spacr.qt.widgets.model_zoo_picker as picker

    field = plaque_screen._settings_model._widgets["plaque_model"]
    if hasattr(field, "setText"):
        field.setText("/already/typed/by/hand")
    monkeypatch.setattr(picker, "choose_model", lambda *a, **k: None)
    plaque_screen._choose_a_model_for(field)

    value = field.text() if hasattr(field, "text") else field.get_value()
    assert value == "/already/typed/by/hand"


def test_only_cellpose_models_are_offered(plaque_screen, monkeypatch):
    """kinds is a rule, not a parameter. The zoo also carries the YOLO well
    detector, and offering it where CellposeModel consumes the value produces
    a run that fails minutes later naming a file we showed the user."""
    import spacr.qt.widgets.model_zoo_picker as picker

    seen = {}
    monkeypatch.setattr(picker, "choose_model",
                        lambda parent=None, kinds=None: seen.update(kinds=kinds))
    plaque_screen._choose_a_model_for(
        plaque_screen._settings_model._widgets["plaque_model"])
    assert seen["kinds"] == ("cellpose",)
