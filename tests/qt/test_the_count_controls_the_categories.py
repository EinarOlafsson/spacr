"""`number_of_organelles` decides both the settings and their categories."""
from __future__ import annotations

import pytest

import spacr.qt.app as app_module
from spacr.qt.widgets.section import Section


@pytest.fixture(scope="module")
def mask(qapp):
    win = app_module.MainWindow()
    win.resize(1200, 800)
    win.show()
    win._on_nav_selected("mask")
    qapp.processEvents()
    yield win._screens["mask"]
    win.close()


def _categories(screen):
    return [str(s.property("settingsCategorySource"))
            for s in screen.findChildren(Section)]


def test_the_count_defaults_to_none(mask):
    """A run has the organelles it says it has."""
    assert (mask._settings_model.collect() or {}).get(
        "number_of_organelles") == 0


def test_no_organelle_settings_at_zero(mask):
    """Four "Organelle N — Channel" rows were there whatever the count."""
    keys = [k for k in mask._settings_model._widgets
            if "organelle" in k.lower() and k != "number_of_organelles"]
    assert keys == [], f"{len(keys)} organelle settings on a form saying none"


def test_no_organelle_categories_at_zero(mask):
    """A category is not its contents: an empty heading reads as a section
    that failed to load rather than one that does not apply."""
    left = [name for name in _categories(mask) if "rganelle" in name]
    assert left == [], f"empty organelle categories survived: {left}"


def test_the_control_itself_survives(mask):
    """Or a run with none would have no way to ask for one."""
    assert "number_of_organelles" in mask._settings_model._widgets


def test_no_category_is_empty(mask):
    """Asked for 2026-08-28, for every category and not only organelles."""
    empty = []
    for section in mask.findChildren(Section):
        rows = getattr(section, "_row_widgets", None) or ()
        if any(w is not None for _label, w in rows):
            continue
        if any(any(w is not None for _l, w in
                   (getattr(child, "_row_widgets", None) or ()))
               for child in section.findChildren(Section)):
            continue
        empty.append(str(section.property("settingsCategorySource")))
    assert empty == [], f"headings over nothing: {empty}"


def test_raising_the_count_builds_the_slots(mask):
    """A control that was never built cannot be revealed, so it must grow."""
    model = mask._settings_model
    assert model.grow_to_fit_the_organelle_count(3) == 3
    assert model._slots_built_for == 3


def test_growing_never_shrinks(mask):
    """A slot built once keeps whatever has since been put in it."""
    model = mask._settings_model
    model.grow_to_fit_the_organelle_count(5)
    before = model._slots_built_for
    model.grow_to_fit_the_organelle_count(1)
    assert model._slots_built_for == before


def test_an_old_settings_file_still_means_what_it_meant():
    """The default is none; a file carrying slots is not claiming none."""
    from spacr.organelle_types import organelle_count

    assert organelle_count({}) == 0
    assert organelle_count({"cell_channel": 1}) == 0
    # Written before the count existed, carrying four slots.
    old = {"organelle_channel": 1, "organelleb_channel": 2,
           "organellec_channel": 3, "organelled_channel": 0}
    assert organelle_count(old) == 4
    # An explicit count still wins over the inference.
    assert organelle_count({**old, "number_of_organelles": 2}) == 2
    # A key present but blank is a placeholder, not a slot in use.
    assert organelle_count({"organelle_channel": ""}) == 0


def test_an_object_with_no_channel_brings_no_settings(mask):
    """"do the same for the other object classes, except cell"."""
    keys = set(mask._settings_model._widgets)
    for role in ("nucleus", "pathogen"):
        owned = [k for k in keys
                 if k.startswith(f"{role}_") and k != f"{role}_channel"]
        assert owned == [], f"{role} brought {len(owned)} settings unasked"


def test_the_channel_itself_always_stays(mask):
    """Or there is no way to say the run has this object after all."""
    for role in ("cell", "nucleus", "pathogen"):
        assert f"{role}_channel" in mask._settings_model._widgets


def test_cell_is_never_gated(mask):
    """It is the object every other one is measured against."""
    cell = [k for k in mask._settings_model._widgets
            if k.startswith("cell_")]
    assert len(cell) > 5, (
        "cell settings were hidden; the form a user just opened is empty")


def test_the_rule_is_decided_once_and_not_while_typing(mask, qapp):
    """Re-running it per keystroke is what made the module hang."""
    import time

    model = mask._settings_model
    widget = model._widgets.get("nucleus_channel")
    if widget is None:
        pytest.skip("nucleus_channel is not on this panel")

    worst = 0.0
    for index in range(6):
        started = time.perf_counter()
        if hasattr(widget, "setText"):
            widget.setText(str(index % 3))
        else:
            widget.setValue(index % 3)
        qapp.processEvents()
        worst = max(worst, time.perf_counter() - started)
    assert worst < 0.20, f"{worst * 1000:.0f} ms a keystroke"
