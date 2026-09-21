"""Item 423, point (3): the Cellpose 3 models are reachable from a button.

"cyto, nuclei, cyto2 and cyto3 ... usable wherever a segmentation backend is
chosen (Mask generation, Make Masks' magnifier)."

The magnifier got its four modes. Mask generation did not, and the reason is
one word in a filter: :meth:`ModelZooPicker.refresh` keeps
``e.kind in self._kinds or e.kind == "backend"``, and every opener in the app
asked for ``kinds=("cellpose",)``. The four rows existed, were listed by a
picker built with no ``kinds`` at all -- which is the picker a test builds and
NOT the one any button builds -- and were invisible from every button. A user
had to type ``cyto3`` by hand.

So these tests open the picker the way the widgets open it, with the kinds the
widgets pass, and press the controls:

  * the per-object Model cell in Mask generation (``ObjectSettingsGrid``),
    which writes ``cell_model_name`` / ``nucleus_model_name`` /
    ``pathogen_model_name`` -- the settings the Cellpose 3 backend reads BY
    NAME (:func:`spacr.settings._get_object_settings`);
  * the "Model zoo…" button beside a ``*_model_name`` settings row.

And the asymmetry is pinned too, because it is deliberate rather than a
leftover: the Cellpose 4 boxes -- the live preview's model, Make Masks'
Cellpose-SAM model, ``plaque_model`` -- load the checkpoint in spaCR's OWN
process with Cellpose 4, so a Cellpose 3 row offered there is a run that
fails minutes after the click.
"""
from __future__ import annotations

import pytest

import spacr._segmentation_backends as SB
from spacr.qt.widgets import model_zoo_picker as mzp
from spacr.qt.widgets.object_settings_grid import MODEL_QUESTION, ObjectSettingsGrid

CELLPOSE3_MODELS = ("cyto3", "cyto2", "cyto", "nuclei")


@pytest.fixture(autouse=True)
def _no_community_fetch(monkeypatch):
    """Nothing here is about the community catalogue.

    Opening a picker starts a JobRunner to warm it off the GUI thread. These
    tests open several, and a runner still going when the dialog is dropped
    is a thread the rest of the session carries -- so say the cache is fresh
    and no runner is started.
    """
    from spacr import model_zoo

    monkeypatch.setattr(model_zoo, "shared_catalogue_is_stale",
                        lambda *a, **k: False)


@pytest.fixture
def installed_cellpose3(tmp_path):
    """A Cellpose 3 environment that looks installed, without installing one.

    The rows say ``needs the Cellpose 3 backend`` and carry no path until the
    backend is here, and "Use this model" is refused without a path -- so a
    test that wants to press Use has to make the backend look present. The
    marker and the interpreter are what :func:`_backend_state` reads; nothing
    is ever run from this directory.
    """
    env = tmp_path / "backend-environments" / SB._CELLPOSE3
    (env / "bin").mkdir(parents=True, exist_ok=True)
    (env / "bin" / "python").write_text("")
    SB._write_marker(str(env), {"backend": SB._CELLPOSE3, "protocol": 1})
    assert SB._backend_state(SB._CELLPOSE3).ready, "the stand-in is not read"
    return str(env)


def _picker(kinds):
    """A picker built with exactly the kinds a widget passes."""
    dialog = mzp.ModelZooPicker(kinds=kinds)
    return dialog


def _rows(picker):
    """(name, kind) for every row the picker actually draws."""
    out = []
    for stem, pairs in picker._groups:
        entry = pairs[picker._chosen[stem]][1]
        out.append((entry.name, entry.kind))
    return out


def _row_of(picker, name):
    """The table row showing ``name``, or fail saying what was there."""
    for row, (model, _kind) in enumerate(_rows(picker)):
        if model == name:
            # The current UI folds Cellpose 3 until its source is enabled.
            # Open the source as a user does before selecting a model row.
            assert picker.sources.set_on("cellpose3", True)
            assert not picker.table.isRowHidden(row)
            return row
    pytest.fail(f"{name} is not listed: {_rows(picker)}")


def _close(picker):
    picker._stop_any_download()
    picker.deleteLater()


# ---------------------------------------------------------------------------
# What the buttons ask for
# ---------------------------------------------------------------------------

def test_the_per_object_model_cell_asks_for_the_cellpose3_kind():
    """The rule itself, stated once and read by both openers."""
    assert "cellpose3" in ObjectSettingsGrid.MODEL_KINDS
    assert "cellpose" in ObjectSettingsGrid.MODEL_KINDS


@pytest.mark.parametrize("key,expected", [
    ("cell_model_name", ("cellpose", "cellpose3")),
    ("nucleus_model_name", ("cellpose", "cellpose3")),
    ("pathogen_model_name", ("cellpose", "cellpose3")),
    ("organelleb_model_name", ("cellpose", "cellpose3")),
    ("plaque_model", ("cellpose",)),
    ("custom_model", ("cellpose",)),
    ("pathogen_model", ("cellpose",)),
])
def test_only_the_fields_the_backend_reads_by_name_offer_cellpose3(key,
                                                                   expected):
    """``plaque_model`` is the one that would break: analyze_plaques loads it
    with spaCR's own Cellpose 4, which has never heard of cyto3."""
    from spacr.qt.screens.app_screen import AppScreen

    assert AppScreen._model_kinds_for(key) == expected


# ---------------------------------------------------------------------------
# What the picker then lists
# ---------------------------------------------------------------------------

def test_the_four_models_are_listed_when_a_model_cell_opens_the_zoo(
        qapp, installed_cellpose3):
    """The defect, in the form a user meets it."""
    picker = _picker(ObjectSettingsGrid.MODEL_KINDS)
    try:
        listed = dict(_rows(picker))
        for model in CELLPOSE3_MODELS:
            assert listed.get(model) == "cellpose3", (
                f"{model} is not offered: {sorted(listed)}")
    finally:
        _close(picker)


def test_a_cellpose4_only_box_still_refuses_them(qapp, installed_cellpose3):
    """The asymmetry is the point, not an oversight."""
    picker = _picker(("cellpose",))
    try:
        listed = dict(_rows(picker))
        for model in CELLPOSE3_MODELS:
            assert listed.get(model) != "cellpose3", (
                f"{model} was offered to a box that runs Cellpose 4")
        assert "backend" in set(listed.values()), (
            "the backend rows must survive every filter")
    finally:
        _close(picker)


def test_the_rows_say_what_they_need_when_the_backend_is_not_here(qapp):
    """No ``installed_cellpose3``: the rows are still listed -- a backend
    nobody can see is a backend nobody installs -- and each says why it
    cannot be used yet, rather than looking like a usable checkpoint."""
    picker = _picker(ObjectSettingsGrid.MODEL_KINDS)
    try:
        row = _row_of(picker, "cyto3")
        picker.table.selectRow(row)
        entry = picker.selected_entry()
        assert entry.kind == "cellpose3"
        assert picker._local_path(entry) == ""
        assert picker.use_button.isEnabled() is False
        assert picker.download_button.text() == "Install"
        assert mzp._status_text(entry, "") == "needs the Cellpose 3 backend"
    finally:
        _close(picker)


# ---------------------------------------------------------------------------
# Pressing Use, from the cell a user clicks
# ---------------------------------------------------------------------------

def test_clicking_the_model_cell_and_pressing_use_writes_cyto3(
        qapp, qtbot, qt_theme_applied, monkeypatch, installed_cellpose3):
    """End to end through the real dialog: click the cell, the real picker
    opens with the real catalogue, cyto3's row is selected, "Use this model"
    is pressed, and the setting the pipeline reads holds ``cyto3``."""
    from spacr.organelle_types import NUMBER_OF_ORGANELLES
    from spacr.settings import get_timelapse_settings

    settings = get_timelapse_settings()
    settings[NUMBER_OF_ORGANELLES] = 1
    grid = ObjectSettingsGrid()
    qtbot.addWidget(grid)
    grid.set_settings(settings)

    seen = {}

    def press_use(dialog):
        """Stand in for the user, inside the modal loop."""
        seen["kinds"] = dialog._kinds
        seen["rows"] = _rows(dialog)
        dialog.table.selectRow(_row_of(dialog, "cyto3"))
        assert dialog.use_button.isEnabled(), (
            "Use is refused on an installed Cellpose 3 model")
        dialog.use_button.click()
        return dialog.result()

    monkeypatch.setattr(mzp.ModelZooPicker, "exec", press_use)

    source = grid._model.index(list(grid.questions()).index(MODEL_QUESTION),
                               grid.objects().index("cell"))
    mapper = getattr(grid._table.model(), "mapFromSource", None)
    grid._table.clicked.emit(mapper(source) if mapper else source)

    assert grid.settings()["cell_model_name"] == "cyto3"
    assert seen["kinds"] == ObjectSettingsGrid.MODEL_KINDS
    assert ("cyto3", "cellpose3") in seen["rows"]
    assert grid.settings()["nucleus_model_name"] != "cyto3", (
        "picking for one object changed another")


def _zoo_button_for(screen, key):
    """The "Model zoo…" button the Mask panel laid out beside ``key``.

    The field is wrapped in a holder that carries ``_spacr_field``; the
    button is the holder's, not the field's.
    """
    from PySide6.QtWidgets import QPushButton, QWidget

    field = screen._settings_model._widgets.get(key)
    assert field is not None, f"the Mask panel offers no {key} row"
    holder = field.parentWidget()
    while isinstance(holder, QWidget):
        if getattr(holder, "_spacr_field", None) is field:
            break
        holder = holder.parentWidget()
    assert holder is not None, f"{key} has no wrapper"
    buttons = [b for b in holder.findChildren(QPushButton)
               if "zoo" in b.text().lower()]
    assert buttons, f"no model-zoo button beside {key}"
    return field, buttons[0]


def test_the_mask_panels_model_zoo_button_writes_cyto2(
        qapp, qtbot, monkeypatch, installed_cellpose3):
    """The other way in, pressed for real: the Mask panel's own
    ``cell_model_name`` row and the "Model zoo…" button beside it.

    Nothing is stubbed between the click and the field: the button's own
    connection decides the kinds from the key it was built with, the real
    picker lists the real catalogue, and "Use this model" writes the name.

    ``cell_model_name`` and not ``nucleus_model_name``: only the Cell
    section's row is laid out with the wrapper today, which is a gap of its
    own and not this item's.
    """
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)
    field, button = _zoo_button_for(screen, "cell_model_name")
    seen = {}

    def press_use(dialog):
        seen["kinds"] = dialog._kinds
        dialog.table.selectRow(_row_of(dialog, "cyto2"))
        assert dialog.use_button.isEnabled()
        dialog.use_button.click()
        return dialog.result()

    monkeypatch.setattr(mzp.ModelZooPicker, "exec", press_use)
    button.click()

    value = field.text() if hasattr(field, "text") else field.get_value()
    assert value == "cyto2"
    assert seen["kinds"] == ("cellpose", "cellpose3")
    assert screen._settings_model.collect()["cell_model_name"] == "cyto2"


def test_the_plaque_panels_button_is_unchanged(qapp, qtbot, monkeypatch,
                                               installed_cellpose3):
    """The same button on a field Cellpose 4 loads: no Cellpose 3 row."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key="analyze_plaques")
    qtbot.addWidget(screen)
    _field, button = _zoo_button_for(screen, "plaque_model")
    seen = {}

    def look(dialog):
        seen["kinds"] = dialog._kinds
        seen["rows"] = _rows(dialog)
        return 0

    monkeypatch.setattr(mzp.ModelZooPicker, "exec", look)
    button.click()

    assert seen["kinds"] == ("cellpose",)
    assert "cellpose3" not in {kind for _name, kind in seen["rows"]}
