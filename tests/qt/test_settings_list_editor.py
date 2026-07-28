"""List and list-of-list settings are edited as chips, not as Python literals.

Every list-valued setting used to be a text box holding ``repr(value)``:
``[['c1'], ['c2']]``, ``['r', 'g', 'b']``, ``[224, 224]``. Two things were
wrong with that.

*It was unusable.* A missing bracket is a parse failure the user cannot
diagnose, and ``_ListEdit.get_value`` silently returned the unparseable text
as a plain string rather than saying so.

*It was not even reached.* ``gui_utils.convert_settings_dict_for_gui``
stringifies every list default before this module sees it, so
``isinstance(default, list)`` in ``_widget_for`` was never true and every
list setting got a ``_ScalarEdit``. ``collect()`` then handed the pipeline
the raw text, because ``_coerce_to_expected_type`` only ever parsed bool,
int and float. That is how ``class_metadata`` arrived at
``io.generate_training_dataset`` as the string ``"[['c1'], ['c2']]"`` and was
iterated one character at a time.

What must not change is the stored value: a settings CSV written before this
widget existed has to load into it, and every consumer has to read exactly
what it always did.

CPU-only, offline, deterministic; Qt offscreen, no modal dialogs.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.settings_model import (          # noqa: E402
    SettingsWidgets, _ChipStrip, _ListEditor, list_shape_for,
)

#: Every module whose panel the app can build.
APP_KEYS = [
    "mask", "timelapse", "motility", "measure", "classify", "umap",
    "train_cellpose", "ml_analyze", "cellpose_masks", "cellpose_all",
    "map_barcodes", "regression", "recruitment", "activation",
    "analyze_plaques", "invasion", "replication",
]


def _model(qapp, app_key):
    model = SettingsWidgets(app_key)
    model.build_sections()
    return model


# ---------------------------------------------------------------------------
# the value survives the round trip, for every module
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", APP_KEYS)
def test_every_default_round_trips_untouched(qapp, app_key):
    """Building the panel and immediately collecting must return the module's
    own defaults -- not a stringified, clamped or substituted version."""
    model = _model(qapp, app_key)
    collected = model.collect()
    drift = {}
    for key, value in model._defaults.items():
        if key not in model._widgets:
            continue
        got = collected.get(key)
        if got == value:
            continue
        # Two pre-existing, harmless normalisations: an empty path string
        # reads back as None, and a float spin box keeps six decimals.
        if value == "" and got is None:
            continue
        if isinstance(value, float) and isinstance(got, float) \
                and round(value, 6) == round(got, 6):
            continue
        drift[key] = (value, got)
    assert not drift, drift


@pytest.mark.parametrize("app_key", APP_KEYS)
def test_no_list_setting_reaches_the_pipeline_as_a_string(qapp, app_key):
    """A list default must collect as a list, whatever widget rendered it.

    This covers the curated combos too ('channels', 'crop_mode',
    'train_channels', ...), whose options are TEXT -- "['r','g','b']" -- and
    which therefore shipped a string to the pipeline as well.
    """
    model = _model(qapp, app_key)
    collected = model.collect()
    strings = {k: collected.get(k) for k, v in model._defaults.items()
               if isinstance(v, (list, tuple)) and k in model._widgets
               and isinstance(collected.get(k), str)}
    assert not strings, strings


# ---------------------------------------------------------------------------
# which keys get the editor -- and which deliberately do not
# ---------------------------------------------------------------------------

def test_the_nested_class_setting_gets_a_nested_editor(qapp):
    model = _model(qapp, "classify")
    widget = model._widgets["class_metadata"]
    assert isinstance(widget, _ListEditor)
    assert widget._nested is True
    assert len(widget._strips) == 2          # one row per class
    assert widget.get_value() == [["c1"], ["c2"]]


def test_a_flat_list_setting_gets_one_strip(qapp):
    model = _model(qapp, "classify")
    widget = model._widgets["classes"]
    assert isinstance(widget, _ListEditor)
    assert widget._nested is False
    assert widget.get_value() == ["nc", "pc"]


def test_src_keeps_its_line_edit(qapp):
    """``src`` is declared ``(str, list)`` but is a path. It has to stay a
    QLineEdit: drag-and-drop, the empty-state banner and the column picker's
    ``_settings_src_path`` all test for one."""
    from PySide6.QtWidgets import QLineEdit
    for app_key in ("mask", "measure", "classify"):
        widget = _model(qapp, app_key)._widgets["src"]
        assert isinstance(widget, QLineEdit), app_key
        assert not isinstance(widget, _ListEditor), app_key


def test_a_list_declared_key_with_a_placeholder_string_default_is_left_alone(qapp):
    """``count_data``/``score_data`` are declared ``list`` but ship the
    *string* 'list of paths'. Chipping that would turn a placeholder into a
    one-element list."""
    model = _model(qapp, "regression")
    for key in ("count_data", "score_data"):
        assert not isinstance(model._widgets.get(key), _ListEditor), key


def test_a_none_default_declared_list_still_gets_the_editor(qapp):
    """``tables`` is None in the classify defaults and declared ``list``."""
    model = _model(qapp, "classify")
    widget = model._widgets["tables"]
    assert isinstance(widget, _ListEditor)
    assert widget.get_value() is None       # empty stays None, not []


# ---------------------------------------------------------------------------
# element typing
# ---------------------------------------------------------------------------

def test_numbers_stay_numbers_and_text_stays_text(qapp):
    model = _model(qapp, "measure")
    dims = model._widgets["png_dims"]
    dims._strips[0]._entry.setText("3")
    dims._strips[0]._commit_entry()
    assert dims.get_value() == [0, 1, 2, 3]
    assert all(isinstance(v, int) for v in dims.get_value())

    ratios = model._widgets["dialate_png_ratios"]
    ratios._strips[0]._entry.setText("0.5")
    ratios._strips[0]._commit_entry()
    assert ratios.get_value() == [0.2, 0.5]

    classes = _model(qapp, "classify")._widgets["classes"]
    classes._strips[0]._entry.setText("3")
    classes._strips[0]._commit_entry()
    # element type inferred from the default (['nc','pc']) -> text, so a
    # class literally named "3" is not silently turned into the integer 3
    assert classes.get_value() == ["nc", "pc", "3"]


# ---------------------------------------------------------------------------
# adding and removing
# ---------------------------------------------------------------------------

def test_a_chip_can_be_added_and_removed(qapp):
    model = _model(qapp, "classify")
    widget = model._widgets["classes"]
    strip = widget._strips[0]

    strip._entry.setText("mid")
    strip._commit_entry()
    assert widget.get_value() == ["nc", "pc", "mid"]

    strip._chips[0].removed.emit(strip._chips[0])
    assert widget.get_value() == ["pc", "mid"]


def test_a_comma_splits_a_pasted_run_into_chips(qapp):
    widget = _model(qapp, "classify")._widgets["classes"]
    strip = widget._strips[0]
    for text in ("a,", "b,", "c"):
        strip._entry.setText(strip._entry.text() + text)
        strip._on_typed(strip._entry.text())
    assert widget.get_value() == ["nc", "pc", "a", "b", "c"]


def test_uncommitted_text_is_still_collected(qapp):
    """A user who types a value and presses Run without leaving the field
    must not lose it."""
    widget = _model(qapp, "classify")._widgets["classes"]
    widget._strips[0]._entry.setText("typed")
    assert widget.get_value() == ["nc", "pc", "typed"]


def test_add_group_adds_a_row_and_removing_the_last_one_flattens(qapp):
    widget = _model(qapp, "classify")._widgets["class_metadata"]
    widget._on_footer()                       # + Add group
    assert len(widget._strips) == 3
    widget._strips[2]._entry.setText("c3")
    widget._strips[2]._commit_entry()
    assert widget.get_value() == [["c1"], ["c2"], ["c3"]]

    # An empty group contributes nothing rather than an empty inner list
    widget._on_footer()
    assert widget.get_value() == [["c1"], ["c2"], ["c3"]]

    for strip in list(widget._strips):
        widget._drop_strip(strip)
    assert widget._nested is False
    assert widget.get_value() is None or widget.get_value() == []


def test_a_flat_nested_capable_key_can_be_grouped(qapp):
    """``png_size`` is [224, 224] but documented as accepting a list of lists.
    The literal box could express that; the chip editor has to as well."""
    widget = _model(qapp, "measure")._widgets["png_size"]
    assert widget._nested is False
    assert widget._nested_capable is True
    widget._on_footer()                       # "Use groups"
    assert widget._nested is True
    assert widget.get_value() == [[224, 224]]


def test_a_plain_list_key_offers_no_grouping(qapp):
    widget = _model(qapp, "measure")._widgets["homogeneity_distances"]
    assert widget._nested_capable is False
    assert widget._footer.isVisible() is False


# ---------------------------------------------------------------------------
# loading what is already on disk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("[['c1'], ['c2']]", [["c1"], ["c2"]]),
    ("['nc', 'pc']", ["nc", "pc"]),
    ("[1, 2]", [1, 2]),
    ("c1, c2, c3", ["c1", "c2", "c3"]),       # hand-edited CSV, no brackets
    ("", None),
    ("None", None),
])
def test_set_value_parses_what_a_settings_csv_holds(qapp, text, expected):
    """Settings CSVs store the repr; ``set_value`` has to read it back."""
    widget = _ListEditor(key="class_metadata", default=None,
                         nested_capable=True, allow_none=True)
    widget.set_value(text)
    assert widget.get_value() == expected


def test_importing_a_settings_dict_reaches_the_chip_editor(qapp):
    """AppScreen._apply_value used to have only a QLineEdit branch, and the
    chip editor is a QWidget -- an imported list would have been dropped."""
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("classify")
    applied = screen.apply_settings_dict({"class_metadata": "[['r1'], ['r2']]",
                                          "classes": ["a", "b"]})
    assert applied == 2
    assert screen._settings_model._widgets["class_metadata"].get_value() \
        == [["r1"], ["r2"]]
    assert screen._settings_model._widgets["classes"].get_value() == ["a", "b"]
    screen.deleteLater()


def test_live_preview_propagation_reaches_the_chip_editor(qapp):
    model = _model(qapp, "measure")
    assert model.set_value_for_key("png_dims", [1, 2]) is True
    assert model._widgets["png_dims"].get_value() == [1, 2]


# ---------------------------------------------------------------------------
# shape resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,default,expect_editor", [
    ("class_metadata", [["c1"], ["c2"]], True),
    ("classes", ["nc", "pc"], True),
    ("png_dims", [0, 1, 2], True),
    ("tables", None, True),
    ("src", "path", False),
    ("count_data", "list of paths", False),
    ("sample", None, False),
    ("epochs", 2, False),
])
def test_list_shape_for_picks_the_right_keys(key, default, expect_editor):
    assert (list_shape_for(key, default) is not None) is expect_editor


def test_a_tuple_declared_setting_collects_as_a_tuple(qapp):
    """``motility_xlim`` is declared ``tuple`` and defaults to (100, -100)."""
    model = _model(qapp, "motility")
    widget = model._widgets.get("motility_xlim")
    assert isinstance(widget, _ListEditor)
    assert widget.get_value() == (100, -100)


# ---------------------------------------------------------------------------
# the strip itself
# ---------------------------------------------------------------------------

def test_chip_strip_set_values_replaces_rather_than_appends(qapp):
    strip = _ChipStrip()
    strip.set_values(["a", "b"])
    strip.set_values(["c"])
    assert strip.values() == ["c"]
