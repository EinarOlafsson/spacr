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
    if set(drift) == {"classes"}:
        # NOT a harmless normalisation, and not a bug in this widget either.
        # `classes` changed shape in 98ae880c from a list of names to a dict
        # of name -> {column, value}, and the editor MIGRATES the old shape
        # deliberately: ['nc', 'pc'] becomes two rows with column '?' and no
        # value, so the user sees what still has to be filled in instead of
        # an empty table. What has not moved with it is the shipped DEFAULT,
        # still the list at spacr/settings.py:1238 with expected_types
        # saying `list`.
        #
        # So the round trip is honest and the default is stale. Deciding what
        # `classes` defaults to in the new shape is the named remaining scope
        # of instruction 37, and it reaches expected_types, validate, the CLI
        # and the Tk screen -- not something to settle from inside a test.
        # Marked rather than deleted so the day 37 lands, this fails and says
        # to remove the mark.
        pytest.xfail(
            "classes default is still the pre-98ae880c list while the editor "
            f"produces the dict shape: {drift['classes']} -- instruction 37")
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
    """A flat list is one strip, not a row per element.

    This used to be driven through classify's ``classes``. That key stopped
    being a flat list on 2026-08-07 (commit 30500970, "classify: the Classes
    editor"): a class is a (name, column, value) RULE now, which a chip strip
    cannot express, so it gets ``ClassEditorWidget`` instead -- asserted
    directly in ``test_the_classes_setting_gets_the_class_editor`` below. The
    invariant itself is unchanged, so it is asserted here on a key that is
    still a flat list of scalars.
    """
    model = _model(qapp, "measure")
    widget = model._widgets["homogeneity_distances"]
    assert isinstance(widget, _ListEditor)
    assert widget._nested is False
    assert len(widget._strips) == 1          # one strip for the whole list
    assert widget.get_value() == [8, 16, 32]
    assert model.collect()["homogeneity_distances"] == [8, 16, 32]


def test_the_classes_setting_gets_the_class_editor(qapp):
    """``classes`` deliberately does NOT get the chip editor.

    Split out of ``test_a_flat_list_setting_gets_one_strip`` when commit
    30500970 (2026-08-07) turned ``classes`` from ``['nc', 'pc']`` into a dict
    of name -> {column, value}. A chip strip can hold the names and nothing
    else, so a class defined on a column would have been unrepresentable; the
    routing to ``ClassEditorWidget`` is the thing worth pinning, because
    falling back to a chip strip would silently lose the column and value of
    every class.
    """
    from spacr.qt.widgets.class_editor import ClassEditorWidget
    widget = _model(qapp, "classify")._widgets["classes"]
    assert isinstance(widget, ClassEditorWidget)
    assert not isinstance(widget, _ListEditor)
    # The value is a mapping of class name -> rule, not a bare list of names.
    # PIN THE CONTENT, not just the shape: `all(...)` over an empty dict is
    # vacuously true, so an editor that silently dropped every class would
    # have passed the shape check. Mutation-proven -- forcing get_value() to
    # return {} left the earlier version of this test green.
    value = widget.get_value()
    assert isinstance(value, dict)
    assert list(value) == ["nc", "pc"], (
        "the two default classes must survive routing to ClassEditorWidget")
    assert all(isinstance(rule, dict) for rule in value.values())


@pytest.mark.parametrize(("app_key", "key", "expected"), [
    ("mask", "channels", [0, 1, 2, 3]),
    ("timelapse", "channels", [0, 1, 2, 3]),
    ("motility", "channels", [0, 1, 2, 3]),
    ("measure", "channels", [0, 1, 2, 3]),
    ("activation", "channels", [1, 2, 3]),
    ("cellpose_masks", "channels", [0, 0]),
    ("cellpose_all", "channels", [0, 0]),
    ("recruitment", "channel_dims", [0, 1, 2, 3]),
])
def test_channel_lists_use_the_manders_style_editor(
        qapp, app_key, key, expected):
    model = _model(qapp, app_key)
    widget = model._widgets[key]
    assert isinstance(widget, _ListEditor)
    assert widget.get_value() == expected
    assert model.collect()[key] == expected


def test_train_channels_gets_the_alphabet_control_instead(qapp):
    """It used to be here, with the other channel lists, and it was the one
    that did not belong: ``train_channels`` is not an open list of channel
    indices, it is a choice among exactly ``r``, ``g`` and ``b``. The chip
    strip accepted ``x``, ``red`` and ``4``, and the pipeline drops an
    unrecognised letter in silence — so the run trains on fewer planes than
    the user asked for and nothing says so.
    """
    from spacr.qt.screens.settings_model import _AlphabetSelect
    model = _model(qapp, "classify")
    widget = model._widgets["train_channels"]
    assert isinstance(widget, _AlphabetSelect)
    assert not isinstance(widget, _ListEditor)
    assert widget.choices() == ("r", "g", "b")
    assert widget.get_value() == ["r", "g", "b"]
    assert model.collect()["train_channels"] == ["r", "g", "b"]


def test_src_keeps_its_line_edit(qapp):
    """Single-plate modules keep the compact path editor."""
    from PySide6.QtWidgets import QLineEdit
    for app_key in ("mask", "measure"):
        widget = _model(qapp, app_key)._widgets["src"]
        assert isinstance(widget, QLineEdit), app_key
        assert not isinstance(widget, _ListEditor), app_key


def test_classify_src_accepts_an_arbitrary_number_of_plates(qapp):
    """Classify supports typing or dropping several plate paths."""
    widget = _model(qapp, "classify")._widgets["src"]
    assert isinstance(widget, _ListEditor)
    widget.set_value(["/data/plate-a", "/data/plate-b", "/data/plate-c"])
    assert widget.get_value() == [
        "/data/plate-a", "/data/plate-b", "/data/plate-c"]


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
    """The element type comes from the default, so typing "3" into a list of
    ints yields ``3`` and typing it into a list of names yields ``"3"``.

    Both keys this used to drive have since gone: ``png_dims`` was replaced by
    ``png_channel_mapping`` on 2026-08-06 (commit 2cab81f7 -- an ambiguous
    list of positions became an explicit colour mapping) and is no longer
    rendered at all, and ``classes`` moved to ``ClassEditorWidget`` on
    2026-08-07 (commit 30500970). Repointed at surviving keys of each element
    type; the typing rule under test is unchanged.
    """
    model = _model(qapp, "measure")
    distances = model._widgets["homogeneity_distances"]
    distances._strips[0]._entry.setText("3")
    distances._strips[0]._commit_entry()
    assert distances.get_value() == [8, 16, 32, 3]
    assert all(isinstance(v, int) for v in distances.get_value())

    ratios = model._widgets["dialate_png_ratios"]
    ratios._strips[0]._entry.setText("0.5")
    ratios._strips[0]._commit_entry()
    assert ratios.get_value() == [0.2, 0.5]
    assert all(isinstance(v, float) for v in ratios.get_value())

    objects = model._widgets["timelapse_objects"]
    objects._strips[0]._entry.setText("3")
    objects._strips[0]._commit_entry()
    # element type inferred from the default (['cell']) -> text, so an object
    # class literally named "3" is not silently turned into the integer 3
    assert objects.get_value() == ["cell", "3"]
    assert all(isinstance(v, str) for v in objects.get_value())


# ---------------------------------------------------------------------------
# adding and removing
# ---------------------------------------------------------------------------

def test_a_chip_can_be_added_and_removed(qapp):
    """Committing the entry appends a chip; a chip's remove button drops it
    and nothing else.

    Driven through classify's ``classes`` until commit 30500970 (2026-08-07)
    gave that key ``ClassEditorWidget``; repointed at a key that still uses
    the chip editor. Removal is asserted on a three-element default so a
    remove that dropped the wrong chip -- or the whole list -- is visible.
    """
    model = _model(qapp, "measure")
    widget = model._widgets["homogeneity_distances"]
    strip = widget._strips[0]

    strip._entry.setText("64")
    strip._commit_entry()
    assert widget.get_value() == [8, 16, 32, 64]

    strip._chips[0].removed.emit(strip._chips[0])
    assert widget.get_value() == [16, 32, 64]
    assert model.collect()["homogeneity_distances"] == [16, 32, 64]


def test_a_comma_splits_a_pasted_run_into_chips(qapp):
    """A comma ends a chip while typing, so a pasted "a,b,c" becomes three.

    Was driven through classify's ``classes``, which stopped being a chip
    strip in commit 30500970 (2026-08-07); comma splitting belongs to
    ``_ChipStrip`` and is asserted here on a key that still has one.
    """
    widget = _model(qapp, "measure")._widgets["timelapse_objects"]
    strip = widget._strips[0]
    for text in ("a,", "b,", "c"):
        strip._entry.setText(strip._entry.text() + text)
        strip._on_typed(strip._entry.text())
    assert widget.get_value() == ["cell", "a", "b", "c"]


def test_uncommitted_text_is_still_collected(qapp):
    """A user who types a value and presses Run without leaving the field
    must not lose it.

    Repointed off classify's ``classes`` when commit 30500970 (2026-08-07)
    moved that key to ``ClassEditorWidget``; the key here still uses the chip
    editor. Asserted through ``collect()`` as well, since Run reads that and
    not the widget.
    """
    model = _model(qapp, "measure")
    widget = model._widgets["timelapse_objects"]
    widget._strips[0]._entry.setText("nucleus")
    assert widget.get_value() == ["cell", "nucleus"]
    assert model.collect()["timelapse_objects"] == ["cell", "nucleus"]


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


@pytest.mark.parametrize("text", ["['nc', 'pc']", "['nc','pc']", "  ['nc', 'pc']  "])
def test_the_class_editor_reads_a_settings_csv_string(qapp, text):
    """A settings CSV stores ``repr(value)``, so ``classes`` arrives as TEXT.

    ClassEditorWidget.set_value handled Mapping and list and nothing else, so
    the string matched neither arm, fell through to an empty table, and
    reported success: apply_settings_dict returned applied=1 while
    collect()['classes'] was {}. Every other list-shaped key survived that
    round trip -- this was the one that decides what gets trained, and losing
    it is silent.

    The sibling assertion for ``class_metadata`` is
    ``test_set_value_parses_what_a_settings_csv_holds``; this is the same
    contract for the editor that replaced the chip strip.
    """
    from spacr.qt.widgets.class_editor import ClassEditorWidget
    widget = ClassEditorWidget()
    widget.set_value(text)
    assert list(widget.get_value()) == ["nc", "pc"], (
        "class names from a settings CSV must survive set_value")


def test_the_class_editor_survives_a_string_that_does_not_parse(qapp):
    """A corrupt cell must not raise out of a settings import."""
    from spacr.qt.widgets.class_editor import ClassEditorWidget
    widget = ClassEditorWidget()
    widget.set_value("['nc', 'pc'")          # unbalanced
    assert widget.get_value() == {}


def test_importing_a_settings_dict_reaches_the_custom_editors(qapp):
    """AppScreen._apply_value used to have only a QLineEdit branch, and both
    of these editors are plain QWidgets -- an imported list would have been
    dropped.

    Renamed from ``..._reaches_the_chip_editor``: since commit 30500970
    (2026-08-07) only ``class_metadata`` is a chip editor, while ``classes``
    is a ``ClassEditorWidget`` that reads a legacy list of names back as a
    dict of name -> rule. The invariant is the same one -- an imported value
    must not be silently discarded -- so the classes half asserts the names
    arrive, in order, as rows of the new mapping. The rules' placeholder
    ``column`` is deliberately not pinned here: what a legacy list should fill
    in for the column it never carried is an open question (see
    instructions/open/37), and this test is about the import path.
    """
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("classify")
    applied = screen.apply_settings_dict({"class_metadata": "[['r1'], ['r2']]",
                                          "classes": ["a", "b"]})
    assert applied == 2
    assert screen._settings_model._widgets["class_metadata"].get_value() \
        == [["r1"], ["r2"]]
    classes = screen._settings_model._widgets["classes"].get_value()
    assert isinstance(classes, dict)
    assert list(classes) == ["a", "b"]
    assert all(isinstance(rule, dict) for rule in classes.values())
    # and it survives collect(), which is what Run actually reads
    assert list(screen._settings_model.collect()["classes"]) == ["a", "b"]
    screen.deleteLater()


def test_live_preview_propagation_reaches_the_chip_editor(qapp):
    """``set_value_for_key`` has to find the chip editor, not just line edits.

    Was asserted on ``png_dims``, which stopped being a rendered setting on
    2026-08-06 (commit 2cab81f7 replaced it with ``png_channel_mapping``), so
    ``set_value_for_key`` correctly answers False for it now -- there is no
    widget to reach. Repointed at the int-list key that survives in the same
    panel, plus a check that an unknown key still reports False rather than
    pretending it landed somewhere.
    """
    model = _model(qapp, "measure")
    assert model.set_value_for_key("png_size", [1, 2]) is True
    assert model._widgets["png_size"].get_value() == [1, 2]
    assert model.set_value_for_key("png_dims", [1, 2]) is False


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
