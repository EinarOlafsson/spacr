"""The Classes editor, and the settings it makes redundant."""
import pandas as pd
import pytest


@pytest.fixture
def frame():
    return pd.DataFrame({
        "annot_1": [1, 1, 2, None, 2],
        "annot_2": [0, 0, 0, 1, 1],
        "plateID": ["p1"] * 5,
        "columnID": ["c1", "c1", "c3", "c3", "c3"],
        "area": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


def _editor(qtbot, frame, **kwargs):
    from spacr.qt.widgets.class_editor import ClassEditorWidget

    widget = ClassEditorWidget(frame=frame, **kwargs)
    qtbot.addWidget(widget)
    return widget


def test_choosing_a_column_fills_the_table_with_its_values(qtbot, frame):
    """"you set the column then the keys of this dict get populated and the
    user fills in their names"."""
    editor = _editor(qtbot, frame)
    editor.column.setCurrentText("annot_1")
    editor.populate_from_column()

    value = editor.value()
    assert len(value) == 2, value
    assert all(spec["column"] == "annot_1" for spec in value.values())
    assert {spec["value"] for spec in value.values()} == {1, 2}


def test_unannotated_rows_do_not_become_a_class(qtbot, frame):
    editor = _editor(qtbot, frame)
    editor.column.setCurrentText("annot_1")
    editor.populate_from_column()
    assert not any(spec["value"] is None for spec in editor.value().values())


def test_a_second_column_adds_to_the_table(qtbot, frame):
    """The thing the old settings could not express at all."""
    editor = _editor(qtbot, frame)
    editor.column.setCurrentText("annot_1")
    editor.populate_from_column()
    editor.column.setCurrentText("annot_2")
    editor.populate_from_column()

    columns = {spec["column"] for spec in editor.value().values()}
    assert columns == {"annot_1", "annot_2"}


def test_re_adding_a_column_does_not_duplicate_or_reset_names(qtbot, frame):
    editor = _editor(qtbot, frame)
    editor.column.setCurrentText("annot_1")
    editor.populate_from_column()
    editor.table.topLevelItem(0).setText(0, "infected")

    editor.populate_from_column()
    assert len(editor.value()) == 2
    assert "infected" in editor.value()


def test_the_user_names_each_class(qtbot, frame):
    editor = _editor(qtbot, frame)
    editor.column.setCurrentText("annot_1")
    editor.populate_from_column()
    editor.table.topLevelItem(0).setText(0, "infected")

    value = editor.value()
    assert "infected" in value
    assert value["infected"]["column"] == "annot_1"


def test_a_blank_name_is_refused(qtbot, frame):
    """A class with no name cannot be trained on or reported."""
    editor = _editor(qtbot, frame)
    editor.column.setCurrentText("annot_1")
    editor.populate_from_column()
    before = list(editor.value())

    editor.table.topLevelItem(0).setText(0, "   ")
    assert list(editor.value()) == before


def test_a_random_rest_class_can_be_added(qtbot, frame):
    editor = _editor(qtbot, frame)
    editor.column.setCurrentText("annot_1")
    editor.populate_from_column()
    editor.add_random_complement()

    rest = [name for name, spec in editor.value().items()
            if spec.get("random_complement")]
    assert len(rest) == 1


def test_only_one_random_rest_is_allowed(qtbot, frame):
    editor = _editor(qtbot, frame)
    editor.add_random_complement()
    editor.add_random_complement()
    rest = [s for s in editor.value().values() if s.get("random_complement")]
    assert len(rest) == 1


def test_the_metadata_basis_offers_the_plate_coordinates(qtbot, frame):
    """Which is what replaces location_column and the two control wells."""
    editor = _editor(qtbot, frame, basis="metadata")
    offered = [editor.column.itemText(i) for i in range(editor.column.count())]
    assert offered == ["plateID", "columnID"]
    assert "annot_1" not in offered


def test_a_control_well_is_just_a_class_on_a_metadata_column(qtbot, frame):
    editor = _editor(qtbot, frame, basis="metadata")
    editor.column.setCurrentText("columnID")
    editor.populate_from_column()
    editor.table.topLevelItem(0).setText(0, "negative control")
    editor.table.topLevelItem(1).setText(0, "positive control")

    value = editor.value()
    assert value["negative control"] == {"column": "columnID", "value": "c1"}
    assert value["positive control"] == {"column": "columnID", "value": "c3"}


def test_a_free_form_column_is_refused_with_advice(qtbot, frame):
    editor = _editor(qtbot, frame)
    # Past a hundred distinct values a column is a measurement, not a label.
    editor.set_frame(pd.DataFrame({"area": range(500)}))
    editor.column.setCurrentText("area")
    editor.populate_from_column()

    assert "Gate Editor" in editor._hint.text(), editor._hint.text()
    assert editor.value() == {}, "a measurement became 500 classes"


def test_an_existing_dict_is_shown_for_editing(qtbot, frame):
    editor = _editor(qtbot, frame, value={
        "infected": {"column": "annot_1", "value": 1}})
    assert editor.table.topLevelItemCount() == 1
    assert editor.value() == {"infected": {"column": "annot_1", "value": 1}}


def test_the_old_list_of_names_is_shown_rather_than_dropped(qtbot, frame):
    """So the user sees what has to be filled in, instead of an empty table."""
    editor = _editor(qtbot, frame, value=["nc", "pc"])
    assert editor.table.topLevelItemCount() == 2
    assert set(editor.value()) == {"nc", "pc"}


def test_the_panel_reads_it_through_get_value(qtbot, frame):
    editor = _editor(qtbot, frame, value={
        "a": {"column": "annot_1", "value": 1}})
    assert editor.get_value() == editor.value()


def test_a_row_can_be_removed(qtbot, frame):
    editor = _editor(qtbot, frame)
    editor.column.setCurrentText("annot_1")
    editor.populate_from_column()
    editor.table.setCurrentItem(editor.table.topLevelItem(0))
    editor.remove_selected()
    assert len(editor.value()) == 1


# ---------------------------------------------------------------------------
# The settings the dict makes redundant
# ---------------------------------------------------------------------------

def test_the_merged_classify_module_no_longer_offers_the_control_wells():
    """"the logic in Classes should remove the need to have location column,
    positive controll and negative controll settings here"."""
    from spacr.settings import set_default_classify

    settings = set_default_classify({})
    for retired in ("location_column", "positive_control",
                    "negative_control"):
        assert retired not in settings, f"{retired} is still offered"


def test_classify_ml_on_its_own_still_offers_them():
    """Dropped from the MERGED module only -- the standalone ML module has
    its own screen and its own users."""
    from spacr.settings import set_default_analyze_screen

    settings = set_default_analyze_screen({})
    assert "location_column" in settings


def test_an_old_csv_using_the_control_wells_still_trains_on_them():
    """They become class rules before anything reads them."""
    from spacr.classify_classes import assign_classes, normalize_settings

    old = {"dataset_mode": "metadata", "location_column": "columnID",
           "negative_control": "c1", "positive_control": "c3",
           "classes": ["nc", "pc"]}
    frame = pd.DataFrame({"columnID": ["c1", "c1", "c3"]})
    labels = assign_classes(frame, normalize_settings(old))
    assert list(labels) == ["nc", "nc", "pc"]
