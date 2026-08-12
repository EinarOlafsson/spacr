"""Classes as a dict of name -> (column, value).

The load-bearing test in here is the backward-compatibility one: a settings
CSV written before this exists in every user's project folder, and a run from
one has to select the same objects it did before. Everything else is about
what the old shape could not say.
"""
import numpy as np
import pandas as pd
import pytest

from spacr.classify_classes import (
    METADATA_COLUMNS, ClassDefinitionError, ClassRule, assign_classes,
    candidate_columns, class_names, class_rules, normalize_settings,
    values_in,
)


def _annotated(n=20):
    return pd.DataFrame({
        "annot_1": [1] * 5 + [2] * 5 + [None] * 10,
        "annot_2": [0] * 10 + [1] * 10,
        "plateID": ["p1"] * n,
        "columnID": ["c1"] * 10 + ["c3"] * 10,
        "object_label": range(n),
    })


# ---------------------------------------------------------------------------
# The dict
# ---------------------------------------------------------------------------

def test_a_class_names_its_value_and_the_column_it_came_from():
    settings = {"classes": {"infected": {"column": "annot_1", "value": 1},
                            "clean": {"column": "annot_1", "value": 2}}}
    rules = class_rules(settings)
    assert [r.name for r in rules] == ["infected", "clean"]
    assert rules[0].column == "annot_1" and rules[0].value == 1


def test_classes_can_come_from_more_than_one_column():
    """The thing the old shape could not express at all."""
    settings = {"classes": {"a": {"column": "annot_1", "value": 1},
                            "b": {"column": "annot_2", "value": 1}}}
    labels = assign_classes(_annotated(), settings)
    assert set(labels.dropna()) == {"a", "b"}


def test_the_order_of_the_classes_is_kept():
    """It is the label order the model trains with."""
    settings = {"classes": {"z": {"column": "annot_1", "value": 1},
                            "a": {"column": "annot_1", "value": 2}}}
    assert class_names(settings) == ["z", "a"]


def test_a_class_with_no_column_is_refused():
    with pytest.raises(ClassDefinitionError, match="which column"):
        ClassRule(name="x", value=1)


def test_a_class_cannot_be_both_a_rule_and_the_complement():
    with pytest.raises(ClassDefinitionError, match="only be one"):
        ClassRule(name="x", column="c", value=1, random_complement=True)


def test_two_random_complements_are_refused():
    """Two classes that both mean 'everything else' have no boundary."""
    settings = {"classes": {"a": {"random_complement": True},
                            "b": {"random_complement": True}}}
    with pytest.raises(ClassDefinitionError, match="boundary"):
        class_rules(settings)


def test_the_old_list_shape_says_to_normalize_first():
    with pytest.raises(ClassDefinitionError, match="normalize_settings"):
        class_rules({"classes": ["nc", "pc"]})


# ---------------------------------------------------------------------------
# Populating the keys
# ---------------------------------------------------------------------------

def test_the_values_of_a_column_are_what_the_keys_come_from():
    assert set(values_in(_annotated(), "annot_1")) == {1, 2}


def test_unannotated_rows_are_not_offered_as_a_class():
    """'Not annotated' is the absence of a class; offering it is how a user
    trains on their own blanks."""
    values = values_in(_annotated(), "annot_1")
    assert not any(pd.isna(v) for v in values)


def test_a_free_form_column_is_refused_with_advice():
    frame = pd.DataFrame({"area": np.linspace(0, 1, 500)})
    with pytest.raises(ClassDefinitionError, match="Gate Editor"):
        values_in(frame, "area")


def test_the_metadata_basis_offers_the_plate_coordinates():
    columns = candidate_columns({"dataset_mode": "metadata"})
    assert columns == METADATA_COLUMNS


def test_a_metadata_column_the_table_lacks_is_not_offered():
    columns = candidate_columns({"dataset_mode": "metadata"},
                                available=["plateID", "columnID", "area"])
    assert columns == ("plateID", "columnID")


def test_the_annotation_basis_offers_the_tables_own_columns():
    columns = candidate_columns({"dataset_mode": "annotation"},
                                available=["annot_1", "annot_2"])
    assert columns == ("annot_1", "annot_2")


# ---------------------------------------------------------------------------
# The random complement -- what replaced write_random_annotation_column
# ---------------------------------------------------------------------------

def test_one_annotated_class_gets_a_random_comparison_group():
    """"if only one lable is present in the annotation column then the other
    class needs to be a random selection of the non annotated images"."""
    settings = {"classes": {"infected": {"column": "annot_1", "value": 1},
                            "control": {"random_complement": True}}}
    labels = assign_classes(_annotated(), settings)
    assert (labels == "infected").sum() == 5
    assert (labels == "control").sum() == 5, "the groups are lopsided"


def test_the_complement_never_takes_an_annotated_object():
    settings = {"classes": {"infected": {"column": "annot_1", "value": 1},
                            "clean": {"column": "annot_1", "value": 2},
                            "control": {"random_complement": True}}}
    frame = _annotated()
    labels = assign_classes(frame, settings)
    annotated = frame["annot_1"].notna()
    assert not (labels[annotated] == "control").any()


def test_the_complement_is_reproducible():
    """A training set that changes every time it is built cannot be compared
    with the run before it."""
    settings = {"classes": {"infected": {"column": "annot_1", "value": 1},
                            "control": {"random_complement": True}}}
    frame = _annotated()
    first = assign_classes(frame, settings, seed=7)
    second = assign_classes(frame, settings, seed=7)
    assert first.equals(second)
    assert not first.equals(assign_classes(frame, settings, seed=8))


def test_a_complement_with_nothing_left_says_so():
    settings = {"classes": {"a": {"column": "annot_2", "value": 0},
                            "b": {"column": "annot_2", "value": 1},
                            "rest": {"random_complement": True}}}
    with pytest.raises(ClassDefinitionError, match="already claimed"):
        assign_classes(_annotated(), settings)


# ---------------------------------------------------------------------------
# Old settings keep working -- the one that matters
# ---------------------------------------------------------------------------

def test_an_old_annotation_settings_dict_selects_the_same_objects():
    old = {"dataset_mode": "annotation", "annotation_column": "annot_1",
           "annotated_classes": [1, 2], "classes": ["nc", "pc"]}
    settings = normalize_settings(old)

    assert class_names(settings) == ["nc", "pc"]
    labels = assign_classes(_annotated(), settings)
    assert (labels == "nc").sum() == 5
    assert (labels == "pc").sum() == 5


def test_an_old_write_random_annotation_column_becomes_a_complement():
    old = {"dataset_mode": "annotation", "annotation_column": "annot_1",
           "annotated_classes": [1], "classes": ["pos", "neg"],
           "write_random_annotation_column": True}
    settings = normalize_settings(old)
    rules = class_rules(settings)
    assert [r.name for r in rules] == ["pos", "neg"]
    assert rules[1].random_complement is True

    labels = assign_classes(_annotated(), settings)
    assert (labels == "pos").sum() == 5
    assert (labels == "neg").sum() == 5


def test_old_metadata_controls_become_rules():
    """"the logic in Classes should remove the need to have location column,
    positive controll and negative controll settings"."""
    old = {"dataset_mode": "metadata", "location_column": "columnID",
           "negative_control": "c1", "positive_control": "c3",
           "classes": ["nc", "pc"]}
    settings = normalize_settings(old)
    labels = assign_classes(_annotated(), settings)
    assert (labels == "nc").sum() == 10
    assert (labels == "pc").sum() == 10


def test_a_control_naming_several_wells_becomes_several_rules():
    old = {"dataset_mode": "metadata", "location_column": "columnID",
           "negative_control": ["c1"], "positive_control": ["c3"],
           "classes": ["nc", "pc"]}
    settings = normalize_settings(old)
    assert class_names(settings) == ["nc", "pc"]


def test_normalize_never_modifies_what_it_was_given():
    old = {"dataset_mode": "annotation", "annotation_column": "annot_1",
           "annotated_classes": [1, 2], "classes": ["nc", "pc"]}
    before = dict(old)
    normalize_settings(old)
    assert old == before


def test_a_dict_that_is_already_new_is_left_alone():
    settings = {"classes": {"a": {"column": "annot_1", "value": 1}}}
    assert normalize_settings(settings)["classes"] == settings["classes"]


def test_names_with_nothing_selecting_them_are_not_invented():
    """A guessed column would train on the wrong labels and report success."""
    settings = normalize_settings({"classes": ["nc", "pc"]})
    assert settings["classes"] == ["nc", "pc"]
    assert class_names(settings) == ["nc", "pc"]


def test_downstream_still_gets_a_list_of_names():
    """deep_spacr and model_zoo read a list; they should not learn the dict."""
    settings = normalize_settings(
        {"classes": {"a": {"column": "annot_1", "value": 1},
                     "b": {"column": "annot_1", "value": 2}}})
    assert settings["class_names"] == ["a", "b"]


def test_a_rule_on_a_column_the_table_lacks_names_it():
    settings = {"classes": {"a": {"column": "ghost", "value": 1}}}
    with pytest.raises(ClassDefinitionError, match="ghost"):
        assign_classes(_annotated(), settings)


# ---------------------------------------------------------------------------
# The refusal has to name what is actually wrong (instruction 37)
# ---------------------------------------------------------------------------

def test_unbound_names_are_refused_without_blaming_the_caller():
    """Names that survive normalization arrive here as a list.

    `normalize_settings` translates the old shape only when the settings
    carry a basis to derive rules from. With none -- which is any settings
    file naming classes that were never bound to a column -- it deliberately
    leaves the names alone rather than guessing, and that is right. But
    `class_rules` then said "run normalize_settings first", sending the user
    to do the one thing they had just done, and saying nothing about the
    column that is actually missing.
    """
    settings = normalize_settings({"classes": ["nc", "pc"], "src": "/x"})
    assert not isinstance(settings["classes"], dict), (
        "this test is about the case normalize_settings cannot bind")

    with pytest.raises(ClassDefinitionError) as excinfo:
        class_rules(settings)

    message = str(excinfo.value)
    # Names the classes it could not bind...
    assert "'nc'" in message and "'pc'" in message
    # ...says what is missing...
    assert "nothing says which objects belong to them" in message
    # ...and what to do about it.
    assert "column" in message
    assert "annotation_column" in message


def test_the_refusal_still_mentions_normalization_for_a_raw_settings_dict():
    """A dict that never went through normalize_settings is the OTHER case.

    Both arrive at the same refusal, so it has to serve both: the message
    keeps pointing at normalization for the caller who really has skipped
    it.
    """
    with pytest.raises(ClassDefinitionError) as excinfo:
        class_rules({"classes": ["a", "b"]})

    assert "normalize_settings" in str(excinfo.value)


# ---------------------------------------------------------------------------
# `classes` carried two contracts; the folder names moved out (instruction 37)
# ---------------------------------------------------------------------------

def test_the_shipped_defaults_separate_the_two_meanings():
    """One key held both "what a class means" and "where its crops are"."""
    from spacr.settings import deep_spacr_defaults

    s = deep_spacr_defaults({})
    assert s["classes"] == {}, "no class is defined until the user defines one"
    assert s["class_folder_names"] == ["nc", "pc"]


def test_a_settings_file_written_before_the_split_still_trains():
    """`classes` as a list is what every settings CSV in the wild holds.

    It has to keep naming the training folders, or the split silently
    retrains every existing project against no classes at all.
    """
    from spacr.classify_classes import folder_names

    assert folder_names({"classes": ["alive", "dead"]}) == ["alive", "dead"]


def test_the_explicit_key_wins_over_the_legacy_one():
    from spacr.classify_classes import folder_names

    names = folder_names({"classes": ["old", "older"],
                          "class_folder_names": ["new", "newer"]})
    assert names == ["new", "newer"]


def test_defined_classes_name_their_own_folders():
    """With neither list set, the definitions are the answer, in order."""
    from spacr.classify_classes import folder_names

    names = folder_names({"classes": {
        "pc": {"column": "columnID", "value": "c3"},
        "nc": {"column": "columnID", "value": "c1"}}})
    assert names == ["pc", "nc"]


def test_folder_names_does_not_raise_on_a_malformed_definition():
    """Naming the folders is not the place to refuse a bad rule.

    `class_rules` raises on its own terms for whoever needs the rules; a
    listing helper that raised too would turn one bad row into a crash in
    unrelated code.
    """
    from spacr.classify_classes import folder_names

    assert folder_names({"classes": {"a": "not-a-mapping"}}) == []


def test_an_empty_definition_still_derives_from_the_older_keys():
    """`{}` is a Mapping, and that is the trap.

    The default is now an empty dict meaning "nothing defined yet". A
    normalize_settings that tested only the TYPE would skip its derivation
    for exactly those settings, so a plate carrying the retired
    location_column / control keys would train on no classes and say
    nothing.
    """
    out = normalize_settings({
        "classes": {},
        "class_folder_names": ["nc", "pc"],
        "location_column": "columnID",
        "negative_control": "c1",
        "positive_control": "c3",
    })

    assert isinstance(out["classes"], dict) and out["classes"], (
        "the retired keys were not translated")
    assert sorted(out["classes"]) == ["nc", "pc"]
    assert out["classes"]["pc"]["column"] == "columnID"
    assert out["classes"]["pc"]["value"] == "c3"


def test_generate_training_dataset_writes_folders_not_definitions():
    """io wrote the folder listing over `classes`, discarding the rules.

    Asserted on the source, because reaching this line needs a full dataset
    build: what matters is that the write targets the folder-name key.
    """
    import inspect
    from spacr import io

    source = inspect.getsource(io)
    assert "settings['class_folder_names'] = final_names" in source
    assert "settings['classes'] = final_names" not in source


def test_deep_spacr_reads_the_folder_names():
    """Every `classes=` argument in deep_spacr means the FOLDER names."""
    import inspect
    from spacr import deep_spacr

    source = inspect.getsource(deep_spacr)
    assert "classes=settings['classes']" not in source
    assert "classes=_class_folder_names(settings)" in source


def test_deep_spacr_resolves_a_legacy_settings_dict():
    from spacr.deep_spacr import _class_folder_names

    assert _class_folder_names({"classes": ["alive", "dead"]}) == [
        "alive", "dead"]
