"""The compatibility keys derived from class definitions, and their fallbacks.

``annotation_column`` and ``class_metadata`` are still read all over the
pipeline, so the Classes dict has to be able to produce them -- and has to
stand aside when it cannot. A dict that says nothing usable about a column
must leave the value a pre-Classes settings file already carries, because
overwriting it with an empty string is how an old settings CSV stops
selecting anything while still reporting a successful run.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr import classify_classes as cc


# ---------------------------------------------------------------------------
# annotation_column_of: the first usable column wins, and nothing else does
# ---------------------------------------------------------------------------

def test_a_class_defined_as_something_other_than_a_rule_is_skipped_over():
    """A hand-edited settings CSV can leave a bare string under a class name.
    It is not a rule, so it must not stop the search at the first key."""
    settings = {"classes": {"junk": "pc", "infected": {"column": "annot_1",
                                                       "value": 1}}}

    assert cc.annotation_column_of(settings) == "annot_1"


def test_a_rule_with_a_blank_column_is_skipped_over():
    """A class half-filled in the editor -- a value chosen, the column not
    yet -- must not become an empty annotation column downstream."""
    settings = {"classes": {"pending": {"column": "  ", "value": 1},
                            "infected": {"column": "annot_2", "value": 1}}}

    assert cc.annotation_column_of(settings) == "annot_2"


def test_no_rule_names_a_column_so_the_legacy_setting_still_decides():
    """The dict exists but says nothing about a column. The value the file
    was already running with is the right answer, not ''."""
    settings = {"classes": {"a": "junk", "b": {"column": "", "value": 1}},
                "annotation_column": "test"}

    assert cc.annotation_column_of(settings) == "test"


def test_with_neither_a_rule_nor_a_legacy_column_the_answer_is_empty():
    assert cc.annotation_column_of({"classes": {}}) == ""
    assert cc.annotation_column_of({}) == ""


def test_a_legacy_column_of_none_reads_as_empty_rather_than_the_string_none():
    """`annotation_column` defaults to None in one of the two modules this
    merged; `str(None)` would put the word 'None' into a SQL query."""
    assert cc.annotation_column_of({"annotation_column": None}) == ""


# ---------------------------------------------------------------------------
# class_metadata_of: the values, in class order, or the legacy list
# ---------------------------------------------------------------------------

def test_a_class_with_no_value_contributes_nothing_to_the_metadata():
    settings = {"classes": {"complement": {"random_complement": True},
                            "pc": {"column": "columnID", "value": "c3"}}}

    assert cc.class_metadata_of(settings) == [["c3"]]


def test_a_dict_where_no_class_has_a_value_falls_back_to_the_legacy_list():
    """Every rule is a complement or a half-filled row: the dict cannot
    describe the metadata, so the file's own list stands."""
    settings = {"classes": {"complement": {"random_complement": True},
                            "junk": "pc"},
                "class_metadata": [["c1"], ["c2"]]}

    assert cc.class_metadata_of(settings) == [["c1"], ["c2"]]


def test_no_values_anywhere_is_an_empty_list_not_a_none():
    assert cc.class_metadata_of({"classes": {}}) == []
    assert cc.class_metadata_of({"class_metadata": None}) == []


def test_a_legacy_metadata_tuple_is_returned_as_a_list():
    assert cc.class_metadata_of({"class_metadata": (["c1"], ["c2"])}) == \
        [["c1"], ["c2"]]


# ---------------------------------------------------------------------------
# fold_into_classes writes only what it actually derived
# ---------------------------------------------------------------------------

def test_folding_derives_both_compatibility_keys_from_the_classes():
    settings = {"classes": {"nc": {"column": "columnID", "value": "c1"},
                            "pc": {"column": "columnID", "value": "c3"}}}

    out = cc.fold_into_classes(settings)

    assert out is settings
    assert out["annotation_column"] == "columnID"
    assert out["class_metadata"] == [["c1"], ["c3"]]


def test_folding_a_settings_dict_with_nothing_to_derive_adds_no_keys():
    """Writing '' and [] would erase what a pre-Classes file is running on."""
    settings = {"src": "/data/plate1"}

    out = cc.fold_into_classes(settings)

    assert out == {"src": "/data/plate1"}
    assert "annotation_column" not in out
    assert "class_metadata" not in out


def test_folding_keeps_a_legacy_column_when_the_classes_only_give_values():
    """One half derivable and the other not: the derived half is written and
    the other is left exactly as the file had it."""
    settings = {"classes": {"pc": {"column": "", "value": "c3"}},
                "annotation_column": "test"}

    out = cc.fold_into_classes(settings)

    assert out["annotation_column"] == "test"
    assert out["class_metadata"] == [["c3"]]


def test_folding_keeps_a_legacy_metadata_list_when_no_class_has_a_value():
    settings = {"classes": {"pc": {"column": "rowID"}},
                "class_metadata": [["r2"]]}

    out = cc.fold_into_classes(settings)

    assert out["annotation_column"] == "rowID"
    assert out["class_metadata"] == [["r2"]]


# ---------------------------------------------------------------------------
# Recording the folders dataset generation actually wrote
# ---------------------------------------------------------------------------

def test_recording_folders_leaves_a_definitions_dict_in_place():
    """The legacy list is dropped because `classes` meant the folder list in
    that spelling. A definitions dict means something else and must survive:
    dropping it would lose every column and value the user chose."""
    defined = {"nc": {"column": "columnID", "value": "c1"}}
    settings = {"classes": defined}

    recorded = cc._record_generated_folder_names(settings, ["nc", "pc"])

    assert recorded == ["nc", "pc"]
    assert settings["class_folder_names"] == ["nc", "pc"]
    assert settings["classes"] == defined


def test_recording_folders_with_no_classes_key_at_all_just_records():
    settings = {}

    cc._record_generated_folder_names(settings, ("a", "b"))

    assert settings == {"class_folder_names": ["a", "b"]}


def test_recording_folders_retires_the_ambiguous_legacy_list():
    settings = {"classes": ["nc", "pc"]}

    cc._record_generated_folder_names(settings, ["nc", "pc", "extra"])

    assert "classes" not in settings
    assert settings["class_folder_names"] == ["nc", "pc", "extra"]


# ---------------------------------------------------------------------------
# Deriving rules when the old keys are unevenly filled in
# ---------------------------------------------------------------------------

def test_more_annotation_values_than_names_get_a_name_from_the_value():
    """Three values and two folder names: the third class still has to be
    called something, and naming it after its value keeps it identifiable."""
    settings = {"dataset_mode": "annotation",
                "annotation_column": "annot_1",
                "annotation_values": [0, 1, 2],
                "class_folder_names": ["nc", "pc"]}

    rules = cc._rules_from_annotation(settings)

    assert [r.name for r in rules] == ["nc", "pc", "class_2"]
    assert [r.column for r in rules] == ["annot_1"] * 3


def test_more_values_than_columns_pair_off_against_the_first_column():
    """Several columns pair positionally; past the end the first column is
    what the old readers used."""
    settings = {"dataset_mode": "annotation",
                "annotation_columns": ["annot_1", "annot_2"],
                "annotation_values": [1, 1, 1],
                "class_folder_names": ["a", "b", "c"]}

    rules = cc._rules_from_annotation(settings)

    assert [r.column for r in rules] == ["annot_1", "annot_2", "annot_1"]


def test_a_random_complement_takes_the_first_name_no_class_has_taken():
    settings = {"dataset_mode": "annotation",
                "annotation_column": "annot_1",
                "annotation_values": [1],
                "write_random_annotation_column": True,
                "class_folder_names": ["infected", "uninfected"]}

    rules = cc._rules_from_annotation(settings)

    assert [r.name for r in rules] == ["infected", "uninfected"]
    assert rules[1].random_complement is True


def test_a_complement_with_every_name_already_used_is_called_random():
    settings = {"dataset_mode": "annotation",
                "annotation_column": "annot_1",
                "annotation_values": [0, 1],
                "write_random_annotation_column": True,
                "class_folder_names": ["nc", "pc"]}

    rules = cc._rules_from_annotation(settings)

    assert [r.name for r in rules] == ["nc", "pc", "random"]
    assert rules[-1].random_complement is True


def test_blank_annotation_column_entries_are_not_columns():
    settings = {"dataset_mode": "annotation", "annotation_columns": ["", "  "],
                "annotation_values": [1]}

    assert cc._rules_from_annotation(settings) == []


def test_metadata_controls_with_no_folder_names_are_named_after_the_setting():
    """Nothing recorded a folder list, so the control's own name is the only
    thing left to call the class."""
    settings = {"dataset_mode": "metadata", "location_column": "columnID",
                "negative_control": "c1", "positive_control": "c3"}

    rules = cc._rules_from_metadata(settings)

    assert [r.name for r in rules] == ["negative control", "positive control"]
    assert [r.value for r in rules] == ["c1", "c3"]


def test_a_metadata_run_with_no_location_column_derives_nothing():
    settings = {"dataset_mode": "metadata", "location_column": "   ",
                "positive_control": "c3"}

    assert cc._rules_from_metadata(settings) == []


# ---------------------------------------------------------------------------
# class_rules refuses what it cannot turn into a rule
# ---------------------------------------------------------------------------

def test_a_class_defined_as_a_bare_value_names_itself_in_the_refusal():
    with pytest.raises(cc.ClassDefinitionError) as excinfo:
        cc.class_rules({"classes": {"infected": 1}})

    assert "'infected'" in str(excinfo.value)
    assert "needs a column and a value" in str(excinfo.value)


def test_an_absent_classes_key_is_no_rules_rather_than_an_error():
    assert cc.class_rules({}) == ()
    assert cc.class_rules({"classes": {}}) == ()
    assert cc.class_rules({"classes": None}) == ()


# ---------------------------------------------------------------------------
# Assigning the labels
# ---------------------------------------------------------------------------

def test_the_first_rule_wins_when_two_claim_the_same_object():
    """Silently relabelling is worse than keeping the order the user wrote."""
    frame = pd.DataFrame({"a": [1, 1, 0], "b": [1, 0, 0]})
    settings = {"classes": {"first": {"column": "a", "value": 1},
                            "second": {"column": "b", "value": 1}}}

    labels = cc.assign_classes(frame, settings)

    assert labels.tolist()[:2] == ["first", "first"]
    assert pd.isna(labels.iloc[2])


def test_a_complement_with_no_explicit_class_takes_the_whole_pool():
    """No rule labelled anything, so there is no largest class to match; the
    comparison group is everything there is."""
    frame = pd.DataFrame({"a": [0, 0, 0, 0]})
    settings = {"classes": {"everything": {"random_complement": True}}}

    labels = cc.assign_classes(frame, settings, seed=0)

    assert list(labels) == ["everything"] * 4


def test_the_complement_is_capped_by_what_is_left_unclaimed():
    """Ten annotated objects and two unannotated ones: the comparison group
    is the two, not ten, and sampling ten without replacement would raise."""
    frame = pd.DataFrame({"a": [1] * 10 + [0, 0]})
    settings = {"classes": {"pc": {"column": "a", "value": 1},
                            "rest": {"random_complement": True}}}

    labels = cc.assign_classes(frame, settings, seed=0)

    assert (labels == "pc").sum() == 10
    assert (labels == "rest").sum() == 2


def test_the_complement_matches_the_largest_explicit_class():
    frame = pd.DataFrame({"a": [1, 1, 1] + [0] * 20})
    settings = {"classes": {"pc": {"column": "a", "value": 1},
                            "rest": {"random_complement": True}}}

    labels = cc.assign_classes(frame, settings, seed=7)

    assert (labels == "pc").sum() == 3
    assert (labels == "rest").sum() == 3
    assert labels.isna().sum() == 17
