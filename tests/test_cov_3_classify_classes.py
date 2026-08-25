"""Class definitions refuse the shapes that would train a model on nothing.

Every refusal in this module protects a training run from starting on labels
nobody chose: a class with no name, a free-form measurement offered as a set
of labels, a settings file whose annotation half is blank, and a frame handed
to `assign_classes` with no rules at all. Each of those, allowed through,
produces a model that trains and reports success on the wrong thing.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr import classify_classes as cc


# ---------------------------------------------------------------------------
# A rule has to say what it is
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["", "   ", "\t\n"])
def test_a_class_without_a_name_is_refused(name):
    """The name is the folder the crops are written to and the label the
    model is trained with; a blank one silently becomes the directory the
    run is standing in."""
    with pytest.raises(cc.ClassDefinitionError) as excinfo:
        cc.ClassRule(name=name, column="condition", value="pc")

    assert "must have a name" in str(excinfo.value)


# ---------------------------------------------------------------------------
# A column has to be a label, and has to exist
# ---------------------------------------------------------------------------

def test_enumerating_a_column_the_table_lacks_names_the_column():
    """The Classes editor offers the values of a chosen column. A stale
    choice must name itself in the error, not raise KeyError from pandas."""
    frame = pd.DataFrame({"condition": ["nc", "pc"]})

    with pytest.raises(cc.ClassDefinitionError) as excinfo:
        cc.values_in(frame, "treatment")

    assert "'treatment'" in str(excinfo.value)
    assert "not in this table" in str(excinfo.value)


def test_a_free_form_column_is_refused_with_its_size():
    """Past the limit a column is a measurement. The count is in the message
    because it is what tells the user they picked the wrong column."""
    frame = pd.DataFrame({"cell_area": range(150)})

    with pytest.raises(cc.ClassDefinitionError) as excinfo:
        cc.values_in(frame, "cell_area", limit=100)

    assert "150 distinct values" in str(excinfo.value)
    assert "Gate Editor" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Deriving rules from a settings file
# ---------------------------------------------------------------------------

def test_an_annotation_run_with_no_column_derives_no_classes():
    """`dataset_mode=annotation` with nothing naming a column has no basis
    for a rule. Inventing one would train on a guessed label."""
    out = cc.normalize_settings({"dataset_mode": "annotation"})

    assert cc.class_rules(out) == ()
    assert out.get(cc.CLASSES) in (None, {}, [])


def test_a_single_annotation_value_given_as_a_string_becomes_one_class():
    """A settings CSV holds one value unbracketed. Iterating the string
    would make one class per character."""
    out = cc.normalize_settings({"dataset_mode": "annotation",
                                 "annotation_column": "test",
                                 "annotation_values": "positive"})
    rules = cc.class_rules(out)

    assert len(rules) == 1, [r.name for r in rules]
    assert rules[0].column == "test"
    assert rules[0].value == "positive"


def test_a_metadata_run_skips_the_control_that_is_not_set():
    """Only the positive control is named, so only one rule exists. An empty
    negative control must not become a rule matching the empty string."""
    out = cc.normalize_settings({"dataset_mode": "metadata",
                                 "location_column": "column_name",
                                 "negative_control": "",
                                 "positive_control": "c3"})
    rules = cc.class_rules(out)

    assert len(rules) == 1, [(r.name, r.value) for r in rules]
    assert rules[0].value == "c3"
    assert rules[0].column == "column_name"


# ---------------------------------------------------------------------------
# Applying them
# ---------------------------------------------------------------------------

def test_assigning_classes_with_none_defined_refuses_before_it_labels():
    """An empty label column would be handed to the trainer as a dataset of
    NaN; the refusal names what has to be set instead."""
    frame = pd.DataFrame({"condition": ["nc", "pc"]})

    with pytest.raises(cc.ClassDefinitionError) as excinfo:
        cc.assign_classes(frame, {})

    assert "no classes are defined" in str(excinfo.value)
