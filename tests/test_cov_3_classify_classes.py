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
# Applying them
# ---------------------------------------------------------------------------

def test_assigning_classes_with_none_defined_refuses_before_it_labels():
    """An empty label column would be handed to the trainer as a dataset of
    NaN; the refusal names what has to be set instead."""
    frame = pd.DataFrame({"condition": ["nc", "pc"]})

    with pytest.raises(cc.ClassDefinitionError) as excinfo:
        cc.assign_classes(frame, {})

    assert "no classes are defined" in str(excinfo.value)
