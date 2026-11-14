"""Table-driven tests of how ``spacr.gene_facts`` shows a cell and a record.

Every value on a gene tile comes out of a merged frame, so it arrives as a
numpy scalar, sometimes an array, sometimes an object whose unwrap is broken;
none of those may raise inside a tile that is already painting.  The text form
of a record is what a log records, so it has to carry the same sections the Qt
tile does, separated, and an empty record must still say something.  One
general test per unit, each walking a case table with ``subtests``.
"""
from __future__ import annotations

import numpy as np

from spacr import gene_facts


class _BrokenItem:
    """An object whose ``item`` raises, as an odd cell type can."""

    def __init__(self, error):
        self.error = error

    def item(self):
        raise self.error


def test_a_cell_is_unwrapped_to_a_python_scalar_or_shown_as_it_is(subtests):
    """Numpy scalars unwrap; arrays, broken unwraps and plain values pass through."""
    array = np.array([1, 2])
    broken_attr = _BrokenItem(AttributeError("no scalar here"))
    broken_value = _BrokenItem(ValueError("not one value"))
    plain = object()
    cases = [
        # (case id, cell, expected value, expected type or None for identity)
        ("numpy_float", np.float64(2.5), 2.5, float),
        ("numpy_bool", np.bool_(True), True, bool),
        ("numpy_int", np.int64(3), 3, int),
        ("size_one_array", np.array([7]), 7, int),
        ("multi_valued_array", array, array, None),
        ("item_raises_attribute_error", broken_attr, broken_attr, None),
        ("item_raises_value_error", broken_value, broken_value, None),
        ("python_string", "SRS40F", "SRS40F", str),
        ("none", None, None, type(None)),
        ("object_without_item", plain, plain, None),
    ]
    for case_id, cell, expected, kind in cases:
        with subtests.test(msg=case_id):
            out = gene_facts._plain(cell)
            if kind is None:
                assert out is expected
            else:
                assert out == expected
                assert type(out) is kind


def test_the_text_form_renders_sections_or_its_reason(subtests):
    """Sections are upper-cased, indented and blank-line separated; empty says why."""
    cases = [
        ("two_sections",
         gene_facts.GeneFacts(gene="224750",
                              values={"gene_name": "SRS40F",
                                      "topology": "SP"}),
         "IDENTITY\n  gene name: SRS40F\n\n"
         "MEMBRANE TOPOLOGY\n  DeepTMHMM class: SP"),
        ("one_section",
         gene_facts.GeneFacts(gene="224750", values={"gene_name": "SRS40F"}),
         "IDENTITY\n  gene name: SRS40F"),
        ("empty_with_reason",
         gene_facts.GeneFacts(gene="1", reason="no row in the annotation"),
         "no row in the annotation"),
        ("empty_without_reason", gene_facts.GeneFacts(),
         "Nothing is known about this gene."),
    ]
    for case_id, facts, expected in cases:
        with subtests.test(msg=case_id):
            assert facts.to_text() == expected
