"""`evaluate_barcode_regex`: a barcode regex is only usable if it captures all three groups.

An empty field asks for an expression rather than failing it; a regex that
does not match the read, or matches with an empty group, is refused with the
group named; a full capture is accepted.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.barcode_regex import evaluate_barcode_regex  # noqa: E402

pytestmark = pytest.mark.qt

_ALL_THREE = r"(?P<columnID>[ACGT]{2})(?P<grna>[ACGT]*)(?P<rowID>[ACGT]{2})"


def test_evaluate_barcode_regex_answers_each_kind_of_expression(subtests):
    """(valid, message, captures) for blank, unmatched, empty-group and full captures."""
    for case, pattern, sample, expected in (
            ("an empty field", "", "", (False, "Enter a regular expression.", {})),
            ("a blank field", "   ", "", (False, "Enter a regular expression.", {})),
            ("no field at all", None, "", (False, "Enter a regular expression.", {})),
            ("no match against the read", _ALL_THREE, "NNNNNNNN",
             (False, "The sample sequence did not match.", {})),
            ("a match with an empty group", _ALL_THREE, "ACGT",
             (False, "Matched, but these groups were empty: grna",
              {"columnID": "AC", "grna": "", "rowID": "GT"})),
            ("all three captured", _ALL_THREE, "ACGGGGTT",
             (True, "Sample matched successfully.",
              {"columnID": "AC", "grna": "GGGG", "rowID": "TT"})),
    ):
        with subtests.test(msg=case):
            result = evaluate_barcode_regex(pattern, sample)
            assert (result.valid, result.message, result.captures) == expected
