"""A refusal example names how many values it is showing, not just three.

When a text identifier differs within a group, the merge refuses to pick one
of its values and says so with an example. The example is truncated to three
values, and a reader who sees three values has no way to know whether the
column held three or thirty unless the sentence says the count. Without the
count the refusal understates the problem it is reporting.
"""
from __future__ import annotations

from spacr.plate_measurements import describe_identifier_refusal


def test_a_truncated_example_says_how_many_values_it_hid():
    """Three shown out of eight has to read as eight, or the sentence
    misrepresents the scale of the disagreement it is refusing to resolve."""
    detail = {
        "groups": 4,
        "examples": [(("plate1", "A01"),
                      [f"object_{i}" for i in range(8)])],
    }
    line = describe_identifier_refusal("cell", "prcfo", detail)

    assert "(8 values)" in line, line
    assert "object_0, object_1, object_2" in line, line
    assert "object_3" not in line, "the fourth value was not truncated away"
    assert "plate1/A01" in line, line
    assert line.endswith(".")


def test_an_example_of_exactly_three_values_states_no_count():
    """The count is there to signal truncation. Printing it when nothing was
    truncated would tell the reader something was hidden when nothing was."""
    detail = {"groups": 1, "examples": [(("plate1", "A01"), ["a", "b", "c"])]}
    line = describe_identifier_refusal("cell", "prcfo", detail)

    assert "a, b, c" in line
    assert "values)" not in line, line
