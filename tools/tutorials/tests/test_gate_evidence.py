"""Equal gate totals must not conceal marks on another object or type."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate_evidence import check_exported_rows


def population():
    return [('plate1', 'r12', 'c2', 'f17', 1, 'cell'),
            ('plate1', 'r12', 'c2', 'f18', 1, 'cell')]


def test_same_objects_in_another_order_are_the_same_gate():
    expected = population()
    assert check_exported_rows(expected, list(reversed(expected))) == 2


@pytest.mark.parametrize('position,value', [(0, 'plate2'), (1, 'r11'), (2, 'c3'),
                                           (3, 'f19'), (4, 2), (5, 'nucleus')])
def test_equal_count_on_another_identity_axis_is_not_the_same_gate(position, value):
    expected = population()
    changed = list(expected[0])
    changed[position] = value
    with pytest.raises(ValueError, match='identities or types'):
        check_exported_rows(expected, [tuple(changed), expected[1]])


def test_repeated_objects_are_not_a_larger_population():
    expected = population()
    with pytest.raises(ValueError, match='duplicate'):
        check_exported_rows(expected, expected + [expected[0]])


def test_a_missing_mark_is_not_the_same_gate():
    with pytest.raises(ValueError, match='identities or types'):
        check_exported_rows(population(), population()[:1])
