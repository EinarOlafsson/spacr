"""Collective labels represent exact slots without changing internal names."""
import re
import pytest
from spacr.schema import object_type_summary
from spacr.organelle_types import ALL_ORGANELLE_ROLES


@pytest.mark.parametrize('roles', [ALL_ORGANELLE_ROLES, ('organelle',), ('organelleb',),
                                  ('organelle', 'organelleb', 'organellec', 'organelled'),
                                  ('organelleb', 'organelled', 'organellez')])
def test_collective_pattern_matches_exactly_the_requested_slots(roles):
    pattern = object_type_summary(roles)
    assert pattern.count('organelle') == 1
    assert {role for role in ALL_ORGANELLE_ROLES if re.fullmatch(pattern, role)} == set(roles)
    assert not re.fullmatch(pattern, 'organellea')
    assert not re.fullmatch(pattern, 'organelleaaa')


def test_other_types_retain_order_without_expanding_slots():
    assert object_type_summary(('cell', *ALL_ORGANELLE_ROLES, 'nucleus', 'cell')) == 'cell, organelle(?:[b-z]|[a-z]{2})?, nucleus'
