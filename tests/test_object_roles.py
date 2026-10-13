"""One registry says what object kinds exist; the orders stay local.

Eleven modules used to spell the vocabulary out independently, and only two
of them derived from another. They agreed on MEMBERSHIP and disagreed on
ORDER -- three orderings of the same five names, plus two four-name variants
that leave out cytoplasm because it is derived rather than segmented.

Adding a sixth kind therefore meant finding all eleven, and missing one
produced a column that silently vanished from a model matrix rather than an
error. That is the problem instruction 76 has to solve before a second
organelle can exist.

These tests pin two things:

  * membership is single-source -- no module may name a kind the registry
    does not have;
  * the per-module ORDER is unchanged, because plane order, table order and
    crop order are each load-bearing where they live and centralising them
    would be a silent behaviour change dressed as a cleanup.
"""

import importlib

import pytest

from spacr.object_roles import (
    ALL_ROLES, DERIVED_ROLES, ORGANELLE_ROLES, SEGMENTED_ROLES,
    is_segmented, ordered,
)

ORG2 = ORGANELLE_ROLES[1:]


#: The exact tuples as they were before the registry existed, captured from a
#: running import. A change here is a behaviour change, not a refactor.
FROZEN = {
    ("measure_hooks", "OBJECT_TYPES"):
        ("cell", "nucleus", "pathogen", "organelle", *ORG2, "cytoplasm"),
    ("feature_dict", "OBJECT_TYPES"):
        ("cell", "nucleus", "pathogen", "organelle", *ORG2, "cytoplasm"),
    ("schema", "OBJECT_TYPES"):
        ("cell", "cytoplasm", "nucleus", "pathogen", "organelle", *ORG2),
    ("schema", "OBJECT_TABLES"):
        ("cell", "cytoplasm", "nucleus", "pathogen", "organelle", *ORG2),
    ("crops", "OBJECT_TYPES"):
        ("cell", "nucleus", "pathogen", "organelle", *ORG2, "cytoplasm"),
    ("io", "CROP_OBJECT_TYPES"):
        ("cell", "nucleus", "pathogen", "cytoplasm", "organelle", *ORG2),
    ("measure", "CROP_MODES"):
        ("cell", "nucleus", "pathogen", "cytoplasm", "organelle", *ORG2),
    ("filters", "OBJECT_TABLES"):
        ("cell", "nucleus", "pathogen", "cytoplasm", "organelle", *ORG2),
    ("merge_tables", "OBJECT_TABLES"):
        ("cell", "nucleus", "pathogen", "cytoplasm", "organelle", *ORG2),
    ("diameter", "OBJECT_TYPES"):
        ("cell", "nucleus", "pathogen", "organelle", *ORG2),
    ("validate", "OBJECT_NAMES"):
        ("cell", "nucleus", "pathogen", "organelle", *ORG2),
}


@pytest.mark.parametrize(("module", "name"), sorted(FROZEN))
def test_each_modules_order_is_unchanged(module, name):
    """The registry unified MEMBERSHIP, not order.

    Each of these orders means something where it lives -- the merged-array
    plane order, the object-table order, the order crops are written in. A
    refactor that quietly reordered one would change output while looking
    like tidying.
    """
    value = tuple(getattr(importlib.import_module(f"spacr.{module}"), name))
    assert value == FROZEN[(module, name)]


@pytest.mark.parametrize(("module", "name"), sorted(FROZEN))
def test_no_module_names_a_kind_the_registry_lacks(module, name):
    """Membership is single-source: this is what makes a sixth kind possible."""
    value = set(getattr(importlib.import_module(f"spacr.{module}"), name))
    stray = sorted(value - set(ALL_ROLES))
    assert not stray, f"spacr.{module}.{name} names {stray}, absent from ALL_ROLES"


def test_the_four_name_lists_are_exactly_the_segmented_roles():
    """diameter and validate omit cytoplasm for a REASON, not by accident.

    Cytoplasm is cell-minus-the-rest: it has no channel to validate and no
    diameter to measure. Pinning this stops someone "fixing the
    inconsistency" by adding it back.
    """
    from spacr import diameter, validate
    assert tuple(diameter.OBJECT_TYPES) == SEGMENTED_ROLES
    assert tuple(validate.OBJECT_NAMES) == SEGMENTED_ROLES
    assert "cytoplasm" not in SEGMENTED_ROLES
    assert "cytoplasm" in DERIVED_ROLES


def test_segmented_and_derived_partition_all_roles():
    assert set(SEGMENTED_ROLES) | set(DERIVED_ROLES) == set(ALL_ROLES)
    assert not set(SEGMENTED_ROLES) & set(DERIVED_ROLES)
    assert len(set(ALL_ROLES)) == len(ALL_ROLES), "a role is listed twice"


def test_is_segmented_answers_for_every_role():
    assert all(is_segmented(role) for role in SEGMENTED_ROLES)
    assert not any(is_segmented(role) for role in DERIVED_ROLES)


def test_is_segmented_is_false_for_an_unknown_name_rather_than_raising():
    """Callers ask this to decide whether to look for a channel setting."""
    assert is_segmented("sausage") is False


def test_ordered_preserves_what_it_is_given():
    assert ordered("cytoplasm", "cell") == ("cytoplasm", "cell")
    assert ordered() == ()


def test_ordered_refuses_a_typo_and_names_it():
    with pytest.raises(ValueError, match="sausage"):
        ordered("cell", "sausage")
