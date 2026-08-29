"""The twenty regression families are grouped by what they assume.

Every one of them was already there -- the quantile fit, the two robust
losses and the rank aggregation included -- and none of them could be found,
because twenty unlabelled names in one alphabetical list is a menu that
hides its own contents. The grouping is the discoverability half; the stored
values do not move, so no settings file changes meaning.

The distinction the groups encode is the one a reviewer checks: `rlm`,
`huber` and `quantile` are ROBUST, not nonparametric. They fit a linear model
and are parametric in the coefficients. Only `rra` reads nothing but the
order of the wells.
"""
import pytest

from spacr.regression_families import (GROUP_TITLES,
                                       REGRESSION_FAMILY_ASSUMPTIONS,
                                       REGRESSION_FAMILY_GROUPS,
                                       family_group, family_label,
                                       regression_family_choices)
from spacr.regression_spec import (NO_P_VALUE_TYPES, REGRESSION_TYPES,
                                   UNSUPPORTED_REGRESSION_TYPES)

FITTABLE = {name for name in REGRESSION_TYPES
            if name not in UNSUPPORTED_REGRESSION_TYPES}


def test_every_family_that_fits_is_placed_exactly_once():
    """A family in the inventory and in no group would vanish from the menu;
    one in two groups would appear twice with two different claims."""
    placed = [name for _group, families in REGRESSION_FAMILY_GROUPS
              for name in families]

    assert len(placed) == len(set(placed)), "a family is in two groups"
    assert set(placed) == FITTABLE, (
        f"unplaced: {sorted(FITTABLE - set(placed))}; "
        f"placed but not fittable: {sorted(set(placed) - FITTABLE)}")
    assert len(placed) == 20


def test_the_groups_are_the_three_honest_ones():
    keys = [group for group, _families in REGRESSION_FAMILY_GROUPS]

    assert keys == ["parametric", "robust_semiparametric", "rank_based"]
    assert GROUP_TITLES["robust_semiparametric"] == "robust/semiparametric"
    assert set(GROUP_TITLES) == set(keys)


@pytest.mark.parametrize("family", ["rlm", "huber", "quantile"])
def test_nothing_merely_robust_is_called_nonparametric(family):
    """Labelling a linear model with a robust loss 'nonparametric' would be
    wrong in a way a reviewer would catch: it is parametric in the
    coefficients, and only the error term is left unspecified."""
    assert family_group(family) == "robust_semiparametric"
    assert "nonparametric" not in family_label(family).lower()


def test_only_the_rank_family_reads_nothing_but_order():
    assert family_group("rra") == "rank_based"
    assert "order" in family_label("rra").lower()
    assert [name for group, names in REGRESSION_FAMILY_GROUPS
            if group == "rank_based" for name in names] == ["rra"]


def test_nothing_is_renamed():
    """A settings CSV written before the grouping asks for the same fit."""
    values = [value for value, _label in regression_family_choices()]

    assert set(values) == FITTABLE
    for value, label in regression_family_choices():
        assert label.startswith(f"{value} — "), label


def test_each_family_states_what_it_assumes():
    for value, label in regression_family_choices():
        assumption = REGRESSION_FAMILY_ASSUMPTIONS[value]
        assert len(assumption.split()) >= 6, f"{value}: {assumption!r}"
        assert assumption in label
        assert GROUP_TITLES[family_group(value)] in label


def test_a_family_with_no_p_value_says_so():
    """Substituting a normal-theory p-value for a fit that has none puts the
    assumption back in through the door the method was chosen to close."""
    assert NO_P_VALUE_TYPES, "nothing to check"
    for family in NO_P_VALUE_TYPES:
        assert "no p value from the fit" in family_label(family)
    assert "no p value from the fit" not in family_label("ols")


def test_the_menu_leads_with_the_default_and_then_the_groups():
    choices = regression_family_choices()
    order = [family_group(value) for value, _label in choices]

    assert choices[0][0] == "mixed", "the default is not first"
    assert order == sorted(
        order, key=["parametric", "robust_semiparametric",
                    "rank_based"].index), "the groups are interleaved"


def test_a_family_the_table_does_not_place_is_named_not_guessed():
    """The signal that the inventory grew and this table did not."""
    with pytest.raises(KeyError, match="no regression family group"):
        family_group("a family spaCR never had")


def test_the_fit_module_offers_the_same_menu():
    """One inventory. A panel importing it from `spacr.ml` and a panel
    importing it from the table cannot disagree about what is on offer."""
    from spacr.ml import regression_family_choices as from_ml

    assert from_ml() == regression_family_choices()
