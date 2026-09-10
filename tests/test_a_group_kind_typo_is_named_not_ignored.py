"""A typo in a group kind, which must be refused rather than select nothing.

``columns_in``'s own docstring says why: "A typo here would otherwise select
nothing and read as 'this table has no intensity measurements'." That is the
whole argument -- a silent empty selection is indistinguishable from a real
absence, and the user would go looking at their data instead of their spelling.

``resolve`` makes the same refusal for the same reason, and both raises name
the kinds that do exist, which is what turns the error into the documentation.
"""
from __future__ import annotations

import pytest


COLUMNS = [
    "cell_area", "cell_perimeter",
    "cell_channel_0_mean_intensity", "cell_channel_1_mean_intensity",
    "nucleus_area", "nucleus_channel_1_mean_intensity",
    "prcfo",
]


def test_the_known_kinds_select_columns():
    """The baseline: each kind resolves to something."""
    from spacr.column_groups import GROUP_KINDS, group_names

    names = group_names(COLUMNS)

    assert set(names) == set(GROUP_KINDS)
    assert any(names[kind] for kind in GROUP_KINDS)


def test_a_mistyped_kind_is_refused_by_columns_in():
    """The raise the docstring argues for, with the kinds listed.

    Returning [] would read as "this table has no such measurements", and the
    user would inspect their data rather than their spelling.
    """
    from spacr.column_groups import GROUP_KINDS, columns_in

    with pytest.raises(KeyError) as excinfo:
        columns_in(COLUMNS, "channels", "channel_0")

    message = str(excinfo.value)
    assert "channels" in message
    for kind in GROUP_KINDS:
        assert kind in message


def test_a_mistyped_kind_is_refused_by_resolve():
    """The same refusal on the other entry point, which the picker calls.

    Two functions taking the same vocabulary must agree about it; one
    accepting silently would make the picker and the reduction disagree about
    what was selected.
    """
    from spacr.column_groups import GROUP_KINDS, resolve

    with pytest.raises(KeyError) as excinfo:
        resolve(COLUMNS, {"channels": ["channel_0"]})

    for kind in GROUP_KINDS:
        assert kind in str(excinfo.value)


def test_a_known_kind_with_an_unknown_group_name_selects_nothing():
    """The contrast, and it is deliberate: the KIND is the vocabulary.

    A group NAME comes from the data -- channel 7 exists or it does not -- so
    an unknown name is an empty selection rather than an error, while an
    unknown kind is a typo in the code or the settings.
    """
    from spacr.column_groups import columns_in, resolve

    assert columns_in(COLUMNS, "channel", "channel_99") == []
    assert resolve(COLUMNS, {"channel": ["channel_99"]}) == []


def test_explicit_columns_join_whatever_the_groups_selected():
    """The docstring's promise that the two mechanisms do not fight."""
    from spacr.column_groups import resolve

    out = resolve(COLUMNS, {"channel": ["channel_1"]}, explicit=["cell_area"])

    assert "cell_area" in out
    assert any("channel_1" in c for c in out)


def test_the_result_follows_the_table_order_not_the_click_order():
    """The reproducibility rule the docstring states.

    "A UMAP whose axes depend on click order is not reproducible" -- so the
    output order comes from ``columns``, whichever way the selection was built.
    """
    from spacr.column_groups import resolve

    one = resolve(COLUMNS, {"channel": ["channel_0", "channel_1"]})
    other = resolve(COLUMNS, {"channel": ["channel_1", "channel_0"]})

    assert one == other
    assert one == [c for c in COLUMNS if c in set(one)]


def test_no_selection_at_all_selects_only_the_explicit_columns():
    """The empty mapping, which is what an untouched picker sends."""
    from spacr.column_groups import resolve

    assert resolve(COLUMNS, None) == []
    assert resolve(COLUMNS, {}, explicit=["cell_area"]) == ["cell_area"]
