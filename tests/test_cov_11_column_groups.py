"""A column the feature dictionary cannot describe is dropped, not fatal.

Grouping runs over whatever column names a database happens to hold, and the
dictionary that names them is a separate, evolving thing. If describing one
column throws, the picker must still offer every group it could build from
the rest -- an exception here would leave a user staring at an empty
reduction dialog with a working table behind it.
"""
from __future__ import annotations

import pytest

import spacr.feature_dict as feature_dict
from spacr.column_groups import classify, group_names, resolve

COLUMNS = ["cell_area", "cell_channel_1_mean_intensity", "nucleus_area"]


@pytest.fixture()
def parse_explodes_on_nucleus(monkeypatch):
    """Make the dictionary raise for one column and behave for the others."""
    real = feature_dict.parse_column

    def fussy(name, *args, **kwargs):
        if str(name).startswith("nucleus"):
            raise RuntimeError("the dictionary could not describe this column")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(feature_dict, "parse_column", fussy)
    return fussy


def test_a_column_that_cannot_be_described_is_left_out_of_every_group(
        parse_explodes_on_nucleus):
    """The undescribable column disappears from the groups; the rest survive."""
    grouped = classify(COLUMNS)

    assert grouped["object"]["cell"] == ["cell_area",
                                         "cell_channel_1_mean_intensity"]
    assert "nucleus" not in grouped["object"]
    assert "nucleus_area" not in grouped["family"].get("morphology", [])


def test_the_picker_still_offers_the_groups_it_could_build(
        parse_explodes_on_nucleus):
    """One bad column must not empty the group list the dialog shows."""
    names = group_names(COLUMNS)

    assert names["object"] == ["cell"]
    assert names["channel"] == ["channel_1"]
    assert names["family"]


def test_selecting_a_group_never_returns_the_undescribable_column(
        parse_explodes_on_nucleus):
    """What a selection resolves to is only columns that were classified."""
    chosen = resolve(COLUMNS, {"object": ["cell"]})

    assert chosen == ["cell_area", "cell_channel_1_mean_intensity"]


def test_asking_for_one_group_by_name_returns_its_columns():
    """`columns_in` is the single-group form of `resolve`, and agrees with it.

    An unknown group name is an empty list rather than an error, because a
    saved selection naming a group this table does not have must open, not
    crash.
    """
    from spacr.column_groups import columns_in

    assert columns_in(COLUMNS, "object", "cell") == [
        "cell_area", "cell_channel_1_mean_intensity"]
    assert columns_in(COLUMNS, "channel", "channel_9") == []
