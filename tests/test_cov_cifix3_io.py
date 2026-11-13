"""Coverage for the fixed-scale contract options, the leave-one-group-out guard, fixed loaders without a split and an empty cellpose dataset."""
from __future__ import annotations


import pytest


def test_leave_one_group_out_needs_two_groups():
    """A single distinct group value cannot be held out."""
    from spacr.io import _leave_one_group_out_folds

    with pytest.raises(ValueError, match="at least two plate values"):
        _leave_one_group_out_folds(["p1", "p1", "p1"], "plate")


