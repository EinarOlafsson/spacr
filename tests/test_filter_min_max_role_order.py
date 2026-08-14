"""`filter_min_max` was applied to whichever object came next in the stack.

The setting is documented in ROLE order -- ``[cell, nucleus, pathogen]`` --
while ``mask_dims`` is the compacted list of the planes that actually exist.
``_filter_objects_in_plot`` zipped them by position:

    for i, mask_dim in enumerate(mask_dims):
        min_max = filter_min_max[i]

With ``cell_mask_dim=4``, ``nucleus_mask_dim=None``, ``pathogen_mask_dim=6``
the compacted list is ``[4, 6]``: i=0 gives the cell its own range and i=1
gives the PATHOGEN the nucleus's. So on any run with a disabled object, one
object type is size-filtered by another's limits -- objects dropped from the
figure the settings never asked to drop, and objects kept that they did.

`_remove_outside_objects` is patched out here. It removes objects with no
parent, which on a synthetic stack is all of them; the behaviour under test
is which RANGE reaches which plane, and that is what these read.
"""

import numpy as np
import pytest

import spacr.plot as plot_module
from spacr.plot import _filter_objects_in_plot


@pytest.fixture(autouse=True)
def keep_every_object(monkeypatch):
    monkeypatch.setattr("spacr.utils._remove_outside_objects",
                        lambda stack, *a, **k: stack)
    monkeypatch.setattr("spacr.utils._remove_multiobject_cells",
                        lambda stack, *a, **k: stack)


def stack_with(cell_dim=4, pathogen_dim=6, planes=7):
    """Two cells and two pathogens of known, different areas."""
    stack = np.zeros((60, 60, planes), dtype=np.uint16)
    stack[2:7, 2:7, cell_dim] = 1            # area 25
    stack[10:30, 10:30, cell_dim] = 2        # area 400
    stack[40:44, 40:44, pathogen_dim] = 1    # area 16
    stack[10:25, 40:55, pathogen_dim] = 2    # area 225
    return stack


def labels_on(stack, dim):
    return sorted(int(v) for v in np.unique(stack[:, :, dim]) if v)


def test_the_pathogen_keeps_its_own_range_when_the_nucleus_is_disabled():
    """The regression, with a range that tells the two apart.

    Pathogen range keeps anything over 10, so both survive. The NUCLEUS
    range keeps only what is over 300, which would drop both -- so if the
    pathogen plane comes back empty, it was filtered by the nucleus's limits.
    """
    out = _filter_objects_in_plot(
        stack_with(), cell_mask_dim=4, nucleus_mask_dim=None,
        pathogen_mask_dim=6, mask_dims=[4, 6],
        filter_min_max=[[10, 100000], [300, 100000], [10, 100000]],
        nuclei_limit=True, pathogen_limit=True)

    assert labels_on(out, 6) == [1, 2], (
        "the pathogen plane was filtered by the nucleus's range")


def test_the_cell_still_gets_the_first_entry():
    """The role that was already correct must stay correct."""
    out = _filter_objects_in_plot(
        stack_with(), cell_mask_dim=4, nucleus_mask_dim=None,
        pathogen_mask_dim=6, mask_dims=[4, 6],
        filter_min_max=[[100, 100000], [0, 100000], [0, 100000]],
        nuclei_limit=True, pathogen_limit=True)

    assert labels_on(out, 4) == [2], "the area-25 cell should fail >100"


def test_every_role_enabled_is_unchanged():
    """The case that always worked, pinned so the fix does not move it."""
    stack = np.zeros((60, 60, 7), dtype=np.uint16)
    stack[2:7, 2:7, 4] = 1
    stack[10:30, 10:30, 5] = 1
    stack[40:44, 40:44, 6] = 1

    out = _filter_objects_in_plot(
        stack, cell_mask_dim=4, nucleus_mask_dim=5, pathogen_mask_dim=6,
        mask_dims=[4, 5, 6],
        filter_min_max=[[0, 100000], [0, 100000], [0, 100000]],
        nuclei_limit=True, pathogen_limit=True)

    assert labels_on(out, 4) == [1]
    assert labels_on(out, 5) == [1]
    assert labels_on(out, 6) == [1]


def test_a_plane_that_is_not_a_named_role_is_left_unfiltered():
    """Unfiltered beats borrowing a neighbour's range."""
    stack = stack_with()
    stack[5:9, 50:54, 3] = 1        # area 16 on an unnamed plane

    out = _filter_objects_in_plot(
        stack, cell_mask_dim=4, nucleus_mask_dim=None, pathogen_mask_dim=6,
        mask_dims=[3, 4, 6],
        filter_min_max=[[100, 100000], [100, 100000], [100, 100000]],
        nuclei_limit=True, pathogen_limit=True)

    assert labels_on(out, 3) == [1], (
        "an unnamed plane was filtered by some role's range")


def test_no_filter_at_all_keeps_everything():
    out = _filter_objects_in_plot(
        stack_with(), cell_mask_dim=4, nucleus_mask_dim=None,
        pathogen_mask_dim=6, mask_dims=[4, 6], filter_min_max=None,
        nuclei_limit=True, pathogen_limit=True)

    assert labels_on(out, 4) == [1, 2]
    assert labels_on(out, 6) == [1, 2]
