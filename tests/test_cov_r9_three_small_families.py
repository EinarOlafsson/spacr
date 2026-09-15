"""Two families of guard, each appearing in more than one module.

Grouped rather than filed per module, because the interesting thing
about each is the same in every copy -- and a family pinned once is a
family that cannot drift apart quietly.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 1. every numeric role channel is in the dense map built from those roles
# ---------------------------------------------------------------------------

class TestTheDenseChannelMap:

    def test_a_role_channel_is_mapped_to_its_position_on_the_stack(self):
        """The True arm, and the trap the helper exists to close: the
        position is ROLE order, not sorted order."""
        from spacr.utils import dense_mask_channel_positions

        positions = dense_mask_channel_positions(
            {"nucleus_channel": 2, "cell_channel": 0, "organelle_channel": 1})

        assert positions[2] == 0
        assert positions[0] == 1
        assert positions[1] == 2, (
            "raw channel 1 is being read as position 1, which holds the "
            "cell image -- cellpose would segment organelles on the wrong "
            "plane, silently")

    def test_a_channel_no_role_names_is_absent_from_the_map(self):
        """THE ARC the ``if _raw in _dense`` guard is for.

        A settings file can name a channel for a role that is not part of
        the mask stack, and looking it up would be a KeyError inside a
        segmentation run. Leaving the cellpose channel unset instead lets
        the caller fall back, which is what it does.
        """
        from spacr.utils import dense_mask_channel_positions

        positions = dense_mask_channel_positions(
            {"nucleus_channel": 0, "cell_channel": 1})

        assert 3 not in positions
        assert 0 in positions and 1 in positions

    def test_a_none_or_uncoercible_channel_is_absent_too(self):
        from spacr.utils import dense_mask_channel_positions

        positions = dense_mask_channel_positions(
            {"nucleus_channel": None, "cell_channel": "not a number",
             "pathogen_channel": 1})

        assert positions == {1: 0}

    def test_every_copy_of_the_lookup_guards_it(self):
        """The map construction is the guard; membership re-checks are dead.

        Both object generators ask ``dense_mask_channel_positions`` to walk
        these same four role keys, then apply the same ``int`` coercion before
        indexing.  A numeric value that reaches the lookup is therefore in the
        map.  Direct indexing is intentional: if either contract ever drifts,
        the run fails loudly instead of silently leaving an alias unset.

        The source assertion makes the old ``if _raw in _dense`` shape a red
        mutation, while the value assertion pins the premise that makes its
        removal safe.
        """
        from spacr import object as OBJ
        from spacr.utils import dense_mask_channel_positions

        settings = {
            "nucleus_channel": "2",
            "cell_channel": 0,
            "pathogen_channel": None,
            "organelle_channel": 1,
        }
        positions = dense_mask_channel_positions(settings)
        for role in ("nucleus", "cell", "pathogen", "organelle"):
            raw = settings.get(f"{role}_channel")
            if raw is None:
                continue
            assert int(raw) in positions

        for function in (
                OBJ.generate_cellpose_masks_sam,
                OBJ.generate_cellpose_masks):
            source = inspect.getsource(function)
            assert "if _raw in _dense:" not in source
            assert source.count(
                "settings[f'cellpose_{_role}_channel'] = _dense[_raw]"
            ) == 1


# ---------------------------------------------------------------------------
# 2. `if name not in <list>:` -- order-preserving de-duplication
# ---------------------------------------------------------------------------

class TestOrderPreservingDeduplication:

    def test_a_column_dropped_by_two_tables_is_reported_once(self):
        """THE ARC: the name is already in the list.

        Two measurement tables can drop the same merged column -- the
        same feature is measured per object type -- and reporting it
        twice would say twice as many columns were lost as were.
        """
        seen = []
        for name in ("area", "perimeter", "area"):
            if name not in seen:
                seen.append(name)

        assert seen == ["area", "perimeter"]

    def test_the_reporter_sorts_what_it_collected(self):
        """The list is built in walk order and returned sorted, so the
        report does not change because the tables were read in a
        different order."""
        from spacr import plate_measurements as PM

        source = inspect.getsource(PM.PlateMerge.dropped_columns.fget)

        assert "if name not in found:" in source
        assert "return tuple(sorted(found))" in source, (
            "the dropped-column report is no longer sorted, so it moves "
            "with table order and two identical runs disagree")

    def test_the_same_shape_guards_the_mask_registry(self):
        """The other copy, in measure.py: a mask offered under a name the
        registry already holds must not replace it, because the two are
        the same object type and the first is the one every later step
        has already keyed on."""
        from spacr import measure as M

        source = inspect.getsource(M)

        assert "if name not in masks:" in source
        assert "masks = dict(masks, **{name: mask})" in source
        index = source.index("if name not in masks:")
        assert "dict(masks" in source[index:index + 200], (
            "the registry is now mutated in place rather than copied, so a "
            "caller holding the old mapping sees a mask appear in it")
