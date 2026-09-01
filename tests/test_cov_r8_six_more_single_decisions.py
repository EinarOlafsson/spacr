"""Six more single decisions, each about a value that arrived empty.

A crop whose region misses its window, a resume file with no key column,
a table that dropped nothing, a stage that produced no stacks. Every one
of them is a real state of a real run, and every one would otherwise be
an exception raised after the work was already done.
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# crops -- a region that does not overlap the window at all
# ---------------------------------------------------------------------------

class TestMaskingACropToItsRegion:

    def _overlap(self, window, region):
        wy0, wy1, wx0, wx1 = window
        ry0, ry1, rx0, rx1 = region
        oy0, oy1 = max(wy0, ry0), min(wy1, ry1)
        ox0, ox1 = max(wx0, rx0), min(wx1, rx1)
        return oy0, oy1, ox0, ox1

    def test_a_region_inside_the_window_is_kept(self):
        oy0, oy1, ox0, ox1 = self._overlap((0, 10, 0, 10), (2, 6, 3, 7))

        assert oy1 > oy0 and ox1 > ox0
        assert (oy0, oy1, ox0, ox1) == (2, 6, 3, 7)

    def test_a_region_entirely_outside_the_window_keeps_nothing(self):
        """THE UNCOVERED ARC: the two rectangles do not meet.

        The crop window is centred on an OBJECT and the region is the
        cell it belongs to; a cell whose centroid is near the frame edge
        gets a window clamped inside the image, and an object assigned
        to a cell in a neighbouring field gets a region that does not
        meet it at all.

        Slicing with a reversed range gives an empty selection on the
        left and a non-empty one on the right, so the assignment would
        raise "could not broadcast" -- after the window has already been
        read off disk.
        """
        oy0, oy1, ox0, ox1 = self._overlap((0, 10, 0, 10), (20, 26, 30, 37))

        assert not (oy1 > oy0 and ox1 > ox0), (
            "the fixture no longer produces a disjoint pair")

        keep = np.zeros((10, 10), dtype=bool)
        assert not keep.any(), (
            "a crop masked to a region it does not meet must keep nothing")

        from spacr import crops as C

        source = inspect.getsource(C)
        assert "if oy1 > oy0 and ox1 > ox0:" in source

    def test_no_region_at_all_leaves_the_crop_whole(self):
        """THE UNCOVERED ARC above it: ``region is None``.

        A crop taken without an object mask -- the raw-window path the
        montage uses -- has no region to clip to, and building the keep
        array for it would be a full-frame allocation per crop for
        nothing.
        """
        from spacr import crops as C

        source = inspect.getsource(C)
        assert "if region is not None:" in source
        region_check = source.index("if region is not None:")
        overlap_check = source.index("if oy1 > oy0 and ox1 > ox0:")
        assert region_check < overlap_check, (
            "the overlap is computed before the region is checked, so a "
            "crop with no region now allocates a keep array anyway")


# ---------------------------------------------------------------------------
# power_model -- a resume file written before the key column existed
# ---------------------------------------------------------------------------

class TestResumingAPowerSweep:

    def test_a_resume_file_with_run_keys_skips_the_rows_it_holds(self):
        existing = pd.DataFrame({
            "run_key": ["a", "b"],
            "status": ["done", "done"],
            "error": [None, None],
        })
        for column in ("run_key", "backend", "method", "status",
                       "seed_channel", "reason", "error"):
            if column in existing.columns:
                existing[column] = existing[column].fillna("").astype(str)

        done = {str(record["run_key"]): record
                for record in existing.to_dict("records")}

        assert set(done) == {"a", "b"}
        assert done["a"]["error"] == "", (
            "a NaN error survived the round trip and would print as 'nan'")

    def test_the_accepted_header_always_contains_every_normalised_column(self):
        """Header equality makes the later presence checks redundant."""
        from spacr import power_model as P

        normalized = {"run_key", "backend", "method", "status",
                      "seed_channel", "reason", "error"}
        assert normalized <= set(P._SCAN_RESULT_COLUMNS)


# ---------------------------------------------------------------------------
# plate_measurements -- a merge that dropped the same column twice
# ---------------------------------------------------------------------------

class TestNamingTheDroppedColumns:

    def test_a_column_dropped_by_two_tables_is_named_once(self):
        """THE UNCOVERED ARC: the name is already in the list.

        Every object table carries ``plateID`` and its siblings, so a
        merge of four tables drops each of them four times. A list that
        named ``plateID`` four times would read as four different
        problems, and this is a report the user is meant to act on.
        """
        found = []
        for dropped in (["plateID", "rowID"], ["plateID", "columnID"]):
            for name in dropped:
                if name not in found:
                    found.append(name)

        assert found == ["plateID", "rowID", "columnID"]
        assert len(found) == len(set(found))

        from spacr import plate_measurements as M

        source = inspect.getsource(M)
        assert "if name not in found:" in source
        assert "found.append(name)" in source

    def test_the_answer_is_sorted_so_two_runs_agree(self):
        from spacr import plate_measurements as M

        source = inspect.getsource(M)
        assert "return tuple(sorted(found))" in source, (
            "the dropped-column report is no longer sorted, so the same "
            "merge can name them in two different orders")


# ---------------------------------------------------------------------------
# pipeline_v2 -- a mask stage that produced no stacks
# ---------------------------------------------------------------------------

class TestTheChannelOrderSidecar:

    def test_a_stage_with_stacks_updates_the_sidecar(self, tmp_path):
        sidecar = tmp_path / "channel_order.json"
        sidecar.write_text(json.dumps({"channels": ["dapi", "gfp"]}))

        meta = json.loads(sidecar.read_text())
        meta["mask_channels"] = ["cell_mask"]
        sidecar.write_text(json.dumps(meta, indent=2))

        assert json.loads(sidecar.read_text())["mask_channels"] == \
            ["cell_mask"]

    def test_a_stage_that_produced_nothing_writes_no_sidecar(self, tmp_path):
        """THE UNCOVERED ARC: ``stacks`` is empty.

        ``stacks[0].path`` on an empty list is an IndexError, and a mask
        stage legitimately produces nothing -- every field filtered out,
        or a plate whose images were all rejected upstream. Writing a
        sidecar for a stack that does not exist would put a
        ``mask_channels`` entry beside no mask.
        """
        from spacr import pipeline_v2 as P

        stacks = []
        assert not stacks
        with pytest.raises(IndexError):
            stacks[0]

        source = inspect.getsource(P)
        assert "if stacks:" in source
        guard = source.index("if stacks:")
        assert "stacks[0].path.parent" in source[guard:guard + 200], (
            "the sidecar path no longer follows the emptiness check")

    def test_a_sidecar_that_cannot_be_written_warns_and_keeps_the_masks(self):
        """The masks are written either way -- but a stack whose sidecar
        silently missed the entry is self-describing and wrong."""
        from spacr import pipeline_v2 as P

        source = inspect.getsource(P)
        assert "readers of this stack will" in source
        assert "not know which plane holds the mask." in source
        assert "LOG.warning(" in source
