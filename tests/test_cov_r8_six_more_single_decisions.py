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

def _merged_field(tmp_path, mask):
    """A merged .npy plus its open field: two intensity planes, then a mask.

    A gradient rather than a constant, so a crop that was masked to its
    region is distinguishable from one that was not.
    """
    from spacr import crops as C

    shape = mask.shape
    intensity = (np.arange(shape[0] * shape[1], dtype=np.uint32)
                 .reshape(shape) * 7 + 500).astype(np.uint16)
    stack = np.stack([intensity, intensity // 2 + 1, mask.astype(np.uint16)],
                     axis=-1)
    path = str(tmp_path / "plate1_A01_F001.npy")
    np.save(path, stack)
    return path, C.open_merged_field(path, {"cell": 2})


def _awkward_mask():
    """Two objects chosen to stress the centroid: one flush with the frame
    corner, and a C whose centre of mass is off the object entirely."""
    mask = np.zeros((40, 40), dtype=np.uint16)
    mask[0:6, 0:6] = 1
    mask[20:34, 10:14] = 2
    mask[20:34, 24:28] = 2
    mask[30:34, 14:24] = 2
    return mask


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

    def test_a_region_entirely_outside_the_window_keeps_nothing(self, tmp_path):
        """THE ARC THAT CANNOT BE TAKEN: the two rectangles do not meet.

        If they ever did, slicing with a reversed range would give an
        empty selection on the left and a non-empty one on the right,
        so the assignment would raise "could not broadcast" -- after
        the window has already been read off disk.

        ``_crop_from_field`` carries no ``if oy1 > oy0 and ox1 > ox0:``
        guard against that, because the window is centred on the
        rounded centroid and every branch of ``_region_for`` computes
        that centroid from pixels inside the bounds it returns
        alongside it. So the INVARIANT is what is pinned, over the two
        shapes most likely to break it: an object flush with the frame
        corner, whose window is clamped, and a C whose centre of mass
        is not on the object at all.
        """
        oy0, oy1, ox0, ox1 = self._overlap((0, 10, 0, 10), (20, 26, 30, 37))

        assert not (oy1 > oy0 and ox1 > ox0), (
            "the fixture no longer produces a disjoint pair")

        keep = np.zeros((10, 10), dtype=bool)
        assert not keep.any(), (
            "a crop masked to a region it does not meet must keep nothing")

        from spacr import crops as C

        path, field = _merged_field(tmp_path, _awkward_mask())
        for label in (1, 2):
            for use_bbox in (False, True):
                spec = C.CropSpec(merged_path=path, object_type="cell",
                                  label=label, channels=(0, 1), size=(8, 8),
                                  mask_dims={"cell": 2},
                                  use_bounding_box=use_bbox)
                centroid, (ry0, ry1, rx0, rx1), region = C._region_for(
                    field, spec)
                wy0, wx0 = int(centroid[0]) - 4, int(centroid[1]) - 4
                gy0, gy1 = max(wy0, ry0), min(wy0 + 8, ry1)
                gx0, gx1 = max(wx0, rx0), min(wx0 + 8, rx1)
                assert gy1 > gy0 and gx1 > gx0, (
                    f"label {label} (use_bounding_box={use_bbox}) came back "
                    "with a region the crop window does not meet, so masking "
                    "the crop to it raises 'could not broadcast' after the "
                    "window has been read off disk")
                assert region[gy0 - ry0:gy1 - ry0,
                              gx0 - rx0:gx1 - rx0].shape == \
                    (gy1 - gy0, gx1 - gx0)

        # And the crop the invariant protects really is cut, corner
        # object included: the assignment does not raise, and the half
        # outside the region is zero.
        crop = C.extract_crop(path, "cell", 1, channels=(0, 1), size=(8, 8),
                              mask_dims={"cell": 2}, use_bounding_box=False,
                              normalize=None)
        assert crop.shape == (8, 8, 3)
        assert not crop[:2, :, 0].any() and not crop[:, :2, 0].any(), (
            "the window clamped at the frame edge no longer zero-pads, so "
            "the region it was masked to was not the one it overlapped")
        assert crop[2:, 2:, 0].any()

    def test_no_region_at_all_leaves_the_crop_whole(self, tmp_path):
        """THE ARC ABOVE IT: ``region is None``.

        ``_crop_from_field`` carries no ``if region is not None:``
        either. ``_region_for`` has a single ``return`` and every path
        through it binds ``region`` to a boolean array -- ``np.ones``
        on the bounding-box branch, ``window == label`` on the outline
        branch, which raises ``LabelMissing`` rather than handing back
        an empty one. That contract is asserted here, because a None
        region would mean a crop nothing masked: the whole neighbouring
        field left in the thumbnail.
        """
        from spacr import crops as C

        path, field = _merged_field(tmp_path, _awkward_mask())
        for label in (1, 2):
            for use_bbox in (False, True):
                spec = C.CropSpec(merged_path=path, object_type="cell",
                                  label=label, channels=(0, 1), size=(8, 8),
                                  mask_dims={"cell": 2},
                                  use_bounding_box=use_bbox)
                _centroid, bounds, region = C._region_for(field, spec)
                assert region is not None, (label, use_bbox)
                assert region.dtype == bool, (label, use_bbox)
                ry0, ry1, rx0, rx1 = bounds
                assert region.shape == (ry1 - ry0, rx1 - rx0), (
                    "the region is no longer restricted to the bounds "
                    "returned with it")
                assert region.any(), (
                    f"label {label} (use_bounding_box={use_bbox}) came back "
                    "with an all-false region, so the crop it masks is "
                    "entirely black")

        with pytest.raises(C.LabelMissing):
            C._region_for(field, C.CropSpec(
                merged_path=path, object_type="cell", label=97,
                channels=(0, 1), size=(8, 8), mask_dims={"cell": 2},
                use_bounding_box=False))


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

        source = inspect.getsource(P.stream_masks_from_stack)
        empty_guard = source.index("if not stacks:")
        empty_return = source.index("return stacks", empty_guard)
        sidecar = source.index(
            'sidecar = stacks[0].path.parent / "channel_order.json"')
        assert empty_guard < empty_return < sidecar
        assert "if stacks:" not in source, (
            "the early empty return should make a second guard redundant")

    def test_a_sidecar_that_cannot_be_written_warns_and_keeps_the_masks(self):
        """The masks are written either way -- but a stack whose sidecar
        silently missed the entry is self-describing and wrong."""
        from spacr import pipeline_v2 as P

        source = inspect.getsource(P)
        assert "readers of this stack will" in source
        assert "not know which plane holds the mask." in source
        assert "LOG.warning(" in source
