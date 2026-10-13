"""Tracker lineage edges: Trackastra's division table and empty SAM2 tracks."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from spacr import timelapse as tl
from tests.test_timelapse_trackastra import (_disc, _install_stub_trackastra,
                                             _moving_stack, _no_figures)  # noqa: F401


def test_trackastra_division_links_become_parent_tracks(monkeypatch, tmp_path):
    masks = _moving_stack(n_frames=3, n_objects=2)
    tracked = np.zeros_like(masks)
    for t in range(3):
        _disc(tracked[t], cy=12, cx=10 + 3 * t, label=1)
        if t:
            _disc(tracked[t], cy=30, cx=10 + 3 * t, label=2)
    _install_stub_trackastra(monkeypatch, relabelled=tracked)
    monkeypatch.setattr(
        sys.modules["trackastra.tracking"], "graph_to_ctc",
        lambda graph, masks, outdir=None: (
            pd.DataFrame({"label": [1, 2], "parent": [0, 1]}), tracked))
    src = tmp_path / "run" / "batch"
    src.parent.mkdir(parents=True, exist_ok=True)
    tl._trackastra_track_cells(src=str(src), name="b1",
                               batch_filenames=["a", "b", "c"],
                               object_type="cell", masks=masks,
                               images=masks.astype(np.float32))
    table = pd.read_csv(tmp_path / "run" / "tracks"
                        / "trackastra_tracks_cell_b1.csv")
    assert "parent_track_id" in table.columns
    assert table.groupby("track_id")["parent_track_id"].first().to_dict()[2] == 1


def test_sam2_following_nothing_writes_an_empty_track_table(tmp_path):
    masks = np.zeros((2, 12, 12), np.int32)
    masks[0, 2:5, 2:5] = 1
    images = masks.astype(np.float32)

    def propagate(frames, seeds, model=None, device=None):
        return np.zeros_like(masks), {"objects": 0, "seconds": 0.0}

    src = tmp_path / "run" / "masks"
    src.mkdir(parents=True)
    out = tl._sam2_track_cells(str(src), "b", ["a", "b"], "cell", masks,
                               images=images, timelapse_remove_transient=True,
                               propagate=propagate)
    assert np.asarray(out).max() == 0


def test_ultrack_without_a_lineage_graph_leaves_parents_unset(monkeypatch,
                                                              tmp_path):
    from tests.test_timelapse_ultrack import (_consistent_stack,
                                              _install_stub_ultrack)
    from tests.test_timelapse_ultrack import _moving_stack as _ultrack_stack

    masks = _ultrack_stack(n_frames=3, n_objects=2)
    pkg = _install_stub_ultrack(monkeypatch, relabelled=_consistent_stack(3))
    real = pkg.to_tracks_layer
    monkeypatch.setattr(pkg, "to_tracks_layer",
                        lambda config, include_parents=True: (
                            real(config)[0], None))
    src = tmp_path / "run" / "batch"
    src.parent.mkdir(parents=True, exist_ok=True)
    tl._ultrack_track_cells(src=str(src), name="b1",
                            batch_filenames=["a", "b", "c"], object_type="cell",
                            masks=masks, images=masks.astype(np.float32))
    table = pd.read_csv(tmp_path / "run" / "tracks" / "ultrack_tracks_cell_b1.csv")
    assert "parent_track_id_source" not in table.columns


def test_a_calcium_trace_unknown_at_its_start_keeps_an_unknown_first_delta(
        tmp_path):
    from tests.test_calcium_bleaching_f539 import (LEVEL, RING, database,
                                                   measured_frame)

    raw = measured_frame()
    first = raw.time.min()
    raw.loc[(raw.object_label == 42) & (raw.time == first), RING] = np.nan
    result, _peaks, _ = tl.analyze_calcium_oscillations(
        str(database(tmp_path, raw)), measurement=LEVEL,
        bleach_correction="ratio", remove_transient=False)
    start = result[(result.object_label == 42) & (result.time == first)]
    assert start["delta_" + LEVEL].isna().all()
