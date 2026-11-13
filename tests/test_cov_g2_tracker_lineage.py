"""Tracker lineage edges: Trackastra's division table and empty SAM2 tracks."""
from __future__ import annotations

import inspect
import re
import sys

import numpy as np
import pandas as pd
import pytest

from spacr import timelapse as tl
from tests.test_timelapse_trackastra import (
    _disc,
    _install_stub_trackastra,
    _moving_stack,  # noqa: F401
    )


def _scn_trackastra_division_links_become_parent_tracks(monkeypatch, tmp_path):
    _obs = []
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
    _obs.append(_seen(("parent_track_id" in table.columns), ("parent_track_id" in table.columns)))
    assert table.groupby("track_id")["parent_track_id"].first().to_dict()[2] == 1
    _obs.append(_seen((table.groupby("track_id")["parent_track_id"].first().to_dict()[2]), (table.groupby("track_id")["parent_track_id"].first().to_dict()[2] == 1)))
    return _obs


def _scn_sam2_following_nothing_writes_an_empty_track_table(tmp_path):
    _obs = []
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
    _obs.append(_seen((np.asarray(out).max()), (np.asarray(out).max() == 0)))
    return _obs


def _scn_ultrack_without_a_lineage_graph_leaves_parents_unset(monkeypatch,
                                                              tmp_path):
    _obs = []
    from tests.test_timelapse_ultrack import _consistent_stack, _install_stub_ultrack
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
    _obs.append(_seen(("parent_track_id_source" not in table.columns), ("parent_track_id_source" not in table.columns)))
    return _obs


def _scn_a_calcium_trace_unknown_at_its_start_keeps_an_unknown_first_delta(
        tmp_path):
    _obs = []
    from tests.test_calcium_bleaching_f539 import LEVEL, RING, database, measured_frame

    raw = measured_frame()
    first = raw.time.min()
    raw.loc[(raw.object_label == 42) & (raw.time == first), RING] = np.nan
    result, _peaks, _ = tl.analyze_calcium_oscillations(
        str(database(tmp_path, raw)), measurement=LEVEL,
        bleach_correction="ratio", remove_transient=False)
    start = result[(result.object_label == 42) & (result.time == first)]
    assert start["delta_" + LEVEL].isna().all()
    _obs.append(_seen((start["delta_" + LEVEL].isna().all()), (start["delta_" + LEVEL].isna().all())))
    return _obs


def _plain(value):
    """Return a plain comparable copy of a value, or raise ValueError."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return re.sub(r"/tmp/pytest-of-[^/]+/pytest-\d+/[^/\s]+", "<tmp>", value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return None if value != value else round(value, 6)
    if isinstance(value, (list, tuple)) and len(value) <= 60:
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)) and len(value) <= 60:
        return sorted((_plain(v) for v in value), key=repr)
    if isinstance(value, dict) and len(value) <= 60:
        return {str(k): _plain(v) for k, v in value.items()}
    raise ValueError(type(value))


def _seen(value, held):
    """Observed value of one check: the value itself when plain, else whether the check held."""
    try:
        return _plain(value)
    except ValueError:
        return bool(held)


def _run_scenario(request, scenario, kwargs):
    """Run one scenario function, resolving its fixtures by name."""
    wanted = [n for n in inspect.signature(scenario).parameters if n not in kwargs]
    fixtures = {n: request.getfixturevalue(n) for n in wanted}
    return scenario(**fixtures, **kwargs)


@pytest.mark.parametrize("scenario, kwargs, expected", [
    pytest.param(_scn_trackastra_division_links_become_parent_tracks, {}, [True, 1], id='trackastra-division-links-become-parent-tracks'),
    pytest.param(_scn_sam2_following_nothing_writes_an_empty_track_table, {}, [0], id='sam2-following-nothing-writes-an-empty-track-table'),
    pytest.param(_scn_ultrack_without_a_lineage_graph_leaves_parents_unset, {}, [True], id='ultrack-without-a-lineage-graph-leaves-parents-unset'),
    pytest.param(_scn_a_calcium_trace_unknown_at_its_start_keeps_an_unknown_first_delta, {}, [True], id='a-calcium-trace-unknown-at-its-start-keeps-an-unknown-first-delta'),
])
def test_tracker_lineage_tables_carry_division_parents_and_empty_tracks_stay_empty(request, scenario, kwargs, expected):
    observed = _run_scenario(request, scenario, kwargs)
    assert observed == expected
