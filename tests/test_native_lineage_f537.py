"""Native tracker divisions survive CSV export without inventing root links."""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from spacr import timelapse
from tests.test_timelapse_ultrack import _install_stub_ultrack

PARENTS = {2: 1, 3: 1, 4: 2, 5: 2, 8: 99}


def _movie():
    masks = np.zeros((8, 48, 48), dtype=np.uint16)
    tracks = [(1, range(2), 5, 5), (2, range(2, 5), 35, 35),
              (3, range(2, 8), 30, 35), (4, range(5, 8), 35, 25),
              (5, range(5, 8), 40, 35), (6, range(8), 10, 10),
              (7, range(3, 8), 13, 10), (8, range(3, 8), 7, 10)]
    for label, frames, x, y in tracks:
        for frame in frames:
            masks[frame, y - 1:y + 2, x - 1:x + 2] = label
    return masks


def _btrack_stub(monkeypatch, masks):
    table = timelapse._relabelled_stack_to_tracks_df(masks)
    tracks = [types.SimpleNamespace(
        ID=int(label) * 10, parent=PARENTS.get(int(label), int(label)) * 10,
        t=rows.frame.tolist(), x=rows.x.tolist(), y=rows.y.tolist(),
        z=[0] * len(rows)) for label, rows in table.groupby('track_id')]

    class Tracker:
        def __enter__(self):
            self.tracks = tracks
            return self

        def __exit__(self, *_args):
            return False

        def configure(self, _path):
            pass

        def append(self, _objects):
            pass

        def track(self, **_kwargs):
            pass

    package = types.ModuleType('btrack')
    package.BayesianTracker = Tracker
    package.datasets = types.SimpleNamespace(cell_config=lambda: 'offline fixture')
    package.utils = types.SimpleNamespace(segmentation_to_objects=lambda *_a, **_k: [1])
    constants = types.ModuleType('btrack.constants')
    constants.BayesianUpdates = types.SimpleNamespace(APPROXIMATE=1)
    monkeypatch.setitem(sys.modules, 'btrack', package)
    monkeypatch.setitem(sys.modules, 'btrack.constants', constants)


@pytest.mark.parametrize('backend', ['ultrack', 'btrack'])
@pytest.mark.parametrize('remove_transient', [False, True])
def test_native_divisions_roots_and_filtered_tracks_survive_actual_adapter_csv(
        tmp_path, monkeypatch, backend, remove_transient):
    masks = _movie()
    original = masks.copy()
    src = tmp_path / 'run' / 'merged'
    src.mkdir(parents=True)
    name = 'plate1_A01_1'
    factor = 1 if backend == 'ultrack' else 10
    if backend == 'ultrack':
        record = {}
        package = _install_stub_ultrack(monkeypatch, relabelled=masks, record=record)
        package.to_tracks_layer = lambda _config: (
            pd.DataFrame({'track_id': [1], 'x': [-999], 'y': [-999]}),
            {child: [parent] for child, parent in PARENTS.items()})
        result = timelapse._ultrack_track_cells(
            str(src), name, [], 'cell', masks,
            timelapse_remove_transient=remove_transient)
        assert not __import__('os').path.exists(record['working_dir'])
    else:
        _btrack_stub(monkeypatch, masks)
        result = timelapse._btrack_track_cells(
            str(src), name, [], 'cell', False, False, masks, 'btrack',
            remove_transient, n_jobs=1, run_optimization=False)
    frame = pd.read_csv(src.parent / 'tracks' / f'{backend}_tracks_cell_{name}.csv')
    np.testing.assert_array_equal(masks, original)
    assert set(frame.parent_track_id_source) == {backend}
    expected = original * factor
    if remove_transient:
        expected = np.where(original == 6, 6 * factor, 0)
        assert set(frame.track_id) == {6 * factor}
        assert set(frame.parent_track_id) == {0}
    else:
        assert frame.groupby('track_id').parent_track_id.first().to_dict() == {
            label * factor: PARENTS.get(label, 0) * factor if label in (2, 3, 4, 5) else 0
            for label in range(1, 9)}
        # Daughters are distant from their mother; a nearby continuing track
        # would otherwise falsely acquire the new native root and orphan.
        segments = timelapse._lineage_segments(frame, max_distance=100).set_index('track_id')
        for child, parent in PARENTS.items():
            if parent == 99:
                continue
            assert segments.loc[child * factor, 'parent_segment_id'] == segments.loc[parent * factor, 'segment_id']
        for root in (1, 6, 7, 8):
            assert segments.loc[root * factor, 'parent_segment_id'] == 0
        assert segments.loc[2 * factor, 'generation_time'] == 3
        assert set(segments.loc[[factor, 2 * factor], 'division_source']) == {'tracker'}
        assert len(timelapse._lineage_newick(segments.reset_index()).splitlines()) == 4
    np.testing.assert_array_equal(np.asarray(result), expected)


def test_filtered_native_parent_stays_orphan_instead_of_adopting_neighbour():
    table = pd.DataFrame({'track_id': [6, 6, 8], 'frame': [0, 1, 1],
                          'x': [10, 10, 11], 'y': [10, 10, 10]})
    native = timelapse._native_lineage_columns(table, {6: None, 8: [99]}, 'ultrack')
    assert native.parent_track_id.tolist() == [0, 0, 0]
    assert timelapse._lineage_segments(native).parent_segment_id.eq(0).all()
    # Without provenance, a lone new track beside a mother that goes on is
    # not a division when the tracker reports its own parents.
    legacy = timelapse._lineage_segments(native.drop(columns='parent_track_id_source'))
    assert legacy.parent_segment_id.eq(0).all()


def test_unrepresentable_native_merge_fails_before_replacing_tracks_csv(tmp_path, monkeypatch):
    masks = _movie()
    package = _install_stub_ultrack(monkeypatch, relabelled=masks)
    package.to_tracks_layer = lambda _config: (pd.DataFrame(), {2: [1, 6]})
    src = tmp_path / 'run' / 'merged'
    target = src.parent / 'tracks' / 'ultrack_tracks_cell_batch.csv'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'prior accepted tracks')
    with pytest.raises(ValueError, match='multiple parents'):
        timelapse._ultrack_track_cells(str(src), 'batch', [], 'cell', masks)
    assert target.read_bytes() == b'prior accepted tracks'
