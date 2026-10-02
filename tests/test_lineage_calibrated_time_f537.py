"""Physical lineage times retain frame endpoints and reject conflicting calibration."""
import numpy as np
import pandas as pd
import pytest

from spacr._lineage_trees import (
    _lineage_calibrate_time,
    _lineage_newick,
    _lineage_segments,
    _lineage_tree_figure,
    _lineage_trees_from_tracks,
    _run_lineage_step,
)
from tests.test_lineage_trees_from_tracks import _field


def _complete(result):
    return result['segments'].dropna(subset=['generation_time']).sort_values('start_frame')


def test_interval_roundtrip_preserves_frames_newick_and_statistics(tmp_path):
    frame = _field()
    source = tmp_path / 'tracks.csv'
    frame.to_csv(source, index=False)
    original = source.read_bytes()
    old = _lineage_segments(frame)
    result = _lineage_trees_from_tracks(source, plot=False, frame_interval_s=900,
                                       color_by='generation_time_hours')
    pd.testing.assert_frame_equal(result['segments'][old.columns], old)
    assert _complete(result)['generation_time_hours'].tolist() == [2, 2.5]
    assert result['segments'].loc[result['segments'].generation_time.isna(),
                                  'generation_time_hours'].isna().all()
    assert open(result['paths']['newick']).read() == _lineage_newick(old)
    saved = pd.read_csv(result['paths']['segments'])
    np.testing.assert_allclose(saved['generation_time_hours'],
                               result['segments']['generation_time_hours'], equal_nan=True)
    overall = pd.read_csv(result['paths']['statistics']).iloc[-1]
    assert overall['generation_time_mean'] == 9
    assert overall['generation_time_hours_mean'] == 2.25
    assert overall['generation_time_hours_median'] == 2.25
    assert overall['generation_time_hours_sd'] == pytest.approx(np.std([2, 2.5], ddof=1))
    assert overall['time_calibration_source'] == 'frame_interval_s'
    assert overall['generation_time_unit'] == 'frames'
    assert overall['calibrated_time_unit'] == 'hours'
    assert source.read_bytes() == original


def test_irregular_timestamps_use_exact_birth_to_daughter_birth(tmp_path):
    tracks = _field()
    tracks['time_s'] = 100000 + tracks['frame'] ** 2 * 30
    path = tmp_path / 'tracks.csv'
    tracks.to_csv(path, index=False)
    result = _lineage_trees_from_tracks(path, plot=False)
    complete = _complete(result)
    np.testing.assert_allclose(complete['generation_time_hours'],
                               [(13**2 - 5**2) / 120, (15**2 - 5**2) / 120])
    assert result['calibration']['time_calibration_source'] == 'time_s'
    assert pd.isna(result['calibration']['calibration_frame_interval_s'])


def test_sparse_observations_keep_frame_step_and_allow_timestamp_offset(tmp_path):
    tracks = _field()
    tracks['parent_track_id'] = tracks['track_id'].map({2: 1, 3: 1, 4: 2, 5: 3, 6: 3}).fillna(0)
    tracks['frame'] *= 3
    tracks['time_s'] = -2000 + tracks['frame'] * 900
    path = tmp_path / 'tracks.csv'
    tracks.to_csv(path, index=False)
    result = _lineage_trees_from_tracks(path, plot=False, frame_interval_s=900)
    # Native links make sparse observations independent of inference adjacency.
    complete = _complete(result)
    assert complete['generation_time'].tolist() == [24, 30]
    assert complete['generation_time_hours'].tolist() == [6, 7.5]
    np.testing.assert_allclose(complete['generation_time_hours'],
                               complete['generation_time'] / 4)
    assert result['calibration']['time_calibration_source'] == 'time_s'


@pytest.mark.parametrize('interval', [True, False, 0, -1, float('nan'), float('inf'), 'bad', []])
def test_invalid_interval_preserves_existing_outputs(tmp_path, interval):
    path = tmp_path / 'tracks.csv'
    _field().to_csv(path, index=False)
    output = tmp_path / 'lineage'
    output.mkdir()
    sentinel = output / 'tracks_segments.csv'
    sentinel.write_bytes(b'previous valid result')
    with pytest.raises(ValueError, match='frame_interval_s'):
        _lineage_trees_from_tracks(path, output, plot=False, frame_interval_s=interval)
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b'previous valid result'


@pytest.mark.parametrize('problem', ['conflicting_frame', 'nan', 'backward', 'constant', 'bool', 'conflicting_interval', 'tiny_conflicting_interval'])
def test_bad_timestamp_calibration_fails_before_creating_outputs(tmp_path, problem):
    tracks = _field()
    tracks['time_s'] = tracks['frame'] * 900.0
    interval = None
    if problem == 'conflicting_frame':
        tracks.loc[0, 'time_s'] = 1
    elif problem == 'nan':
        tracks.loc[0, 'time_s'] = np.nan
    elif problem == 'backward':
        tracks.loc[tracks.frame == 2, 'time_s'] = -1
    elif problem == 'constant':
        tracks['time_s'] = 0
    elif problem == 'bool':
        tracks['time_s'] = True
    elif problem == 'tiny_conflicting_interval':
        tracks['time_s'] = tracks['frame'] * 1e-12
        interval = 1e-100
    else:
        interval = 60
    path = tmp_path / 'tracks.csv'
    tracks.to_csv(path, index=False)
    output = tmp_path / 'lineage'
    with pytest.raises(ValueError, match='lineage'):
        _lineage_trees_from_tracks(path, output, plot=False, frame_interval_s=interval)
    assert not output.exists()


def test_missing_calibration_does_not_use_motility_default_or_movie_rate(tmp_path):
    src = tmp_path / 'merged'
    src.mkdir()
    directory = tmp_path / 'tracks'
    directory.mkdir()
    _field().to_csv(directory / 'trackpy_tracks_cell_p.csv', index=False)
    result = _run_lineage_step(str(src), 'p', 'cell', 'iou', {
        'seconds_per_frame': 60, 'fps': 2, 'save': False})
    assert result['segments']['generation_time_hours'].isna().all()
    assert result['statistics']['generation_time_hours_mean'].isna().all()
    assert result['calibration']['time_calibration_source'] == 'missing'
    calibrated = _run_lineage_step(str(src), 'p', 'cell', 'iou', {
        'frame_interval_s': 900, 'seconds_per_frame': 60, 'save': False})
    assert _complete(calibrated)['generation_time_hours'].tolist() == [2, 2.5]


def test_empty_tracks_keep_explicit_units_and_missing_hours():
    tracks = pd.DataFrame(columns=['frame', 'track_id', 'x', 'y'])
    result, provenance = _lineage_calibrate_time(_lineage_segments(tracks), tracks)
    assert result.empty
    assert 'generation_time_hours' in result
    assert provenance['calibrated_time_unit'] == 'hours'


def test_calibrated_colour_keeps_figure_axis_explicitly_in_frames():
    frame = _field()
    segments, _ = _lineage_calibrate_time(_lineage_segments(frame), frame, 900)
    figure = _lineage_tree_figure(segments, segments['generation_time_hours'],
                                  'generation_time_hours')
    assert figure.axes[0].get_xlabel() == 'frame'
    assert figure.axes[1].get_ylabel() == 'generation_time_hours'
    assert len(figure.axes[0].lines) > len(segments)
    assert any(np.array_equal(line.get_xdata(), [5, 13]) for line in figure.axes[0].lines)


def test_large_finite_calibration_does_not_overflow_summary_variance(tmp_path):
    path = tmp_path / 'tracks.csv'
    _field().to_csv(path, index=False)
    result = _lineage_trees_from_tracks(path, plot=False, frame_interval_s=1e308)
    overall = result['statistics'].iloc[-1]
    assert np.isfinite(overall['generation_time_hours_sd'])
    assert overall['generation_time_hours_sd'] == pytest.approx(np.std([8, 10], ddof=1) * (1e308 / 3600))
