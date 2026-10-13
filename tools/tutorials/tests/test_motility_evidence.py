"""A hand-calculated pair of tracks tests positive and corrupt evidence alike."""
import copy
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motility_evidence import pixel_reference, verify_snapshot, equal_number, verify_batch


def example():
    masks = np.zeros((3, 24, 24), dtype=np.uint16)
    xy = {10: [(3, 3), (4, 3), (5, 3)], 20: [(15, 15), (16, 15), (16, 16)]}
    points = []
    for identity, coordinates in xy.items():
        for frame, (x, y) in enumerate(coordinates):
            masks[frame, y, x] = identity
            points.append(dict(frame=frame, cellID=identity, x=x, y=y, area=1, infected=False))
    tracks = [dict(cellID=k, n_frames=3, v_px_per_frame=1., velocity=1.,
                   path_length=2., net_displacement=net, straightness=net / 2,
                   velocity_unit='px/frame', infected=False, too_short=False)
              for k, net in [(10, 2.), (20, math.sqrt(2))]]
    summary = dict(n_tracks=2, n_used=2, n_short=0, min_length=3,
        unit='px/frame', calibrated=False, mean_velocity=1., mean_velocity_infected=float('nan'),
        mean_velocity_uninfected=1., mean_straightness=(1 + math.sqrt(.5)) / 2,
        n_infected=0, n_uninfected=2, n_high_straightness=1, straightness_threshold=.95,
        glitches_fixed=0, tracks_dropped=0)
    return masks, dict(points=points, tracks=tracks, summary=summary,
        knobs=dict(pathogen_plane=-1, tracked_plane=2, n_channels=2, propagate=False,
                   min_length=3, max_displacement=50, pixels_per_um=0, seconds_per_frame=0,
                   straightness_filter=False, straightness=.95),
        plot_visible=True, plot_is_null=False)


def test_actual_pixel_arithmetic_not_the_apps_metric_function():
    masks, snapshot = example()
    assert verify_snapshot(masks, snapshot)['pixel_observations'] == 6
    ref = pixel_reference(masks)
    assert ref['tracks'][10]['straightness'] == 1
    assert ref['tracks'][20]['straightness'] == math.sqrt(.5)
    assert ref['tracks'][10]['path_length'] == 2


@pytest.mark.parametrize('ppu,interval,factor,unit', [(0, 0, 1, 'px/frame'),
    (2, 0, 1, 'px/frame'), (0, 60, 1, 'px/frame'), (2, 60, .5, 'µm/min')])
def test_both_calibration_fields_are_required(ppu, interval, factor, unit):
    masks, s = example()
    s['knobs'].update(pixels_per_um=ppu, seconds_per_frame=interval)
    for row in s['tracks']:
        row.update(velocity=factor, velocity_unit=unit)
    s['summary'].update(unit=unit, calibrated=unit == 'µm/min',
                        mean_velocity=factor, mean_velocity_uninfected=factor)
    assert verify_snapshot(masks, s)['velocity_unit'] == unit


def test_length_cutoff_removes_measurements_not_cached_tracks():
    masks, s = example()
    s['knobs']['min_length'] = 4
    for row in s['tracks']:
        row['too_short'] = True
    s['summary'].update(min_length=4, n_used=0, n_short=2, n_uninfected=0,
        n_high_straightness=0, mean_velocity=float('nan'),
        mean_velocity_uninfected=float('nan'), mean_straightness=float('nan'))
    proof = verify_snapshot(masks, s)
    assert proof['retained_tracks'] == 2 and proof['used_tracks'] == 0


def test_straightness_filter_has_a_positive_retained_counterpart():
    masks, s = example()
    verify_snapshot(masks, s)
    s['knobs']['straightness_filter'] = True
    s['tracks'] = s['tracks'][1:]
    s['summary'].update(n_tracks=1, n_used=1, n_uninfected=1,
                        n_high_straightness=0, mean_straightness=math.sqrt(.5))
    assert verify_snapshot(masks, s)['retained_tracks'] == 1


@pytest.mark.parametrize('change,match', [('missing_point', 'omits'), ('duplicate_point', 'Duplicate'),
    ('centroid', 'pixel observation'), ('area', 'pixel observation'), ('infection', 'infected'),
    ('missing_track', 'inventory'), ('duplicate_track', 'Duplicate'), ('track_velocity', 'converted'),
    ('path_length', 'track path_length'), ('track_unit', 'Track units'), ('short_flag', 'length flag'),
    ('summary_count', 'summary n_used'), ('summary_unit', 'Summary calibration'),
    ('hidden_plot', 'plot is missing'), ('null_plot', 'plot is missing')])
def test_named_corruptions_follow_a_successful_positive_example(change, match):
    masks, s = example()
    verify_snapshot(masks, s)
    if change == 'missing_point': s['points'].pop()
    if change == 'duplicate_point': s['points'].append(copy.deepcopy(s['points'][0]))
    if change == 'centroid': s['points'][0]['x'] += .5
    if change == 'area': s['points'][0]['area'] += 1
    if change == 'infection': s['points'][0]['infected'] = True
    if change == 'missing_track': s['tracks'].pop()
    if change == 'duplicate_track': s['tracks'].append(copy.deepcopy(s['tracks'][0]))
    if change == 'track_velocity': s['tracks'][0]['velocity'] += 1
    if change == 'path_length': s['tracks'][0]['path_length'] += 1
    if change == 'track_unit': s['tracks'][0]['velocity_unit'] = 'µm/s'
    if change == 'short_flag': s['tracks'][0]['too_short'] = True
    if change == 'summary_count': s['summary']['n_used'] += 1
    if change == 'summary_unit': s['summary']['calibrated'] = True
    if change == 'hidden_plot': s['plot_visible'] = False
    if change == 'null_plot': s['plot_is_null'] = True
    with pytest.raises(ValueError, match=match):
        verify_snapshot(masks, s)


@pytest.mark.parametrize('key,value', [('pathogen_plane', 3), ('tracked_plane', 3),
    ('n_channels', 3), ('propagate', True)])
def test_wrong_planes_or_propagation_are_not_accepted(key, value):
    masks, s = example()
    verify_snapshot(masks, s)
    s['knobs'][key] = value
    with pytest.raises(ValueError, match='planes or propagation'):
        verify_snapshot(masks, s)


def test_reference_refuses_glitch_correction_outside_its_scope():
    masks, s = example()
    s['knobs']['max_displacement'] = .5
    with pytest.raises(ValueError, match='teleport'):
        verify_snapshot(masks, s)


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf')])
def test_invalid_calibration_is_not_silently_called_unknown(value):
    masks, s = example()
    s['knobs']['pixels_per_um'] = value
    with pytest.raises(ValueError, match='Invalid calibration'):
        verify_snapshot(masks, s)


def test_stationary_straightness_is_undefined_not_perfect():
    masks = np.ones((3, 4, 4), dtype=np.uint16)
    assert math.isnan(pixel_reference(masks)['tracks'][1]['straightness'])


@pytest.mark.parametrize('value,expected,match', [(None, 1, 'Non-numeric'),
    (1, float('nan'), 'Undefined'), (float('inf'), 1, 'Numeric')])
def test_missing_and_nonfinite_numbers_are_checked(value, expected, match):
    with pytest.raises(ValueError, match=match):
        equal_number(value, expected, 'test')


@pytest.mark.parametrize('kind', ['dimension', 'one_frame', 'float', 'negative', 'empty', 'incomplete'])
def test_only_the_documented_mask_reference_shape_is_accepted(kind):
    masks, _ = example()
    if kind == 'dimension': masks = masks[0]
    if kind == 'one_frame': masks = masks[:1]
    if kind == 'float': masks = masks.astype(float)
    if kind == 'negative': masks = masks.astype(int) - 1
    if kind == 'empty': masks[:] = 0
    if kind == 'incomplete': masks[1][masks[1] == 10] = 0
    with pytest.raises(ValueError, match='Expected|complete tracks'):
        pixel_reference(masks)


def batch_example():
    masks, snapshot = example()
    arrays = np.zeros((*masks.shape, 4), dtype=np.uint16)
    arrays[..., 2] = masks
    arrays[..., 0] = 40
    arrays[..., 1] = 80
    rows = [dict(frame=r['frame'], cellID=r['cellID'], plateID='plate1', wellID='A01',
                 fieldID='1', infected=0, **{'cell_centroid-1': r['x'], 'cell_centroid-0': r['y'],
                   'cell_mean_intensity_ch0': 40, 'cell_mean_intensity_ch1': 80})
            for r in snapshot['points']]
    wells = [dict(plateID='plate1', wellID='A01', n_tracks=2, n_infected_tracks=0,
                  n_uninfected_tracks=2, mean_velocity_all=.5, mean_velocity_uninfected=.5,
                  mean_velocity_infected=None, velocity_unit='µm/min')]
    return arrays, rows, wells


def test_saved_batch_centroids_intensities_and_well_are_independently_checked():
    arrays, rows, wells = batch_example()
    proof = verify_batch(arrays, rows, wells, pixels_per_um=2, seconds_per_frame=60)
    assert proof['centroid_coordinates'] == 12
    assert proof['raw_channel_means'] == 12
    assert proof['hypothetical_mean_velocity'] == .5


@pytest.mark.parametrize('change,match', [('shape', 'four-plane'), ('duplicate', 'Duplicate'),
    ('missing', 'omits'), ('plate', 'identity'), ('infection', 'infected'),
    ('centroid', 'saved centroid'), ('intensity', 'channel mean'), ('well_count', 'one saved'),
    ('well_speed', 'saved well'), ('well_unit', 'units differ'), ('well_null', 'null infection')])
def test_saved_data_corruption_is_rejected_after_a_positive_control(change, match):
    arrays, rows, wells = batch_example()
    verify_batch(arrays, rows, wells, pixels_per_um=2, seconds_per_frame=60)
    if change == 'shape': arrays = arrays[..., :3]
    if change == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    if change == 'missing': rows.pop()
    if change == 'plate': rows[0]['plateID'] = 'wrong'
    if change == 'infection': rows[0]['infected'] = 1
    if change == 'centroid': rows[0]['cell_centroid-1'] += 1
    if change == 'intensity': rows[0]['cell_mean_intensity_ch0'] += 1
    if change == 'well_count': wells.append(copy.deepcopy(wells[0]))
    if change == 'well_speed': wells[0]['mean_velocity_all'] += 1
    if change == 'well_unit': wells[0]['velocity_unit'] = 'px/frame'
    if change == 'well_null': wells[0]['mean_velocity_infected'] = 0
    with pytest.raises(ValueError, match=match):
        verify_batch(arrays, rows, wells, pixels_per_um=2, seconds_per_frame=60)


def test_batch_requires_a_positive_explicit_teaching_scale():
    arrays, rows, wells = batch_example()
    with pytest.raises(ValueError, match='positive hypothetical'):
        verify_batch(arrays, rows, wells, pixels_per_um=0, seconds_per_frame=60)
