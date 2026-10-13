"""Both successful tracks and targeted corruptions exercise the independent audit."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from timelapse_evidence import reference_partitions, verify_tracks, verify_worker_passes


def test_intermediate_value_does_not_prove_that_the_final_threshold_ran():
    records = [dict(threshold=t, segmented=False, error='') for t in [.1, 1, 0, .1]]
    assert verify_worker_passes(records)['final_threshold'] == .1
    # Exactly the observed native failure: final widget=.1, actual last pass=0.
    records[-1]['threshold'] = 0
    with pytest.raises(ValueError, match='restored'):
        verify_worker_passes(records)


@pytest.mark.parametrize('field,value', [('segmented', True), ('segmented', None), ('error', 'failed')])
def test_inference_or_failure_is_never_called_cached_relinking(field, value):
    records = [dict(threshold=t, segmented=False, error='') for t in [.1, 1, .1]]
    verify_worker_passes(records)
    records[1][field] = value
    with pytest.raises(ValueError, match='no segmentation'):
        verify_worker_passes(records)


@pytest.mark.parametrize('thresholds', [[], [.1, 1], [0, 1, .1], [.1, .2, .1]])
def test_every_named_worker_pass_must_be_present(thresholds):
    with pytest.raises(ValueError, match='restored'):
        verify_worker_passes([dict(threshold=t, segmented=False, error='') for t in thresholds])


def example(threshold=.1):
    masks = np.zeros((3, 12, 14), dtype=np.uint16)
    for frame in range(3):
        masks[frame, 2:5, 2 + frame:5 + frame] = 10
        masks[frame, 8:10, 10:12] = 20
    rows = []
    for frame in range(3):
        for label in (10, 20):
            y, x = np.nonzero(masks[frame] == label)
            track = 8 if label == 20 else (1 if threshold <= .5 else frame + 2)
            rows.append(dict(frame=frame, original_label=label, track_id=track,
                             x=float(x.mean()), y=float(y.mean())))
    return masks, rows


def test_stable_and_fragmented_tracks_have_distinct_independent_counts():
    masks, rows = example()
    stable = verify_tracks(masks, rows, .1)
    assert stable['objects_checked'] == 6
    assert stable['independent_statistics']['n_tracks'] == 2
    assert stable['independent_statistics']['mean_length'] == 3
    assert stable['independent_statistics']['max_step'] == 1
    masks, rows = example(.6)
    split = verify_tracks(masks, rows, .6)
    assert split['independent_statistics']['n_tracks'] == 4
    assert split['independent_statistics']['n_short'] == 3
    assert split['independent_statistics']['starts_after_first'] == 2
    assert split['independent_statistics']['ends_before_last'] == 2


def test_exact_overlap_threshold_is_inclusive_and_the_next_float_is_not():
    masks, rows = example(.5)
    assert verify_tracks(masks, rows, .5)['independent_statistics']['n_tracks'] == 2
    threshold = float(np.nextafter(.5, 1))
    masks, rows = example(threshold)
    assert verify_tracks(masks, rows, threshold)['independent_statistics']['n_tracks'] == 4


@pytest.mark.parametrize('change,message', [
    ('missing', 'omits'), ('duplicate', 'twice'), ('phantom', 'absent'),
    ('centroid', 'centroid'), ('nan', 'centroid'), ('fractional', 'integers'),
    ('zero', 'invalid'), ('wrong_link', 'continuations'), ('same_track', 'twice'),
])
def test_targeted_track_corruptions_are_rejected(change, message):
    masks, rows = example()
    verify_tracks(masks, rows, .1)
    if change == 'missing': rows.pop()
    if change == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    if change == 'phantom': rows[0]['original_label'] = 99
    if change == 'centroid': rows[0]['x'] += .25
    if change == 'nan': rows[0]['x'] = float('nan')
    if change == 'fractional': rows[0]['frame'] = .5
    if change == 'zero': rows[0]['track_id'] = 0
    if change == 'wrong_link': rows[2]['track_id'] = 77
    if change == 'same_track': rows[1]['track_id'] = rows[0]['track_id']
    with pytest.raises(ValueError, match=message):
        verify_tracks(masks, rows, .1)


@pytest.mark.parametrize('key', ['n_tracks', 'mean_length', 'median_length', 'n_short',
    'starts_after_first', 'ends_before_last', 'max_step', 'objects_per_frame'])
def test_each_displayed_indicator_is_compared_with_the_pixel_reference(key):
    masks, rows = example()
    stats = verify_tracks(masks, rows, .1)['independent_statistics']
    verify_tracks(masks, rows, .1, stats=stats)
    stats[key] += 1
    with pytest.raises(ValueError, match='indicator differs: ' + key):
        verify_tracks(masks, rows, .1, stats=stats)


def test_both_kinds_of_ambiguous_overlap_refuse_a_false_independent_approval():
    masks = np.zeros((2, 6, 8), dtype=np.uint16)
    masks[0, 1:3, 1:7] = 1
    masks[1, 1:3, 1:4] = 2
    masks[1, 1:3, 4:7] = 3
    with pytest.raises(ValueError, match='split'):
        reference_partitions(masks, .1)
    with pytest.raises(ValueError, match='Competing'):
        reference_partitions(masks[::-1], .1)


@pytest.mark.parametrize('threshold', [0, -1, 1.1, float('nan')])
def test_invalid_threshold_is_not_silently_clamped(threshold):
    masks, _ = example()
    with pytest.raises(ValueError, match='threshold'):
        reference_partitions(masks, threshold)


def test_masks_must_be_an_actual_nonnegative_integer_movie():
    masks, _ = example()
    reference_partitions(masks, .1)
    for invalid in [masks[0], masks.astype(float), -masks.astype(int), masks[:1]]:
        with pytest.raises(ValueError, match='integer label frames'):
            reference_partitions(invalid, .1)
    masks[1] = 0
    with pytest.raises(ValueError, match='every frame'):
        reference_partitions(masks, .1)
