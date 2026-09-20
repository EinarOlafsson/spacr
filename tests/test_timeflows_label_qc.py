"""The label-consistency QC of timeflows step 1, on fixtures with known answers.

Item 426 step 1 asks four questions of the training data: is a label ever
REUSED after its object leaves or dies, are divisions annotated as two new ids
or does one child inherit the parent's, what is the frame interval and is it
constant, and how far does an object move between frames. Each is a verdict in
``spacr.timeflows_qc``, and every fixture here is built so the right answer is
arithmetic rather than opinion: squares of a known size at known positions.

The two tests that matter most are the ones separating a GAP from a REUSED
LABEL. From one frame they are the same thing -- a label that is not there --
and telling them apart is the whole reason the script exists.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import timeflows_qc as qc


# ---------------------------------------------------------------------------
# fixtures: squares whose areas, centroids and displacements are exact
# ---------------------------------------------------------------------------

SIZE = 8


def _square(frame, label, top, left, size=SIZE):
    frame[top:top + size, left:left + size] = label


def clean_stack(n_frames=6):
    """Three objects, no gaps: one drifting 2 px/frame, one 3 px/frame, one still."""
    stack = np.zeros((n_frames, 48, 48), dtype=np.int32)
    for t in range(n_frames):
        _square(stack[t], 1, 2 + 2 * t, 2)
        _square(stack[t], 2, 20, 4 + 3 * t)
        _square(stack[t], 3, 34, 30)
    return stack


def reuse_stack():
    """Label 2 dies in frame 2 and its number is given to a new object in frame 3."""
    stack = np.zeros((6, 48, 48), dtype=np.int32)
    for t in range(6):
        _square(stack[t], 1, 2 + 2 * t, 2)
        if t < 2:
            _square(stack[t], 2, 20, 4)
        elif t > 2:
            _square(stack[t], 2, 38, 38)
    return stack


def flicker_stack():
    """Label 2 is missed in one frame and comes back where it was."""
    stack = np.zeros((5, 48, 48), dtype=np.int32)
    for t in range(5):
        _square(stack[t], 1, 2, 2)
        if t != 2:
            _square(stack[t], 2, 20, 20)
    return stack


def dividing_stack(inherit=False):
    """One object becomes two in frame 3, under either labelling convention."""
    stack = np.zeros((5, 48, 48), dtype=np.int32)
    for t in range(3):
        stack[t][10:26, 10:26] = 1
    for t in (3, 4):
        stack[t][10:26, 10:17] = 1 if inherit else 2
        stack[t][10:26, 19:26] = 2 if inherit else 3
    return stack


def _verdicts(table):
    return dict(zip(table['check'], table['verdict']))


def _values(table):
    return dict(zip(table['check'], table['value']))


# ---------------------------------------------------------------------------
# what the measurement layer reports
# ---------------------------------------------------------------------------

def test_label_frames_measures_area_centroid_and_diameter():
    table = qc.label_frames(clean_stack(2))
    assert len(table) == 6
    first = table[(table['frame'] == 0) & (table['label'] == 3)].iloc[0]
    assert first['area'] == float(SIZE * SIZE)
    assert first['y'] == pytest.approx(34 + (SIZE - 1) / 2)
    assert first['x'] == pytest.approx(30 + (SIZE - 1) / 2)
    assert first['diameter'] == pytest.approx(np.sqrt(4 * SIZE * SIZE / np.pi))


def test_label_frames_skips_empty_frames():
    stack = np.zeros((3, 16, 16), dtype=np.int32)
    _square(stack[1], 4, 2, 2, size=4)
    table = qc.label_frames(stack)
    assert list(table['frame']) == [1]
    assert list(table['label']) == [4]


def test_a_volume_is_refused_rather_than_measured_in_the_plane():
    volume = np.zeros((3, 2, 16, 16), dtype=np.int32)
    with pytest.raises(ValueError) as caught:
        qc.label_frames(volume)
    message = str(caught.value)
    assert '2-D label frames' in message
    assert 'image plane' in message


def test_label_spans_reports_the_missing_frames_between_first_and_last():
    spans = qc.label_spans(qc.label_frames(flicker_stack()))
    row = spans[spans['label'] == 2].iloc[0]
    assert (row['first_frame'], row['last_frame']) == (0, 4)
    assert row['n_frames'] == 4
    assert row['n_missing'] == 1
    assert row['missing_frames'] == (2,)


def test_displacement_only_counts_steps_between_consecutive_frames():
    steps = qc.displacement_table(qc.label_frames(flicker_stack()))
    label_two = steps[steps['label'] == 2]
    assert list(label_two['frame']) == [0, 3]
    assert 1 not in set(label_two['frame'])
    assert float(label_two.iloc[0]['displacement']) == pytest.approx(0.0)


def test_displacement_is_reported_in_pixels_and_in_object_diameters():
    steps = qc.displacement_table(qc.label_frames(clean_stack(3)))
    drifting = steps[steps['label'] == 2]
    assert set(np.round(drifting['displacement'], 6)) == {3.0}
    diameter = np.sqrt(4 * SIZE * SIZE / np.pi)
    assert float(drifting.iloc[0]['displacement_over_diameter']) == \
        pytest.approx(3.0 / diameter)


# ---------------------------------------------------------------------------
# a gap and a recycled id, which look the same from one frame
# ---------------------------------------------------------------------------

def test_a_label_that_comes_back_across_the_field_is_called_reuse():
    gaps = qc.recycled_labels(reuse_stack())
    assert len(gaps) == 1
    row = gaps.iloc[0]
    assert row['label'] == 2
    assert row['verdict'] == 'reuse-suspected'
    assert row['iou_across_gap'] == 0.0
    assert row['displacement_per_frame'] > row['allowed']


def test_a_label_missed_for_one_frame_is_called_a_gap():
    gaps = qc.recycled_labels(flicker_stack())
    assert len(gaps) == 1
    row = gaps.iloc[0]
    assert row['verdict'] == 'gap'
    assert row['iou_across_gap'] == pytest.approx(1.0)
    assert (row['gap_start'], row['gap_end'], row['gap_frames']) == (2, 2, 1)


def test_the_reuse_threshold_is_measured_from_the_data():
    gaps = qc.recycled_labels(reuse_stack())
    steps = qc.displacement_table(qc.label_frames(reuse_stack()))
    expected = qc.REUSE_DISPLACEMENT_FACTOR * float(
        np.percentile(steps['displacement'].to_numpy(), 95))
    assert float(gaps.iloc[0]['allowed']) == pytest.approx(expected)


def test_an_explicit_threshold_overrides_the_measured_one():
    gaps = qc.recycled_labels(reuse_stack(), max_displacement=1000.0)
    assert list(gaps['verdict']) == ['gap']


def test_the_threshold_falls_back_to_object_size_when_nothing_survives():
    stack = np.zeros((3, 32, 32), dtype=np.int32)
    _square(stack[0], 1, 0, 0)
    _square(stack[2], 1, 20, 20)
    gaps = qc.recycled_labels(stack)
    diameter = np.sqrt(4 * SIZE * SIZE / np.pi)
    assert float(gaps.iloc[0]['allowed']) == pytest.approx(diameter)
    assert gaps.iloc[0]['verdict'] == 'reuse-suspected'


# ---------------------------------------------------------------------------
# which division convention the annotation follows
# ---------------------------------------------------------------------------

def test_two_new_ids_is_read_as_children_are_new():
    events = qc.division_events(dividing_stack(inherit=False))
    assert len(events) == 1
    row = events.iloc[0]
    assert (row['frame'], row['parent_label']) == (3, 1)
    assert row['child_labels'] == (2, 3)
    assert row['convention'] == 'children_are_new'


def test_an_inherited_id_is_read_as_child_inherits_parent():
    events = qc.division_events(dividing_stack(inherit=True))
    assert events.iloc[0]['convention'] == 'child_inherits_parent'
    assert events.iloc[0]['child_labels'] == (1, 2)


def test_an_object_that_merely_moves_is_not_a_division():
    assert len(qc.division_events(clean_stack(4))) == 0


def test_a_child_owned_by_nobody_is_not_attached_to_a_parent():
    stack = np.zeros((2, 32, 32), dtype=np.int32)
    _square(stack[0], 1, 0, 0)
    _square(stack[1], 1, 0, 0)
    _square(stack[1], 2, 20, 20)
    assert len(qc.division_events(stack)) == 0


# ---------------------------------------------------------------------------
# the frame interval
# ---------------------------------------------------------------------------

def test_even_timestamps_measure_a_constant_interval():
    timing = qc.frame_interval_summary(timestamps=[0, 30, 60, 90])
    assert timing['source'] == 'measured'
    assert timing['interval'] == pytest.approx(30.0)
    assert timing['spread'] == pytest.approx(0.0)
    assert timing['verdict'] == 'ok'


def test_uneven_timestamps_are_flagged_rather_than_averaged():
    timing = qc.frame_interval_summary(timestamps=[0, 30, 90, 120])
    assert timing['interval'] == pytest.approx(30.0)
    assert timing['spread'] == pytest.approx(30.0)
    assert timing['verdict'] == 'check'


def test_a_stated_interval_is_reported_as_stated():
    timing = qc.frame_interval_summary(frame_interval=600.0)
    assert (timing['source'], timing['verdict']) == ('stated', 'check')


def test_no_timing_at_all_is_unknown_and_not_assumed():
    timing = qc.frame_interval_summary()
    assert timing['source'] == 'missing'
    assert timing['verdict'] == 'unknown'
    assert np.isnan(timing['interval'])


# ---------------------------------------------------------------------------
# the table the script produces
# ---------------------------------------------------------------------------

def test_a_clean_stack_passes_every_check_it_can_answer():
    table = qc.audit_label_consistency(
        clean_stack(), dataset='clean', timestamps=list(range(0, 180, 30)))
    verdicts = _verdicts(table)
    assert verdicts['label_reuse'] == 'ok'
    assert verdicts['frame_interval'] == 'ok'
    assert verdicts['displacement_over_diameter'] == 'ok'
    assert verdicts['division_convention'] == 'unknown'
    assert 'fail' not in set(table['verdict'])


def test_a_recycled_label_fails_the_audit():
    table = qc.audit_label_consistency(reuse_stack(), dataset='reuse')
    assert _verdicts(table)['label_reuse'] == 'fail'
    assert '1 suspected' in _values(table)['label_reuse']


def test_mixed_division_conventions_fail_the_audit():
    first = dividing_stack(inherit=False)
    second = dividing_stack(inherit=True)
    combined = np.zeros((5, 48, 96), dtype=np.int32)
    combined[:, :, :48] = first
    combined[:, :, 48:] = np.where(second > 0, second + 10, 0)
    table = qc.audit_label_consistency(combined, dataset='mixed')
    row = table[table['check'] == 'division_convention'].iloc[0]
    assert row['verdict'] == 'fail'
    assert 'children_are_new' in row['value']
    assert 'child_inherits_parent' in row['value']


def test_objects_moving_further_than_their_own_size_fail_the_audit():
    stack = np.zeros((4, 96, 96), dtype=np.int32)
    for t in range(4):
        _square(stack[t], 1, 2, 2 + 20 * t)
    table = qc.audit_label_consistency(stack, dataset='fast')
    row = table[table['check'] == 'displacement_over_diameter'].iloc[0]
    assert row['verdict'] == 'fail'
    assert 'point past what the head can see' in row['detail']


def test_objects_moving_about_half_their_size_ask_for_a_human():
    stack = np.zeros((4, 96, 96), dtype=np.int32)
    for t in range(4):
        _square(stack[t], 1, 2, 2 + 7 * t)
    row = qc.audit_label_consistency(stack)
    verdict = row[row['check'] == 'displacement_over_diameter'].iloc[0]
    assert verdict['verdict'] == 'check'


def test_gaps_that_are_not_suspected_still_ask_for_a_human():
    table = qc.audit_label_consistency(flicker_stack())
    row = table[table['check'] == 'label_reuse'].iloc[0]
    assert row['verdict'] == 'check'
    assert row['value'] == '1 gaps, 0 suspected'


def test_an_empty_stack_answers_unknown_rather_than_zero():
    table = qc.audit_label_consistency(np.zeros((3, 8, 8), dtype=np.int32))
    verdicts = _verdicts(table)
    assert verdicts['displacement'] == 'unknown'
    assert verdicts['label_reuse'] == 'ok'


def test_several_datasets_are_compared_on_their_frame_interval():
    table = qc.audit_datasets(
        {'a': clean_stack(3), 'b': clean_stack(3)},
        timestamps={'a': [0, 30, 60], 'b': [0, 600, 1200]})
    row = table[table['check'] == 'frame_interval_across_datasets'].iloc[0]
    assert row['dataset'] == 'all'
    assert row['verdict'] == 'check'
    assert 'conditioned on the interval' in row['detail']


def test_datasets_on_one_interval_pass_the_comparison():
    table = qc.audit_datasets(
        {'a': clean_stack(3), 'b': clean_stack(3)},
        frame_intervals={'a': 30.0, 'b': 30.0})
    row = table[table['check'] == 'frame_interval_across_datasets'].iloc[0]
    assert row['verdict'] == 'ok'


def test_a_dataset_with_no_timing_makes_the_comparison_unknown():
    table = qc.audit_datasets(
        {'a': clean_stack(3), 'b': clean_stack(3)},
        frame_intervals={'a': 30.0})
    row = table[table['check'] == 'frame_interval_across_datasets'].iloc[0]
    assert row['verdict'] == 'unknown'
    assert row['value'] == '1 of 2 known'


def test_one_dataset_gets_no_cross_dataset_row():
    table = qc.audit_datasets({'a': clean_stack(3)})
    assert 'frame_interval_across_datasets' not in set(table['check'])


def test_auditing_nothing_returns_the_empty_table_with_its_columns():
    table = qc.audit_datasets({})
    assert list(table.columns) == [
        'dataset', 'check', 'value', 'verdict', 'detail']
    assert len(table) == 0


# ---------------------------------------------------------------------------
# the rendering and the command line
# ---------------------------------------------------------------------------

def test_the_table_renders_with_a_header_and_a_truncated_detail():
    text = qc.format_qc_table(
        qc.audit_label_consistency(clean_stack(2)), width=20)
    lines = text.splitlines()
    assert lines[0].startswith('DATASET')
    assert 'DETAIL' in lines[0]
    assert any('…' in line for line in lines[1:])


def test_an_empty_table_renders_as_no_rows():
    assert qc.format_qc_table(qc.audit_datasets({})) == 'no rows'


def test_a_stack_is_read_from_a_npy_file(tmp_path):
    path = tmp_path / 'stack.npy'
    np.save(path, clean_stack(3))
    assert qc.load_label_stack(str(path)).shape == (3, 48, 48)


def test_a_stack_is_read_from_a_folder_of_frames(tmp_path):
    folder = tmp_path / 'frames'
    folder.mkdir()
    stack = clean_stack(3)
    for index in range(3):
        np.save(folder / f'frame_{index:03d}.npy', stack[index])
    read = qc.load_label_stack(str(folder))
    assert read.shape == (3, 48, 48)
    assert np.array_equal(read, stack)


def test_an_empty_folder_says_what_it_wanted(tmp_path):
    folder = tmp_path / 'frames'
    folder.mkdir()
    with pytest.raises(ValueError, match='no .npy, .tif or .tiff frames'):
        qc.load_label_stack(str(folder))


def test_an_unreadable_suffix_says_what_it_wanted(tmp_path):
    path = tmp_path / 'stack.png'
    path.write_bytes(b'')
    with pytest.raises(ValueError, match='not a label stack'):
        qc.load_label_stack(str(path))


def test_the_command_line_passes_clean_data_and_fails_recycled_data(
        tmp_path, capsys):
    clean = tmp_path / 'clean.npy'
    recycled = tmp_path / 'recycled.npy'
    np.save(clean, clean_stack())
    np.save(recycled, reuse_stack())
    assert qc.main([str(clean), '--frame-interval', '30']) == 0
    assert 'clean.npy' in capsys.readouterr().out
    assert qc.main([str(recycled)]) == 1
    assert 'fail' in capsys.readouterr().out


def test_the_command_line_writes_the_table_and_the_detail(tmp_path, capsys):
    path = tmp_path / 'recycled.npy'
    np.save(path, reuse_stack())
    csv = tmp_path / 'qc.csv'
    detail = tmp_path / 'detail'
    assert qc.main([str(path), '--csv', str(csv), '--detail', str(detail),
                    '--max-displacement', '1000']) == 0
    capsys.readouterr()
    assert csv.exists()
    written = sorted(item.name for item in detail.iterdir())
    assert written == ['recycled.npy_displacement.csv',
                       'recycled.npy_divisions.csv',
                       'recycled.npy_gaps.csv']


def test_a_stack_is_read_from_a_tif_file(tmp_path):
    tifffile = pytest.importorskip('tifffile')
    path = tmp_path / 'stack.tif'
    tifffile.imwrite(str(path), clean_stack(3).astype(np.uint16))
    read = qc.load_label_stack(str(path))
    assert read.shape == (3, 48, 48)
    assert set(np.unique(read)) == {0, 1, 2, 3}
