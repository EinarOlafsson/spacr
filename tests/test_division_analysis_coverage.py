"""Edge cases of spacr.division_analysis: input checks, empty selections and
the degenerate statistics paths."""
import numpy as np
import pandas as pd
import pytest

from spacr import division_analysis as da


# --------------------------------------------------------------- measure_label_stack
def test_measure_rejects_mismatched_lengths_and_non_2d_masks():
    mask = np.zeros((4, 4), int)
    with pytest.raises(ValueError, match='images and masks'):
        da.measure_label_stack([mask, mask], [np.zeros((4, 4))])
    with pytest.raises(ValueError, match='times and masks'):
        da.measure_label_stack([mask], times=[0, 1])
    with pytest.raises(ValueError, match='mask 0 is not 2D'):
        da.measure_label_stack([np.zeros((2, 4, 4), int)])


def test_measure_without_images_leaves_intensity_empty_and_drops_small_objects():
    mask = np.zeros((20, 20), np.int32)
    mask[2:8, 2:8] = 1        # 36 px
    mask[15, 15] = 2          # 1 px, below min_area
    df = da.measure_label_stack([mask], min_area=5)
    assert df['track_id'].tolist() == [1]
    assert df['area'].iat[0] == 36
    assert np.isnan(df['mean_intensity'].iat[0])
    assert df['t'].iat[0] == 0


def test_measure_reads_channel_zero_of_a_channel_first_image():
    mask = np.zeros((10, 10), np.int32)
    mask[2:5, 2:5] = 4
    img = np.zeros((2, 10, 10))
    img[0] = 2.0
    img[0][mask == 4] = 6.0
    img[1] = 1000.0           # must be ignored
    df = da.measure_label_stack([mask], [img])
    # median of channel 0 is the background (2): object reads 6 / 2
    assert df['mean_intensity'].iat[0] == pytest.approx(3.0)


# --------------------------------------------------------------- lineage tables
def test_read_lineage_table_casts_to_int_and_reports_missing_columns(tmp_path):
    good = tmp_path / 'lin.csv'
    pd.DataFrame({'label': [1, 2], 't_start': [0.0, 5.0], 't_end': [4.0, 9.0],
                  'parent': [0, 1], 'extra': ['a', 'b']}).to_csv(good, index=False)
    lin = da.read_lineage_table(good)
    assert list(lin.columns) == list(da.LINEAGE_COLUMNS)
    assert lin['t_start'].dtype == np.int64
    assert lin['t_end'].tolist() == [4, 9]
    bad = tmp_path / 'bad.csv'
    pd.DataFrame({'label': [1], 't_start': [0]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match=r"lacks columns \['t_end', 'parent'\]"):
        da.read_lineage_table(bad)


def test_division_events_rejects_unknown_mode_and_skips_unknown_parents():
    lin = pd.DataFrame({'label': [1, 2, 3, 5], 't_start': [0, 10, 10, 3],
                        't_end': [9, 20, 20, 8], 'parent': [0, 1, 1, 99]})
    with pytest.raises(ValueError, match="unknown mode 'split'"):
        da.division_events(lin, mode='split')
    ev = da.division_events(lin)
    # parent 99 is not a track in the table, so only parent 1 divides
    assert ev['parent'].tolist() == [1]
    assert ev['daughters'].iat[0] == (2, 3)


def test_median_cycle_skips_daughters_missing_from_the_lineage():
    lineage = pd.DataFrame({'label': [1], 't_start': [0], 't_end': [9], 'parent': [0]})
    events = pd.DataFrame({'parent': [1, 7], 't_div': [9, 30], 'kind': 'fission',
                           'n_daughters': 2, 'daughters': [(7, 8), (11, 12)]})
    # track 7 is born and divides, but has no lineage row: nothing is measured
    assert np.isnan(da.median_cycle_frames(lineage, events))


# --------------------------------------------------------------- control slots / matching
def _track(tid, ts, area=100.0):
    return pd.DataFrame({'track_id': tid, 't': list(ts), 'area': area})


def test_control_slots_empty_when_no_track_is_long_enough_or_covered():
    short = _track(1, range(0, 3))                     # hi < lo
    gappy = _track(2, [0, 1, 9, 10])                   # window coverage too low
    slots = da.control_slots(pd.concat([short, gappy]), pd.DataFrame(
        columns=['parent', 't_div']), window=3, horizon=1, min_coverage=0.8)
    assert slots.empty
    assert list(slots.columns) == ['track_id', 'tp', 'base_size']


def test_control_slots_exclude_the_frames_around_a_tracks_own_division():
    meas = _track(1, range(0, 20))
    events = pd.DataFrame({'parent': [1], 't_div': [10]})
    slots = da.control_slots(meas, events, window=2, horizon=2)
    # a slot tp is barred when its [tp-2, tp+2] window holds the division at 10
    assert not slots['tp'].between(8, 12).any()
    assert slots['tp'].min() == 2 and slots['tp'].max() == 17
    assert (slots['base_size'] == 100.0).all()


def test_match_controls_reports_unmatched_events_for_each_reason():
    slots = pd.DataFrame({'track_id': [5, 6, 7], 'tp': [10, 10, 10],
                          'base_size': [100.0, 400.0, 100.0]})
    events = pd.DataFrame({
        'event_id': [0, 1, 2, 3, 4],
        'parent': [1, 1, 1, 5, 1],
        't_div': [50, 10, 10, 10, 10],
        'base_size': [100.0, np.nan, 0.0, 100.0, 1000.0],
        'daughters': [(2,), (2,), (2,), (7,), (2,)],
    })
    out = da.match_controls(events, slots, window=2, size_tol=0.25, rng=0)
    assert out['matched'].tolist() == [False, False, False, False, False]
    assert (out['track_id'] == -1).all() and out['size_ratio'].isna().all()
    # event 3 excludes its own parent (5) and daughter (7); 6 is 4x too big
    ok = da.match_controls(events.iloc[[3]].assign(base_size=400.0), slots,
                           window=2, rng=0)
    assert ok['matched'].tolist() == [True]
    assert ok['track_id'].iat[0] == 6 and ok['size_ratio'].iat[0] == pytest.approx(1.0)


def test_shuffled_alignment_marks_parents_without_slots_unmatched():
    slots = pd.DataFrame({'track_id': [3, 3], 'tp': [4, 6], 'base_size': 1.0})
    events = pd.DataFrame({'event_id': [10, 11], 'parent': [3, 9]})
    out = da.shuffled_alignment(events, slots, rng=0)
    assert out['matched'].tolist() == [True, False]
    assert out['t0'].iat[0] in (4, 6) and out['t0'].iat[1] == -1
    assert out['track_id'].tolist() == [3, -1]


# --------------------------------------------------------------- alignment and changes
def test_align_units_without_daughter_rows_and_difference_changes():
    meas = pd.DataFrame({'track_id': 1, 't': range(0, 8),
                         'area': [10.0, 10, 10, 10, 20, 20, 20, 20],
                         'solidity': [0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9]})
    units = pd.DataFrame({'unit_id': [0], 'track_id': [1], 't0': [7],
                          'daughters': [()]})
    aligned = da.align_units(meas, units, window=7, after=2,
                             columns=('area', 'solidity', 'speed'))
    assert set(aligned['role']) == {'self'}
    assert 'speed' not in aligned.columns
    ch = da.unit_changes(aligned, window=7, k=2, columns=('area', 'solidity'))
    assert ch['area'].iat[0] == pytest.approx(np.log(2.0))
    assert ch['solidity'].iat[0] == pytest.approx(0.4)


def test_unit_changes_turns_an_infinite_log_ratio_into_missing():
    aligned = pd.DataFrame({'unit_id': 0, 'role': 'self', 'rel_t': [-3, -2, -1, 0],
                            'area': [0.0, 0.0, 5.0, 5.0]})
    ch = da.unit_changes(aligned, window=3, k=2, columns=('area',))
    assert np.isnan(ch['area'].iat[0])


# --------------------------------------------------------------- effect sizes
def test_effect_size_is_nan_when_both_groups_are_constant():
    r = da.effect_size([1.0, 1.0, 1.0], [1.0, 1.0], rng=0)
    assert np.isnan(r['g']) and np.isnan(r['var_g'])
    assert r['mean_diff'] == 0.0 and r['n_control'] == 2


def test_effect_size_without_bootstrap_keeps_interval_empty():
    r = da.effect_size([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], n_boot=0)
    # pooled SD 1, mean difference 1, small-sample factor 1 - 3/15
    assert r['g'] == pytest.approx(0.8)
    assert r['var_g'] == pytest.approx(6 / 9 + r['g'] ** 2 / 12)
    assert np.isnan(r['ci_low']) and np.isnan(r['ci_high'])


def test_effect_size_interval_empty_when_every_resample_is_degenerate():
    # with one resample of two values per group, seed 4 draws a constant
    # resample in both groups: its g is not finite, so no interval is left
    r = da.effect_size([0.0, 1.0], [0.0, 2.0], n_boot=1, rng=4)
    assert r['g'] == pytest.approx(-0.2555506, rel=1e-6)
    assert np.isnan(r['ci_low']) and np.isnan(r['ci_high'])


# --------------------------------------------------------------- cell cycles and size control
def test_cell_cycles_skip_unknown_tracks_and_unmeasured_ends():
    lineage = pd.DataFrame({'label': [1, 2, 3, 4, 5, 6, 7],
                            't_start': [0, 5, 5, 10, 10, 10, 10],
                            't_end': [4, 9, 9, 14, 14, 14, 14],
                            'parent': [0, 1, 1, 2, 2, 3, 3]})
    events = da.division_events(lineage, mode='fission')
    # track 2 is measured on both ends; track 3 lacks its last frame
    meas = pd.DataFrame({'track_id': [2, 2, 3], 't': [5, 9, 5],
                         'area': [50.0, 90.0, 40.0], 'major_axis': [5.0, 9.0, 4.0]})
    cyc = da.cell_cycles(meas, lineage, events)
    assert cyc['track_id'].tolist() == [2]
    assert cyc['area_added'].iat[0] == 40.0 and cyc['duration'].iat[0] == 5
    # a born-and-dividing label absent from the lineage table is skipped
    cyc2 = da.cell_cycles(meas, lineage[lineage['label'] != 2], events)
    assert cyc2.empty


def test_size_control_without_bootstrap_has_slopes_but_no_intervals():
    b = np.array([1.0, 2.0, 3.0, 4.0])
    out = da.size_control_fits(b, 2 * b, n_boot=0)
    assert out['slope_division'] == pytest.approx(2.0)
    assert out['slope_added'] == pytest.approx(1.0)
    assert out['pearson_r_added'] == pytest.approx(1.0)
    assert 'slope_division_ci_low' not in out
