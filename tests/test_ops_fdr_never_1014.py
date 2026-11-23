"""Shared never-barcode generation and FDR calibration contracts (item 1014)."""
import numpy as np
import pytest



def _library(n=200, length=10, seed=31):
    """Generate a reproducible unique DNA library independent of the production builder."""
    draws = np.random.default_rng(seed).choice(list('ACGT'), size=(n * 2, length))
    return tuple(sorted({''.join(row) for row in draws})[:n])


def _distances(left, right):
    """Brute-force Hamming distances, with unequal lengths outside every tested shell."""
    return np.array([[sum(a != b for a, b in zip(x, y)) if len(x) == len(y) else -1
                      for y in right] for x in left], dtype=np.int64).reshape(len(left), len(right))


@pytest.mark.parametrize('distance', [1, 2])
def test_never_sets_have_exact_nearest_distance_and_internal_separation(distance):
    """Controls cannot be real guides, ambiguous N1 neighbours or mutually adjacent."""
    from spacr import ops_sbs as S

    guides = _library(40)
    result = S.never_barcodes(guides, distance=distance, k=4, seed=9)
    controls = result['barcodes']
    assert len(controls) >= 3 * len(guides)
    assert not set(controls) & set(guides)
    against_library = _distances(controls, guides)
    assert np.all(against_library.min(axis=1) == distance)
    within = _distances(controls, controls)
    np.fill_diagonal(within, 100)
    assert np.all(within >= 2)
    if distance == 1:
        assert np.all((against_library == 1).sum(axis=1) == 1)
        for index, guide in enumerate(guides):
            children = [control for control, row in zip(controls, against_library) if row[index] == 1]
            positions = [next(p for p, (a, b) in enumerate(zip(guide, control)) if a != b)
                         for control in children]
            assert len(positions) == len(set(positions)) <= 4
    reordered = S.never_barcodes(tuple(reversed(guides)) + guides[:2], distance=distance, k=4, seed=9)
    assert reordered['barcodes'] == controls
    np.testing.assert_array_equal(reordered['m_d'], result['m_d'])


@pytest.mark.parametrize('distance', [1, 2])
def test_opportunity_profiles_equal_brute_force_on_two_hundred_guides(distance):
    """Every distance shell and guide alignment agrees with independent all-pairs counting."""
    from spacr import ops_sbs as S

    guides = _library()
    result = S.never_barcodes(guides, distance=distance, k=3, seed=1)
    assert result['guides'] == guides
    actual = _distances(guides, result['barcodes'])
    library = _distances(guides, guides)
    for shell in (1, 2, 3):
        np.testing.assert_array_equal(result['m_d'][:, shell - 1], (actual == shell).sum(axis=1))
        np.testing.assert_array_equal(result['library_m_d'][:, shell - 1], (library == shell).sum(axis=1))
    assert result['m_d'].dtype == np.int64


def test_n2_respects_previous_decoys_and_n1_without_changing_either():
    """Excluding existing control sets prevents cross-set collisions and adjacency."""
    from spacr import ops_sbs as S

    guides = _library(40)
    decoys = tuple(S.decoy_library(guides, k=2, seed=7))
    first = S.never_barcodes(guides, distance=1, seed=5, exclude=decoys)
    excluded = decoys + first['barcodes']
    second = S.never_barcodes(guides, distance=2, seed=5, exclude=excluded)
    assert second['barcodes']
    assert np.all(_distances(second['barcodes'], excluded) >= 2)
    assert S.never_barcodes(guides, distance=1, seed=5, exclude=decoys)['barcodes'] == first['barcodes']


def test_distance_profiles_handle_short_and_mixed_length_codes():
    """Different lengths and exact self matches contribute to none of the three shells."""
    from spacr import ops_sbs as S

    guides = ('A', 'AA', 'AAA', 'AC', 'AGT')
    controls = ('AAA', 'AC', 'ACG', 'C', 'CCCC', 'TTT')
    result = S._barcode_distance_profiles(guides, controls)
    reference = _distances(guides, controls)
    np.testing.assert_array_equal(result, np.column_stack([(reference == d).sum(axis=1) for d in (1, 2, 3)]))


def test_distinct_n1_positions_bound_short_library_controls():
    """A guide supplies no more than one accepted N1 per position even if k is larger."""
    from spacr import ops_sbs as S

    result = S.never_barcodes(['AAAAAAA'], distance=1, k=20)
    assert len(result['barcodes']) == 7
    assert S.never_barcodes([], distance=2)['m_d'].shape == (0, 3)
    assert S.never_barcodes(['AAAA'], distance=1, k=0)['barcodes'] == ()


@pytest.mark.parametrize('options', [{'distance': 0}, {'distance': 3}, {'distance': True},
                                     {'distance': 1, 'k': -1}, {'distance': 1, 'seed': -1}])
def test_invalid_never_control_parameters_fail_before_generation(options):
    """Invalid control definitions cannot quietly produce a different null library."""
    from spacr import ops_sbs as S

    with pytest.raises(ValueError):
        S.never_barcodes(['AAAA'], **options)


def test_never_scale_recovers_the_known_opportunity_rate_and_wrong_call_count():
    """The independent expected-error formula determines the raw scale exactly."""
    from spacr import ops_decode as D

    result = D.never_scale([1000, 1200, 800], [3, 4, 3], [3, 4, 3],
                           [300, 400, 300], [200, 400, 400], k=4, ci=False)
    assert result['rho2'] == pytest.approx(0.01)
    assert result['expected_wrong'] == pytest.approx(10)
    assert result['scale'] == result['scale_point'] == pytest.approx(4)
    assert result['fdr_cal_pct'] == pytest.approx(100 * 10 / 3000)
    assert result['rho1'] is None and result['rho1_over_rho2'] is None


def test_distance_one_library_pairs_contribute_the_measured_n1_error_rate():
    """A library with adjacent guides needs both error shells rather than N2 alone."""
    from spacr import ops_decode as D

    result = D.never_scale([1000, 1000], [10, 10], [5, 5], [500, 500], [100, 100],
                           k=4, ci=False, never1_calls=[20, 20],
                           opp1_N1=[100, 100], opp1_L=[25, 25])
    assert result['rho1'] == pytest.approx(0.2)
    assert result['rho2'] == pytest.approx(0.01)
    assert result['rho1_over_rho2'] == pytest.approx(20)
    assert result['expected_wrong'] == pytest.approx(12)
    assert result['scale'] == pytest.approx(2.4)
    assert result['fdr_cal_pct'] == pytest.approx(0.6)


def test_never_scale_bootstraps_fields_reproducibly_and_uses_the_upper_interval():
    """The scale chosen with CI is conservative relative to the point estimate."""
    from spacr import ops_decode as D

    args = ([1000] * 6, [5] * 6, [2, 4, 3, 9, 8, 1], [300] * 6, [100] * 6)
    result = D.never_scale(*args, k=4, ci=True, reps=200, seed=4)
    repeated = D.never_scale(*args, k=4, ci=True, reps=200, seed=4)
    assert result == repeated
    assert result['scale_ci95'][0] <= result['scale_point'] <= result['scale_ci95'][1]
    assert result['scale'] == result['scale_ci95'][1]
    assert result['rho2_ci95'][0] <= result['rho2'] <= result['rho2_ci95'][1]


def test_zero_decoys_use_the_existing_tuner_substitute_and_small_scales_are_not_clipped():
    """The estimator preserves a measured scale below one; only the loop applies its floor."""
    from spacr import ops_decode as D

    zero = D.never_scale([1000, 1000], [0, 0], [3, 3], [300, 300], [150, 150], k=4, ci=False)
    assert zero['scale'] == pytest.approx(4 * 0.01 * 300 / D._AUTO_ZERO_DECOYS)
    assert zero['decoy'] == 0
    small = D.never_scale([1000], [10], [1], [100], [100], k=4, ci=False)
    assert small['scale'] == pytest.approx(0.4)
    assert small['scale_ci95'] is None


def test_n1_library_opportunities_cannot_be_used_without_their_control_pass():
    """Skipping a required error shell cannot silently under-estimate wrong calls."""
    from spacr import ops_decode as D

    with pytest.raises(ValueError, match='Distance-one'):
        D.never_scale([1000], [5], [3], [300], [100], k=4, opp1_L=[10])
    with pytest.raises(ValueError, match='both N1'):
        D.never_scale([1000], [5], [3], [300], [100], k=4, never1_calls=[1])


@pytest.mark.parametrize('bad', [[-1], [np.nan], [np.inf], [1, 2]])
def test_never_estimator_rejects_invalid_or_misaligned_field_counts(bad):
    """Invalid opportunity counts cannot enter a calibrated production FDR."""
    from spacr import ops_decode as D

    with pytest.raises(ValueError, match='aligned, finite, nonnegative'):
        D.never_scale([1000], [5], [3], bad, [100], k=4)


def _never_well(fields=40, cycle=False):
    """Two genuine tuner choices with planted control rates and enough held-out calls."""
    counts = np.empty((fields, 2, 2), np.int64)
    counts[:, 0] = (800, 10)
    counts[:, 1] = (600, 2)
    opportunities = [1000, 50] if cycle else [400, 100]
    never = {'never2_calls': np.full((fields, 2), 5),
             'opp2_N2': np.full((fields, 2), 500),
             'opp2_L': np.tile(opportunities, (fields, 1)),
             'library_size': 200, 'never_sets': {'k': 4, 'seed': 1}}
    return counts, never, np.full(fields, 1000), np.arange(fields) + 1


def test_never_calibration_can_lower_the_entered_scale_and_preserves_production_counts():
    """The real tuner converges on the measured opportunity scale rather than an entered floor."""
    from spacr import ops_decode as D

    counts, never, nuclei, sites = _never_well()
    before = counts.copy()
    result = D.calibrate_fdr_scale(counts, None, nuclei, sites, [('floor', [0.0, 0.2])],
                                   k=4, scale=8, method='never', never=never, ci=False, fdr_max=10)
    assert result['fdr_scale'] == pytest.approx(1.6)
    assert result['index'] == [0]
    assert result['calibration']['converged']
    assert result['calibration']['rounds'][0]['scale'] == 8
    assert len(result['calibration']['rounds']) == 2
    real = 800 * len(result['tune_sites'])
    decoy = 10 * len(result['tune_sites'])
    assert result['tune']['fdr_per_100'] == pytest.approx(D.fdr(real, decoy, 4, result['fdr_scale']))
    np.testing.assert_array_equal(counts, before)


def test_a_real_threshold_cycle_keeps_the_larger_tested_scale_and_its_choice():
    """A loose point needs scale four, which selects a strict point needing scale one."""
    from spacr import ops_decode as D

    counts, never, nuclei, sites = _never_well(cycle=True)
    result = D.calibrate_fdr_scale(counts, None, nuclei, sites, [('floor', [0.0, 0.2])],
                                   k=4, scale=1, method='never', never=never, ci=False, fdr_max=0.5)
    assert result['fdr_scale'] == pytest.approx(4)
    assert result['index'] == [1]
    assert result['calibration']['cycle']
    assert not result['calibration']['converged']
    assert [row['scale'] for row in result['calibration']['rounds']] == [1, 4, 1]
    assert result['tune']['fdr_per_100'] == pytest.approx(D.fdr(600, 2, 4, 4))


def test_report_half_never_calls_cannot_change_the_tuning_scale():
    """Changing only held-out errors changes the report estimate but cannot train the scale."""
    from spacr import ops_decode as D

    counts, never, nuclei, sites = _never_well()
    args = (counts, None, nuclei, sites, [('floor', [0.0, 0.2])])
    original = D.calibrate_fdr_scale(*args, k=4, scale=1, method='never', never=never, ci=False, fdr_max=10)
    changed = {**never, 'never2_calls': never['never2_calls'].copy()}
    changed['never2_calls'][np.isin(sites, original['report_sites'])] *= 20
    heldout = D.calibrate_fdr_scale(*args, k=4, scale=1, method='never', never=changed, ci=False, fdr_max=10)
    assert heldout['fdr_scale'] == original['fdr_scale']
    assert heldout['chosen'] == original['chosen']
    assert heldout['calibration']['scale_report_half'] == pytest.approx(20 * original['calibration']['scale_report_half'])


@pytest.mark.parametrize('reason', ['library', 'calls'])
def test_insufficient_never_evidence_keeps_the_entered_scale(reason):
    """Small libraries and sparse tuning-half controls cannot recalibrate production calls."""
    from spacr import ops_decode as D

    counts, never, nuclei, sites = _never_well()
    if reason == 'library':
        never['library_size'] = 49
    else:
        never['never2_calls'][:] = 0
    result = D.calibrate_fdr_scale(counts, None, nuclei, sites, [('floor', [0.0, 0.2])],
                                   k=4, scale=3, method='never', never=never, ci=False, fdr_max=10)
    assert result['fdr_scale'] == 3
    assert result['calibration']['status'] == 'insufficient_never_calls'
    if reason == 'library':
        assert result['calibration']['reason'] == 'insufficient_library_guides'
    assert result['calibration']['scale_ci95'] is None


def test_shuffle_is_reported_but_used_only_by_the_combined_never_method():
    """The prefix reporting rule does not turn a shuffle report into a calibration floor."""
    from spacr import ops_decode as D

    counts, never, nuclei, sites = _never_well()
    shuffled = np.full((len(sites), 2), 30)
    args = (counts, shuffled, nuclei, sites, [('floor', [0.0, 0.2])])
    measured = D.calibrate_fdr_scale(*args, k=4, scale=1, method='never', never=never, ci=False, fdr_max=10)
    combined = D.calibrate_fdr_scale(*args, k=4, scale=1, method='never+shuffle', never=never, ci=False, fdr_max=10)
    assert measured['fdr_scale'] == pytest.approx(1.6)
    assert measured['calibration']['shuffle_call_rate'] == pytest.approx(3)
    assert combined['fdr_scale'] == pytest.approx(6.4)


def _grid_votes():
    """Four objects whose primary and fallback guides change the opportunity sum."""
    return {'owned': {1, 2, 3, 4},
            'primary': {1: {'guide': 'AAAA', 'n_support': 1, 'quality': 0.1},
                        2: {'guide': 'CCCC', 'n_support': 2, 'quality': 0.5},
                        4: {'guide': 'TTTT', 'n_support': 1, 'quality': 0.8}},
            'pixel': {0.1: {1: {'guide': 'GGGG'}, 3: {'guide': 'GGGG'}, 4: {'guide': 'AAAA'}}},
            'tallies': None, 'reads': 12, 'reads_owned': 10, 'reads_near': 1, 'reads_ambiguous': 1}


def _grid_profiles():
    """Independently readable shell weights for three real guides and four libraries."""
    base = np.array([[1, 10, 100], [2, 20, 200], [3, 30, 300]], np.int64)
    return {'guides': ('AAAA', 'CCCC', 'GGGG'),
            **{name: base * factor for name, factor in [('L', 1), ('D', 2), ('N1', 3), ('N2', 4)]}}


@pytest.mark.parametrize('pixels,weights,real', [(True, [6, 8], [3, 3]), (False, [3, 2], [2, 1])])
def test_grid_opportunities_use_the_same_selected_primary_and_pixel_calls(pixels, weights, real):
    """Changed guide identity can change opportunities while real and decoy totals stay fixed."""
    from spacr import ops_decode as D

    args = (_grid_votes(), [0.0, 0.2], [0.1], {'TTTT'}, pixels)
    plain = D._field_candidates(*args)
    actual = D._field_candidates(*args, opportunity_profiles=_grid_profiles())
    np.testing.assert_array_equal(actual['counts'], plain['counts'])
    np.testing.assert_array_equal(actual['counts'][:, 0, 0], real)
    np.testing.assert_array_equal(actual['counts'][:, 0, 1], [1, 1])
    assert len(actual['opportunities']) == 12
    for name, factor in [('L', 1), ('D', 2), ('N1', 3), ('N2', 4)]:
        for shell in (1, 2, 3):
            np.testing.assert_array_equal(actual['opportunities'][f'opp{shell}_{name}'][:, 0],
                                          np.array(weights) * factor * 10 ** (shell - 1))


def test_joint_grid_stacks_opportunities_and_reuses_the_existing_pixel_pass(monkeypatch):
    """Read-threshold axes retain opportunity sums from the actual shared field counter."""
    from spacr import ops_decode as D

    pixel_requests = []

    def votes(evidence, rule, *args, **kwargs):
        """Supply known votes while recording whether the shared caller asks for pixels again."""
        pixel_requests.append(rule['pixels'])
        return _grid_votes()

    monkeypatch.setattr(D, '_field_votes', votes)
    result = D._joint_field_candidates({}, {'pixels': True}, ['AAAA', 'CCCC', 'GGGG'], None,
                                       {'TTTT'}, spot_thresholds=[1, 2], floors=[0.0, 0.2],
                                       pixel_qualities=[0.1], footprint=10, max_distance=0,
                                       min_fraction=0.6, crosstalk='argmax_median',
                                       opportunity_profiles=_grid_profiles())
    assert pixel_requests == [True, False]
    assert result['counts'].shape == (2, 2, 1, 2)
    np.testing.assert_array_equal(result['opportunities']['opp2_L'], [[[60], [80]], [[60], [80]]])


def test_a_decoy_only_field_has_zero_library_opportunities():
    """Decoy calls cannot create real-guide exposure in an otherwise empty field."""
    from spacr import ops_decode as D

    votes = _grid_votes()
    votes['primary'] = {4: votes['primary'][4]}
    result = D._field_candidates(votes, [0.0], [0.1], {'TTTT'}, False,
                                 opportunity_profiles=_grid_profiles())
    np.testing.assert_array_equal(result['counts'], [[[0, 1]]])
    assert all(not np.any(value) for value in result['opportunities'].values())


def test_unknown_called_guides_cannot_silently_lose_their_opportunities():
    """A stale profile library fails instead of underestimating wrong calls."""
    from spacr import ops_decode as D

    with pytest.raises(ValueError, match='every called guide'):
        D._field_candidates(_grid_votes(), [0.0], [0.1], {'TTTT'}, True,
                             opportunity_profiles={'guides': ('AAAA',), 'L': np.ones((1, 3), np.int64)})


def _report_choice(multiplier=1):
    """Build a real two-axis calibrated choice and matching production field records."""
    from spacr import ops_decode as D

    counts, never, nuclei, sites = _never_well()
    control = {key: value[..., None] if isinstance(value, np.ndarray) else value
               for key, value in never.items()}
    control['opp2_L'] = control['opp2_L'] * multiplier
    result = D.calibrate_fdr_scale(counts[:, :, None, :], None, nuclei, sites,
                                   [('single_read_quality', [0.0, 0.2]), ('pixel_quality', [0.1])],
                                   k=4, scale=8, method='never', never=control, ci=False, fdr_max=10)
    decoded = [{'site': int(site), 'rule_a': {'nuclei': 1000, 'assigned_real': 800, 'assigned_decoy': 10}}
               for site in sites]
    return result, decoded


def test_full_decode_reports_write_all_calibration_keys_without_requiring_shuffle(tmp_path):
    """The existing reports accept never-only controls and expose their JSON/CSV artifacts."""
    import json
    from spacr import ops_decode as D
    from spacr.tabular import read_table

    choice, decoded = _report_choice()
    db = str(tmp_path / 'measurements.db')
    threshold_paths = D._write_threshold_report(db, 'P', 'A1', choice)
    report = D.write_well_reports(db, 'P', 'A1', decoded,
                                  {'decoy_k': 4, 'fdr_scale': choice['fdr_scale'], 'threshold_choice': choice}, [])
    required = {'method', 'entered_scale', 'scale', 'scale_ci95', 'scale_report_half',
                'rounds', 'converged', 'rho1', 'rho2', 'rho1_over_rho2', 'fdr_cal_pct',
                'fdr_at_entered_scale_pct', 'never_sets', 'real_call_change_pct', 'fields'}
    from pathlib import Path

    calibration = json.loads(Path(report['paths']['fdr_calibration']).read_text())
    assert required <= calibration.keys()
    assert required <= report['calibration'].keys()
    assert required <= json.loads(Path(threshold_paths['thresholds']).read_text())['calibration'].keys()
    assert report['fdr_per_100'] == pytest.approx(0.5)
    assert report['fdr_calibration'] == 'never'
    assert 'shuffle_call_rate_pct' not in report
    table = read_table(report['paths']['fdr_calibration_csv'], canonicalise=False, report=None)
    assert len(table) == 1 and required <= set(table.columns)
    assert table.iloc[0]['scale'] == pytest.approx(1.6)


def test_plate_fdr_uses_each_well_applied_scale_in_the_same_fdr_formula(tmp_path):
    """A second well cannot overwrite the first well's calibrated FDR scale."""
    import json
    from pathlib import Path
    from spacr import ops_decode as D

    db = str(tmp_path / 'measurements.db')
    for well, multiplier in [('A1', 1), ('A2', 2)]:
        choice, decoded = _report_choice(multiplier)
        report = D.write_well_reports(db, 'P', well, decoded,
                                      {'decoy_k': 4, 'fdr_scale': choice['fdr_scale'], 'threshold_choice': choice}, [])
    summary = json.loads(Path(report['paths']['plate_summary']).read_text())
    assert summary['fdr_per_100'] == pytest.approx(D.fdr(64000, 400 * 1.6 + 400 * 3.2, 4, 1))
    assert summary['per_well']['A1']['fdr_scale'] == pytest.approx(1.6)
    assert summary['per_well']['A2']['fdr_scale'] == pytest.approx(3.2)
    assert summary['per_well']['A1']['fdr_per_100'] == pytest.approx(0.5)
    assert summary['per_well']['A2']['fdr_per_100'] == pytest.approx(1.0)


def test_fixed_scale_rerun_ignores_a_stale_never_sidecar(tmp_path):
    """A prior never report cannot alter the output of settings that request no calibration."""
    import json
    from pathlib import Path
    from spacr import ops_decode as D

    db = str(tmp_path / 'measurements.db')
    choice, decoded = _report_choice()
    earlier = D.write_well_reports(db, 'P', 'A1', decoded,
                                   {'decoy_k': 4, 'fdr_scale': choice['fdr_scale'], 'threshold_choice': choice}, [])
    report = D.write_well_reports(db, 'P', 'A1', decoded, {'decoy_k': 4, 'fdr_scale': 1.268}, [])
    assert Path(earlier['paths']['fdr_calibration']).exists()
    assert 'calibration' not in report
    summary = json.loads(Path(report['paths']['plate_summary']).read_text())
    assert summary['fdr_per_100'] == round(D.fdr(32000, 400, 4, 1.268), 3)


def test_calibration_warning_applies_to_n2_and_not_the_report_only_n1(tmp_path):
    """One summary is logged per well; the N1 perturbation has no warning guard."""
    import warnings
    from spacr import ops_decode as D

    choice, _ = _report_choice()
    calibration = {**choice['calibration'], 'real_call_change_pct': {'n1': -1.1, 'n2': -0.2}}
    messages = []
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        D._write_calibration_report(str(tmp_path), 'P', 'A1', calibration, log=messages.append)
    assert len(messages) == 1 and 'never-should-appear' in messages[0]
    assert not any('N2 control voting' in str(row.message) for row in observed)
    calibration['real_call_change_pct']['n2'] = -0.6
    with pytest.warns(RuntimeWarning, match='0.5 % perturbation guard'):
        D._write_calibration_report(str(tmp_path), 'P', 'A1', calibration, log=None)
    paths = D._write_calibration_report(str(tmp_path / 'prefix'), 'P', 'A1',
                                       choice['calibration'], csv=False, log=None)
    assert set(paths) == {'fdr_calibration'}


def test_zero_control_opportunities_report_an_unavailable_error_rate():
    """An undefined control rate must not masquerade as evidence of zero wrong calls."""
    from spacr import ops_decode as D

    result = D.never_scale([1000], [10], [40], [0], [100], k=4)
    assert result['rho2'] is None
    assert result['scale'] is None and result['scale_point'] is None
    assert result['fdr_cal_pct'] is None


def test_zero_tuning_opportunities_cannot_lower_the_entered_scale():
    """Insufficient exposure preserves the entered scale even when controls have calls."""
    from spacr import ops_decode as D

    counts, never, nuclei, sites = _never_well()
    never['opp2_N2'][:] = 0
    result = D.calibrate_fdr_scale(counts, None, nuclei, sites, [('floor', [0.0, 0.2])],
                                   k=4, scale=3, method='never', never=never, ci=False, fdr_max=10)
    assert result['fdr_scale'] == 3
    assert result['calibration']['status'] == 'insufficient_never_opportunities'
    assert result['calibration']['scale_ci95'] is None


def _never_api_contracts():
    """Read the exact source-only API arrival and revision contract without importing spaCR."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent / 'data/release_contracts/1014_never_api_sources_2026-11-22.json'
    from tests.test_ops_never_s2j_parity_1014 import _current_s2j_records
    from tests.test_docfix_never_docstrings_1014 import _current_docfix_records

    return _current_docfix_records(_current_s2j_records(json.loads(path.read_text())['symbols']))


def _validate_never_api_source(symbol, record, source=None):
    """Bind body, hash, all parameter names/defaults/types and return annotation to source syntax."""
    import ast
    import hashlib
    from pathlib import Path

    if source is None:
        source = (Path(__file__).resolve().parents[1] / record['file']).read_text()
    if symbol == 'spacr.ops_prefix.prefix_settings':
        from tests.test_ops_neighbour_sources_1013 import _neighbour_predecessor_source
        source = _neighbour_predecessor_source(symbol, record, source)
    name = symbol.rsplit('.', 1)[-1]
    node = next(item for item in ast.parse(source).body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name)
    body = ast.get_docstring(node)
    assert body == record['source'], symbol
    assert hashlib.sha256(body.encode()).hexdigest() == record['source_sha256'], symbol
    assert ast.unparse(node.args) == record['signature'], symbol
    annotation = ast.unparse(node.returns) if node.returns is not None else None
    assert annotation == record['return_annotation'], symbol
    return body


def _current_never_api(entries):
    """Apply only exact 1014 source revisions to their immutable historical API snapshots."""
    import hashlib

    result = dict(entries)
    for symbol, record in _never_api_contracts().items():
        if record['before'] is None or symbol not in result:
            continue
        assert result[symbol] == record['before'], symbol
        assert hashlib.sha256(result[symbol].encode()).hexdigest() == record['before_sha256'], symbol
        result[symbol] = _validate_never_api_source(symbol, record)
    from tests.test_ops_never_s2j_parity_1014 import _current_s2j_api

    return _current_s2j_api(result)


def _validated_never_arrivals(callables):
    """Check and project exactly the two newly documented never-control callables."""
    import json
    from pathlib import Path

    records = _never_api_contracts()
    path = Path(__file__).resolve().parent / 'data/release_contracts/1014_never_api_arrivals_2026-11-22.json'
    entries = json.loads(path.read_text())
    entries = _current_never_api(entries)
    assert set(entries) == {'spacr.ops_sbs.never_barcodes', 'spacr.ops_decode.never_scale'}
    found = {item.symbol: item for item in callables if item.symbol in entries}
    assert found.keys() == entries.keys()
    for symbol, item in found.items():
        assert item.category == 'function' and item.exposure == 'autoapi'
        assert item.parameters == item.accepted_documented_parameters
        assert item.variant_count == 1 and item.docless_variant_count == 0
        assert entries[symbol] == _validate_never_api_source(symbol, records[symbol])
        assert item.docstring == entries[symbol]
    return [item for item in callables if item.symbol not in entries]


def test_never_api_contracts_bind_the_actual_source_and_historical_revisions():
    """Every new and revised public body has an exact, admitted source and signature."""
    records = _never_api_contracts()
    assert len(records) == 11
    for symbol, record in records.items():
        assert _validate_never_api_source(symbol, record) == record['source']
    before = {symbol: record['before'] for symbol, record in records.items() if record['before'] is not None}
    current = _current_never_api(before)
    assert len(before) == 9 and current.keys() == before.keys()
    assert all(current[symbol] == records[symbol]['source'] for symbol in current)


@pytest.mark.parametrize('old,new', [
    ('*, distance,', '*, distance_changed,'),
    ('k=4, seed=1', 'k=4.0, seed=1'),
    ('seed=1, exclude=()', 'seed=2, exclude=()'),
    ('def never_barcodes(library, *', 'def never_barcodes(other_library, *'),
    ('Substitution-built barcodes that no cell', 'Incorrect source body that no cell')])
def test_never_api_contract_rejects_parameter_default_type_and_body_drift(old, new):
    """Signature or body drift cannot be hidden behind an unchanged release fixture."""
    from pathlib import Path

    symbol = 'spacr.ops_sbs.never_barcodes'
    record = _never_api_contracts()[symbol]
    source = (Path(__file__).resolve().parents[1] / record['file']).read_text()
    changed = source.replace(old, new, 1)
    assert changed != source
    with pytest.raises(AssertionError):
        _validate_never_api_source(symbol, record, changed)


def _full_never_field():
    """Reuse the established caller fixture with independent never-control context."""
    from tests.test_ops_decode_threshold_auto_939 import _random_field, _RULE, _KW
    from spacr import ops_decode as D

    library, decoys, evidence = _random_field(objects=80)
    context = D._never_context(tuple(sorted(library)), tuple(sorted(decoys)), 4, 1, 'n1_n2')
    rule = D.decode_rule({**_RULE, 'ops_read_neighbour_discount': 1.0})
    return library, decoys, evidence, context, rule, dict(_KW)


def test_control_context_matches_independent_reference_builds():
    """Production setup adds no D/N1 exclusions to the independent paper controls."""
    from spacr import ops_decode as D, ops_sbs as S

    library = _library(60)
    decoys = tuple(sorted(S.decoy_library(library, k=4)))
    context = D._never_context(library, decoys, 4, 1, 'n1_n2')
    for name, distance in [('N1', 1), ('N2', 2)]:
        reference = S.never_barcodes(library, distance=distance, k=4, seed=1)
        assert context['controls'][name]['barcodes'] == reference['barcodes']
        np.testing.assert_array_equal(context['profiles'][name], reference['m_d'])
    assert context is D._never_context(library, decoys, 4, 1, 'n1_n2')
    assert set(D._never_context(library, decoys, 4, 1, 'n2')['controls']) == {'N2'}


def test_full_control_grids_equal_direct_production_calls_and_reuse_only_same_whitelist_pixels(monkeypatch):
    """Each grid entry is the unchanged caller on its own control whitelist."""
    from spacr import ops_decode as D

    library, _decoys, evidence, context, rule, keywords = _full_never_field()
    floors, qualities, spots = [0.0, 0.2], [0.0, 0.1], [315.0, 500.0]
    counts, pixels = D._field_never_counts(
        evidence, rule, library, context, spot_thresholds=spots,
        floors=floors, pixel_qualities=qualities,
        **{k: v for k, v in keywords.items() if k != 'threshold'})
    for name, data in context['controls'].items():
        matcher, decoys = D.matcher_for(library, data['barcodes'])
        for a, threshold in enumerate(spots):
            for b, floor in enumerate(floors):
                for c, quality in enumerate(qualities):
                    direct = D.field_calls(
                        evidence, {**rule, 'pixel_quality': quality}, library, matcher, decoys,
                        single_read_quality=floor, **{**keywords, 'threshold': threshold})
                    assert counts[f'never{name[-1]}_calls'][a, b, c] == direct['assigned_decoy']
                    assert counts[f'real_calls_{name.lower()}'][a, b, c] == direct['assigned_real']
    calls = []
    original = D._field_votes

    def record_votes(*args, **kwargs):
        """Record whether a reused same-whitelist pass requests pixel voting again."""
        calls.append(args[1]['pixels'])
        return original(*args, **kwargs)

    monkeypatch.setattr(D, '_field_votes', record_votes)
    repeated, reused = D._field_never_counts(
        evidence, rule, library, context, spot_thresholds=spots,
        floors=floors, pixel_qualities=qualities, pixel_votes=pixels,
        **{k: v for k, v in keywords.items() if k != 'threshold'})
    assert calls == [False] * (len(spots) * 2)
    assert all(reused[name] is pixels[name] for name in pixels)
    for key in counts:
        np.testing.assert_array_equal(repeated[key], counts[key])


@pytest.mark.parametrize('mode', ['manual', 'auto'])
def test_never_threshold_settings_keep_manual_values_and_auto_grid(mode):
    """Manual calibration uses entered thresholds; automatic mode retains its search grid."""
    from spacr import ops_decode as D

    rule = D.decode_rule({'ops_decoy_library_k': 4, 'ops_decode_pixel_fallback': True})
    settings = {'ops_decode_threshold_mode': mode, 'ops_decode_fdr_calibration': 'never',
                'ops_single_read_quality': 0.17,
                'ops_decode_auto_spot_grid': [200.0, 400.0]}
    chosen = D.threshold_settings(settings, rule)
    assert chosen['mode'] == mode and chosen['fdr_calibration'] == 'never'
    if mode == 'manual':
        assert chosen['floors'] == [0.17]
        assert chosen['pixel_qualities'] == [rule['pixel_quality']]
        assert chosen['spot_thresholds'] == []
    else:
        assert chosen['floors'] == list(D.AUTO_GRID['ops_single_read_quality'])
        assert chosen['spot_thresholds'] == [200.0, 400.0]
    fallback = D.threshold_settings({**settings, 'ops_decode_threshold_mode': 'manual'},
                                    {**rule, 'decoy_k': 0})
    assert fallback == {'mode': 'manual', 'fdr_calibration': 'never', 'no_decoys': True}


def test_full_base_candidates_record_opportunities_without_changing_production_votes(monkeypatch):
    """The engine carries guide opportunities alongside precisely the original call grid."""
    from spacr import ops_decode as D, ops_engine as E

    library, decoys, evidence, context, rule, keywords = _full_never_field()
    monkeypatch.setattr(E, '_LIBRARY', frozenset(library))
    monkeypatch.setattr(E, '_DECOYS', frozenset(decoys))
    rule = {**rule, **keywords}
    auto = {'floors': [0.0, 0.2], 'pixel_qualities': [0.0, 0.1]}
    original = E._rule_a_calls(evidence, rule, auto, [315.0, 500.0], {})
    measured = E._rule_a_calls(evidence, rule,
                               {**auto, 'never': {'k': 4, 'seed': 1, 'sets': 'n1_n2'}},
                               [315.0, 500.0], {})
    np.testing.assert_array_equal(original['counts'], measured['counts'])
    assert len(measured['opportunities']) == 12
    for a, threshold in enumerate([315.0, 500.0]):
        for b, floor in enumerate(auto['floors']):
            for c, quality in enumerate(auto['pixel_qualities']):
                calls = D._candidate_calls(measured['by_spot'][a], floor, quality, frozenset(decoys))
                real = [hit['guide'] for hit in calls['calls'].values() if not hit['decoy']]
                for name in ('L', 'D', 'N1', 'N2'):
                    expected = np.sum([context['profiles'][name][library.index(g)] for g in real],
                                      axis=0) if real else np.zeros(3)
                    for d in (1, 2, 3):
                        assert measured['opportunities'][f'opp{d}_{name}'][a, b, c] == expected[d - 1]


def _prefix_never_field():
    """Build cycle-masked prefix reads containing real, far and both never classes."""
    import pandas as pd
    from spacr import ops_prefix as P

    chosen = P.prefix_settings({'ops_library': 'unused.csv', 'ops_prefix_length': 10,
                                 'ops_cycle_mask': [3], 'ops_sbs_cycles': 12,
                                 'ops_prefix_decoy_k': 4, 'ops_prefix_fdr_calibration': 'never'})
    table = pd.DataFrame({'sgRNA': _library(60, length=12),
                          'gene_symbol': [f'G{i}' for i in range(60)]})
    library = P.spacer_prefix_library(table, prefix_length=10, n_cycles=12, cycle_mask=[3])
    matching = P.prefix_decoy_library(library, 4)
    context = P._prefix_never_context(library, matching, chosen)
    keys = ([library.keys[0], library.keys[1]]
            + [key for key in matching.keys if key not in library.keys][:2]
            + list(context['controls']['N1']['barcodes'][:2])
            + list(context['controls']['N2']['barcodes'][:2]))
    reads = []
    for key in keys:
        full = list('A' * 12)
        for position, letter in zip(library.positions, key):
            full[position - 1] = letter
        reads.extend([''.join(full)] * 2)
    matched = P._matched_rows(reads, matching, {}, metric='hamming', max_distance=1)
    frame = matched.assign(cell=np.repeat(np.arange(1, len(keys) + 1), 2),
                            y=np.arange(len(reads)), x=0, quality=0.8)
    return library, matching, context, chosen, frame


@pytest.mark.parametrize('weighted', [False, True])
def test_prefix_opportunities_and_controls_use_the_existing_winners_with_cycle_masks(weighted):
    """Prefix control passes preserve production winners and all weighted-grid decisions."""
    from spacr import ops_prefix as P

    library, _matching, context, chosen, frame = _prefix_never_field()
    assert len(library.positions) == 9
    if weighted:
        chosen = {**chosen, 'ops_prefix_quality_method': 'weighted_majority'}
    axes = ([(P.WEIGHT_GRID_KEYS[0], [0.0]), (P.WEIGHT_GRID_KEYS[1], [1.0]),
             (P.WEIGHT_GRID_KEYS[2], [0.0, 1.0])] if weighted else [])
    plan = P._weight_plan(chosen, axes)
    method = chosen['ops_prefix_quality_method']
    untouched = frame.copy(deep=True)
    for distance in (0, 1):
        result = P._prefix_never_counts(frame, context, quality=0, distance=distance,
                                        metric='hamming', method=method, minority_k=0,
                                        plan=plan, chosen=chosen, neighbours=None)
        expected_counts = P._setting_counts(P._at_setting(frame, 0, distance),
                                             method, 0, plan, chosen, None)
        np.testing.assert_array_equal(result['counts'], expected_counts)
        for name, whitelist in context['whitelists'].items():
            direct = P._shuffle_frame(frame, whitelist, {}, metric='hamming', max_distance=distance)
            expected = np.asarray(P._setting_counts(P._at_setting(direct, 0, distance),
                                                     method, 0, plan, chosen, None))
            np.testing.assert_array_equal(result['never'][f'never{name[-1]}_calls'], expected[..., 1])
            np.testing.assert_array_equal(result['never'][f'real_calls_{name.lower()}'], expected[..., 0])
        base = P._at_setting(frame, 0, distance).assign(plate='P', well='W1', site=1)
        for index in np.ndindex(np.asarray(expected_counts).shape[:-1]):
            best = dict(zip((key for key, _ in axes),
                            [axes[i][1][value] for i, value in enumerate(index)]))
            cells = P.prefix_cell_calls(base, method=method, **(P._weight_arguments(chosen, best)
                                                               if weighted else {}))
            real = cells[cells['quality'].isin(('perfect', 'great'))
                         & ~cells['gene'].isin((P.DECOY_GENE, P.NEARMISS_GENE))]
            for name in ('L', 'D', 'N1', 'N2'):
                expected = np.sum([context['profiles'][name][library.keys.index(key)]
                                   for key in real['prefix']], axis=0)
                for d in (1, 2, 3):
                    assert np.asarray(result['never'][f'opp{d}_{name}'])[index] == expected[d - 1]
    assert all((name, distance) in context['memos'] for name in ('N1', 'N2') for distance in (0, 1))
    assert frame.equals(untouched)


@pytest.mark.parametrize('tuning,sets', [(False, 'n1_n2'), (True, 'n1_n2'), (False, 'n2')])
def test_prefix_never_runs_end_to_end_and_always_reports_shuffle(tmp_path, tuning, sets):
    """Rendered PERISCOPE-style sites use the shared estimator and write its full report."""
    import json
    from pathlib import Path
    from tests.test_ops_autotune_942 import _run
    from spacr.tabular import read_table

    result = _run(tmp_path, f'never-{tuning}-{sets}', ops_prefix_decoy_k=4,
                   ops_prefix_fdr_calibration='never', ops_never_barcode_sets=sets,
                   ops_prefix_threshold_tuning='per_well_decoy' if tuning else 'none',
                   ops_prefix_tuning_grid={'ops_read_threshold': [10, 5000],
                                           'ops_prefix_min_read_quality': [0, 0.5]})
    folder = Path(result['folder'])
    report = json.loads((folder / 'fdr_W1.json').read_text()) if (folder / 'fdr_W1.json').exists() \
        else json.loads((folder / 'fdr_CP9_W1.json').read_text())
    calibration = json.loads((folder / 'fdr_calibration_W1.json').read_text())
    assert calibration['method'] == 'never'
    assert calibration['status'] == 'insufficient_never_calls'
    assert calibration['reason'] == 'insufficient_library_guides'
    assert calibration['scale'] == 1
    assert {'rho1', 'rho2', 'fields', 'never_sets', 'real_call_change_pct',
            'scale_report_half', 'fdr_cal_pct', 'shuffle_call_rate'} <= calibration.keys()
    assert 'shuffle_call_rate_pct' in report
    assert report['fdr_calibration'] == 'never'
    assert ('n1' in calibration['real_call_change_pct']) == (sets == 'n1_n2')
    summary = read_table(folder / 'site_summary.csv', canonicalise=False, report=None)
    assert summary['cells_real'].tolist() == [3] * 4
    assert summary['fdr_scale'].eq(calibration['scale']).all()


def test_prefix_never_without_decoys_reports_a_fixed_scale_fallback(tmp_path):
    """No-decoy runs finish with a named fallback instead of invoking an invalid tuner."""
    import json
    from pathlib import Path
    from tests.test_ops_autotune_942 import _run

    result = _run(tmp_path, 'never-no-decoys', ops_prefix_fdr_calibration='never')
    report = json.loads((Path(result['folder']) / 'fdr_calibration_W1.json').read_text())
    assert report['method'] == 'none' and report['requested_method'] == 'never'
    assert report['status'] == 'no_decoys' and report['scale'] == 1


def test_full_field_wiring_calibrates_fixed_thresholds_and_writes_reports_without_parent_worker_globals(tmp_path, monkeypatch):
    """Field workers carry the complete control arrays through fixed-mode report writing."""
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from spacr import ops_decode as D, ops_engine as E

    library, decoys, evidence, _context, rule, keywords = _full_never_field()
    monkeypatch.setattr(E, '_LIBRARY', frozenset(library))
    monkeypatch.setattr(E, '_DECOYS', frozenset(decoys))
    monkeypatch.setattr(D, 'field_evidence', lambda *args, **kwargs: evidence)
    thresholds = D.threshold_settings({'ops_decode_threshold_mode': 'manual',
                                       'ops_decode_fdr_calibration': 'never',
                                       'ops_single_read_quality': 0.1,
                                       'ops_decode_auto_ci': False}, rule)
    auto = {key: thresholds[key] for key in ('floors', 'pixel_qualities', 'spot_thresholds')}
    auto['never'] = {'k': 4, 'seed': 1, 'sets': 'n1_n2'}
    rule = {**rule, **keywords, 'single_read_quality': 0.1, 'fdr_scale': 8,
            'auto': auto, 'store_cache': False}
    field = SimpleNamespace(kept=list(range(evidence['values'].shape[1])), channel_shifts={})
    stack = np.zeros((len(field.kept), 4, 8, 8), np.float32)
    task = {'rule_a': rule, 'crosstalk': 'none'}
    decoded = [{**E._decode_field_rule_a(task, field, stack, stack, np.array([[3, 3]]), 0, False),
                'site': site} for site in (1, 2, 3, 4)]
    expected = [D._candidate_calls(row['candidates'], 0.1, rule['pixel_quality'], frozenset(decoys))
                for row in decoded]
    monkeypatch.setattr(E, '_LIBRARY', frozenset())
    monkeypatch.setattr(E, '_DECOYS', frozenset())
    db = str(tmp_path / 'measurements.db')
    applied = E._apply_auto_thresholds(db, 'P', 'A1', rule, thresholds, decoded, decoys)
    assert applied['threshold'] == rule['threshold']
    assert applied['single_read_quality'] == 0.1
    assert applied['pixel_quality'] == rule['pixel_quality']
    for row, original in zip(decoded, expected):
        assert row['rule_a']['calls'] == original['calls']
    report = D.write_well_reports(db, 'P', 'A1', decoded, applied, [])
    calibration = json.loads(Path(report['paths']['fdr_calibration']).read_text())
    assert calibration['method'] == 'never'
    assert calibration['never_sets']['sizes']['L'] == len(library)
    assert report['fdr_per_100'] == pytest.approx(
        round(D.fdr(report['assigned'], report['assigned_decoy'], rule['decoy_k'], applied['fdr_scale']), 3))
    assert not list(Path(report['paths']['fdr_calibration']).parent.glob('decode_tuning_curve*'))


@pytest.mark.parametrize('sort', ['count', 'quality', 'peak'])
def test_prefix_never_sorted_passes_keep_the_existing_cell_call_rule(sort):
    """Controls retain peak values and reuse production sort and tie decisions."""
    from spacr import ops_prefix as P

    _library_object, _matching, context, chosen, frame = _prefix_never_field()
    frame = frame.assign(peak=np.arange(len(frame), dtype=float) + 1)
    plan = P._weight_plan(chosen, [])
    result = P._prefix_never_counts(frame, context, quality=0, distance=1, metric='hamming',
                                    method='spacr', minority_k=0, plan=plan,
                                    chosen=chosen, neighbours=None, sort=sort)
    for name, whitelist in context['whitelists'].items():
        direct = P._shuffle_frame(frame, whitelist, {}, metric='hamming', max_distance=1)
        assert direct['peak'].tolist() == frame['peak'].tolist()
        cells = P._prefix_cell_calls(direct, keys=['cell'], method='spacr', sort=sort)
        hit = cells['quality'].isin(('perfect', 'great'))
        assert result['never'][f'never{name[-1]}_calls'] == int((hit & cells['gene'].eq(P.DECOY_GENE)).sum())


def test_prefix_grid_never_calibration_reuses_fixed_grid_and_reports_applied_scale(tmp_path):
    """The existing rendered-site grid retains both control rates and never diagnostics."""
    import pandas as pd
    from tests.test_ops_grid_search_960 import _settings
    from spacr import ops_grid_search as G

    settings = _settings(tmp_path, 'grid-never', ops_prefix_fdr_calibration='never',
                          ops_prefix_threshold_tuning='none', ops_grid_tiles='1,3',
                          ops_grid_params={'ops_read_threshold': [10, 5000]})
    result = G._run_grid_search(settings)
    table = pd.read_csv(result['csv'])
    assert {'calibration', 'fdr_scale', 'shuffle_rate_pct', 'decoy_fdr_pct'} <= set(table)
    assert table['fdr_scale'].eq(1).all()
    assert table['calibration'].str.contains('insufficient_never_calls').all()
    for key in ('ops_never_barcode_k', 'ops_never_barcode_seed', 'ops_never_barcode_sets'):
        assert key in G._ID_KEYS


def test_prefix_none_keeps_legacy_tables_even_after_a_new_schema_stamp(tmp_path):
    """The spec-owner decision keeps none tables byte-identical across settings eras."""
    import json
    from pathlib import Path
    from tests.test_ops_autotune_942 import _run
    from spacr.tabular import read_table

    old = _run(tmp_path, 'old-none', ops_prefix_decoy_k=4, ops_prefix_fdr_calibration='none')
    new = _run(tmp_path, 'new-none', ops_prefix_decoy_k=4, ops_prefix_fdr_calibration='none',
                spacr_settings_version=957)
    old_report = json.loads((Path(old['folder']) / 'fdr_CP9_W1.json').read_text())
    new_report = json.loads((Path(new['folder']) / 'fdr_CP9_W1.json').read_text())
    assert 'shuffle_call_rate_pct' not in old_report
    assert 'shuffle_call_rate_pct' not in new_report
    assert old_report == new_report
    old_rows = read_table(Path(old['folder']) / 'site_summary.csv', canonicalise=False, report=None)
    new_rows = read_table(Path(new['folder']) / 'site_summary.csv', canonicalise=False, report=None)
    assert old_rows['cells_real'].tolist() == new_rows['cells_real'].tolist()
    assert 'shuffle_called' not in new_rows and 'shuffle_called' not in old_rows
    for path in Path(old['folder']).iterdir():
        assert path.read_bytes() == (Path(new['folder']) / path.name).read_bytes(), path.name


@pytest.mark.parametrize('version,method', [(None, 'none'), (956, 'none'), (957, 'never'), (958, 'never')])
def test_saved_settings_schema_957_preserves_old_methods_and_enables_new_default(tmp_path, version, method):
    """The actual CLI loader folds old files before layering the new defaults."""
    import json
    from spacr.cli import resolve_module, resolve_settings
    from spacr.settings import DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION, SETTINGS_VERSION_KEY

    # 1014 is schema 957; the pilot also carries 1008's 958, so the current
    # schema is at least 957 and any file stamped 957 or later gets 'never'.
    assert SETTINGS_SCHEMA_VERSION >= 957
    assert DEFAULT_CHANGES[957] == {'ops_decode_fdr_calibration': 'none',
                                     'ops_prefix_fdr_calibration': 'none'}
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({} if version is None else {SETTINGS_VERSION_KEY: version}))
    actual = resolve_settings(resolve_module('ops'), str(path))
    assert actual['ops_decode_fdr_calibration'] == actual['ops_prefix_fdr_calibration'] == method
    assert actual['ops_fdr_scale'] == 1.268 and actual['ops_prefix_fdr_scale'] == 1
    assert actual['ops_never_barcode_k'] == 4 and actual['ops_never_barcode_seed'] == 1
    assert actual['ops_never_barcode_sets'] == 'n1_n2'
    fresh = resolve_settings(resolve_module('ops'), None)
    assert fresh['ops_decode_fdr_calibration'] == fresh['ops_prefix_fdr_calibration'] == 'never'


def test_old_run_journal_replay_restores_none_before_entering_the_pipeline(tmp_path, monkeypatch):
    """A real cli_repro replay reaches its entry with both historical calibration methods."""
    import json
    from spacr import cli_repro

    run = tmp_path / 'old-run'
    run.mkdir()
    (run / 'manifest.json').write_text(json.dumps({'app_key': 'ops'}))
    (run / 'settings.json').write_text(json.dumps({'spacr_settings_version': 956}))
    seen = {}

    def pipeline(app_key):
        """Capture the settings passed to the resolved replay entry."""
        assert app_key == 'ops'

        def run_ops(settings):
            """Record fully migrated historical settings without running an image pipeline."""
            seen.update(settings)

        return run_ops

    monkeypatch.setattr(cli_repro, '_resolve_pipeline', pipeline)
    assert cli_repro.main([str(run)]) == 0
    assert seen['ops_decode_fdr_calibration'] == seen['ops_prefix_fdr_calibration'] == 'none'


@pytest.mark.parametrize('bad', [{'ops_never_barcode_k': -1}, {'ops_never_barcode_k': 1.5},
                                  {'ops_never_barcode_seed': True}, {'ops_never_barcode_seed': -2},
                                  {'ops_never_barcode_sets': 'all'}])
def test_both_decode_validators_refuse_invalid_shared_controls(bad):
    """The full and prefix paths share exactly one validation of the new controls."""
    from spacr import ops_decode as D, ops_prefix as P

    with pytest.raises(ValueError, match='ops_never_barcode'):
        D.threshold_settings(bad, D.decode_rule({}))
    with pytest.raises(ValueError, match='ops_never_barcode'):
        P.prefix_settings({'ops_library': 'unused.csv', **bad})


def test_n2_only_cannot_discard_an_unmeasured_distance_one_library_error_term():
    """Skipping N1 keeps the entered scale when real guide pairs require its rate."""
    from spacr import ops_decode as D

    counts, grids, nuclei, sites = _never_well()
    grids = {**grids, 'opp1_L': np.ones_like(grids['opp2_L']),
              'real_calls_n2': counts[..., 0]}
    grids.pop('never1_calls', None)
    metadata = {'library_size': 200, 'never_sets': {'sets': 'n2'}}
    payload = D._never_payload(grids, metadata)
    result = D.calibrate_fdr_scale(counts, None, nuclei, sites, [('floor', [0.0, 0.2])],
                                    k=4, scale=8, method='never', never=payload,
                                    ci=False, fdr_max=10)
    assert result['fdr_scale'] == 8
    assert result['calibration']['status'] == 'insufficient_never_opportunities'
    assert result['calibration']['rho1'] is None
    assert 'n1' not in result['calibration']['real_call_change_pct']


def test_never_controls_are_alpha_but_calibration_methods_remain_visible(qtbot, tmp_path, monkeypatch):
    """Hiding experimental controls preserves their defaults and the active calibration method."""
    from PySide6.QtCore import QSettings
    from spacr import settings
    from spacr.ops_settings import OPS_CATEGORIES, OPS_DEFAULTS, OPS_TYPES
    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    keys = {'ops_never_barcode_k', 'ops_never_barcode_seed', 'ops_never_barcode_sets'}
    assert set(settings.ALPHA_FEATURES[1014]['settings']) == keys
    assert not {'ops_decode_fdr_calibration', 'ops_prefix_fdr_calibration'} & settings._alpha_names('settings')
    assert all(sum(key in rows for rows in OPS_CATEGORIES.values()) == 1 for key in keys)
    assert OPS_TYPES['ops_never_barcode_k'] is OPS_TYPES['ops_never_barcode_seed'] is int
    path = str(tmp_path / 'alpha.ini')
    monkeypatch.setattr(preferences, '_settings', lambda: QSettings(path, QSettings.Format.IniFormat))
    preferences._set_show_alpha_features(False)
    screen = AppScreen('ops')
    qtbot.addWidget(screen)
    try:
        for key in keys | {'ops_decode_fdr_calibration', 'ops_prefix_fdr_calibration'}:
            screen._open_the_heading_of(key)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(key) for key in keys)
        collected = screen._settings_model.collect()
        assert all(collected[key] == OPS_DEFAULTS[key] for key in keys)
        assert collected['ops_decode_fdr_calibration'] == collected['ops_prefix_fdr_calibration'] == 'never'
        from tests.test_ops_autotune_942 import _run

        result = _run(tmp_path, 'alpha-off-never',
                       ops_prefix_decoy_k=collected['ops_prefix_decoy_k'],
                       ops_prefix_threshold_tuning='none',
                       ops_prefix_fdr_calibration=collected['ops_prefix_fdr_calibration'],
                       **{key: collected[key] for key in keys})
        assert result['calibration']['CP9_W1']['calibration']['method'] == 'never'
        preferences._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert all(screen.setting_row_is_visible(key) for key in keys)
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()



@pytest.fixture
def _comparison_scale_plate(tmp_path):
    """Reuse the existing comparison plate fixture without importing production modules at collection."""
    from tests.test_ops_compare_calls_878 import plate

    return plate.__wrapped__(tmp_path)


def _current_well_scale_report(db, method='never', scale=3):
    """Write a current well report separately from the historical calibration sidecar."""
    import json
    from pathlib import Path
    from spacr import ops_decode as D

    folder=Path(D.results_folder(db))
    (folder/'decode_fdr_A1.json').write_text(json.dumps({'well':'A1','fdr_calibration':method,'fdr_scale':scale}))
    return folder


def test_reported_scale_uses_only_matching_current_reports(tmp_path):
    """Legacy methods and stale sidecars cannot inherit a prior never calibration."""
    import json
    from spacr import ops_decode as D

    db=str(tmp_path/'measurements.db')
    assert D._reported_fdr_scale(db,'A1',8,'never')==8
    folder=_current_well_scale_report(db,scale=3)
    assert D._reported_fdr_scale(db,'A1',8,'never')==3
    for method in ('none','shuffle','never+shuffle'):
        assert D._reported_fdr_scale(db,'A1',8,method)==8
    _current_well_scale_report(db,method='none',scale=1)
    (folder/'fdr_calibration_A1.json').write_text(json.dumps({'method':'never','scale':99}))
    assert D._reported_fdr_scale(db,'A1',8,'never')==8


def test_comparison_summary_pools_the_well_scales_in_the_same_fdr():
    """Two calibrated wells cannot be pooled at the last well scale."""
    import pandas as pd
    from spacr import ops_decode as D

    units=pd.DataFrame({'plate':['P']*4,'field':['A1','A1','A2','A2'],
                        'category':['pair_agree']*4,'subcategory':['']*4,'present':['SB']*4,
                        'S_call':['G','D','G','D'],'B_call':['G']*4,
                        'S_decoy':[False,True,False,True],'B_decoy':[False]*4,
                        'fdr_scale':[1,1,3,3]})
    categories,tables=D.compare_summary(units,['S','B'],decoy_k=4,fdr_scale=8)
    assert categories['S_decoy_FDR_pct'].eq(D.fdr(2,4,4,1)).all()
    assert tables[tables['table']=='S']['decoy_FDR_pct'].eq(50).all()
    legacy=D.compare_summary(units.drop(columns='fdr_scale'),['S','B'],decoy_k=4,fdr_scale=8)[1]
    assert legacy[legacy['table']=='S']['decoy_FDR_pct'].eq(200).all()


def test_comparison_phase_reads_the_report_and_preserves_legacy_scale(_comparison_scale_plate,monkeypatch):
    """The real comparison phase uses a saved applied scale without invoking calibration."""
    from spacr import ops_decode as D, ops_sbs as S
    from spacr.tabular import read_table, write_table

    db,path=_comparison_scale_plate
    folder=_current_well_scale_report(db,scale=2)
    file=folder/'calls_A1.parquet'
    calls=read_table(file,canonicalise=False,report=None)
    calls.loc[calls['object_id']==1,'mapped_guide']='DECOY'
    write_table(calls,file,canonicalise=False)
    monkeypatch.setattr(S,'decoy_library',lambda *args,**kwargs:['DECOY'])
    settings={'ops_compare_tables':{'X':path},'ops_decoy_library_k':4,'ops_fdr_scale':8,
              'ops_decode_fdr_calibration':'never'}
    result=D.run_compare_calls(db,'P',['A1'],settings,['GUIDE1','GUIDE2','GUIDE3'])
    units=read_table(result['paths']['units'],canonicalise=False,report=None)
    assert units['fdr_scale'].eq(2).all()
    table=read_table(result['paths']['tables'],canonicalise=False,report=None)
    spa=table[(table['plate']=='P') & (table['table']=='spaCR')].iloc[0]
    assert spa['decoy_FDR_pct']==round(D.fdr(spa['real_calls'],spa['decoy_calls'],4,2),3)
    legacy=D.run_compare_calls(db,'P',['A1'],{**settings,'ops_decode_fdr_calibration':'none'},['GUIDE1','GUIDE2','GUIDE3'])
    old=read_table(legacy['paths']['units'],canonicalise=False,report=None)
    assert 'fdr_scale' not in old
    table=read_table(legacy['paths']['tables'],canonicalise=False,report=None)
    spa=table[(table['plate']=='P') & (table['table']=='spaCR')].iloc[0]
    assert spa['decoy_FDR_pct']==round(D.fdr(spa['real_calls'],spa['decoy_calls'],4,8),3)


def test_live_options_read_the_saved_scale_without_recalibrating_the_crop(tmp_path,monkeypatch):
    """The preview keeps manual field thresholds while carrying the calibrated well scale."""
    from spacr import ops_decode as D, ops_live as L

    db=str(tmp_path/'measurements.db');_current_well_scale_report(db,scale=2)

    def refuse_calibration(*args,**kwargs):
        """Fail if the preview tries to estimate a scale from a crop."""
        raise AssertionError('A crop cannot calibrate the well')

    monkeypatch.setattr(D,'calibrate_fdr_scale',refuse_calibration)
    settings={'dst_root':str(tmp_path),'ops_decode_preset':'rule_a_s2',
              'ops_decode_fdr_calibration':'never','ops_fdr_scale':8}
    options=L._decode_options(settings,['ACGTACGT','TTTTGGGG'],well='A1')
    assert options['task']['rule_a']['fdr_scale']==2
    assert 'auto' not in options['task']['rule_a']
    old=L._decode_options({**settings,'ops_decode_fdr_calibration':'none'},['ACGTACGT','TTTTGGGG'],well='A1')
    assert old['task']['rule_a']['fdr_scale']==8
