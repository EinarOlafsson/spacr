"""Item 1013 shares neighbour-bleed flags while preserving the existing vote arithmetic."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import pytest

from spacr import ops_prefix as P, ops_sbs as S


@pytest.fixture
def legacy():
    """Load the hash-bound pre-1013 flag owners from the announced nightly source."""
    path = Path(__file__).parent / 'data/release_contracts/1013_neighbour_bleed_legacy_299179d953.json'
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'da17e65b55637da31a10930997cf687ed1277cce8fb326386c47385995111b5e'
    record = json.loads(raw)
    assert record['base'] == '299179d9530ac618d82f11cc823f3ad8abd1c001'
    namespace = {'np': np, 'pd': pd, 'Any': Any, 'Dict': Dict,
                 'Mapping': Mapping, 'Optional': Optional}
    for name, source in record['functions'].items():
        exec(compile(source, str(path) + ':' + name, 'exec'), namespace)
    return namespace


def _grouped_votes(seed):
    """Build distinct groups with ties, corrected reads, directed edges and absent neighbours."""
    rng = np.random.default_rng(seed)
    cells, barcodes, counts = [], [], []
    votes, exact, corrected = {}, {}, {}
    for cell in rng.permutation(np.arange(11, 41, 2)):
        names = rng.choice(np.array(['A', 'B', 'C', 'D', 'E']), rng.integers(1, 6), replace=False)
        values = rng.integers(1, 10, len(names))
        votes[int(cell)], exact[int(cell)], corrected[int(cell)] = {}, {}, {}
        for name, count in zip(names, values):
            cells.append(int(cell)); barcodes.append(str(name)); counts.append(int(count))
            correction = int(rng.integers(0, count + 1))
            votes[int(cell)][str(name)] = float(rng.integers(1, 12)) / 7
            exact[int(cell)][str(name)] = int(count) - correction
            corrected[int(cell)][str(name)] = correction
    owners = np.array(list(votes))
    pairs = np.array([(int(owner), int(other)) for owner in owners
                      for other in rng.choice(np.r_[owners, 999], 4, replace=False)], np.int64)
    neighbours = {int(owner): set(pairs[pairs[:, 0] == owner, 1]) for owner in owners}
    return np.array(cells), np.array(barcodes), np.array(counts), pairs, votes, exact, corrected, neighbours


@pytest.mark.parametrize('discount', [1.0, 0.5, 0.25, 0.0])
def test_full_barcode_votes_equal_the_frozen_owner_for_all_four_discounts(legacy, discount):
    """Compare every in-place vote value and insertion order against the original source."""
    for seed in range(12):
        cells, barcodes, counts, pairs, votes, exact, corrected, neighbours = _grouped_votes(seed)
        before, after = deepcopy(votes), deepcopy(votes)
        legacy['_discount_neighbour_bleed'](before, exact, corrected, neighbours, discount)
        S._discount_neighbour_bleed(after, exact, corrected, neighbours, discount)
        assert list(before) == list(after)
        assert before == after
        for cell in before:
            assert list(before[cell]) == list(after[cell])
            assert np.array(list(before[cell].values())).tobytes() == np.array(list(after[cell].values())).tobytes()


def test_prefix_group_arrays_equal_the_frozen_owner_with_the_shared_flags(legacy):
    """Pin every prefix group array rather than only the final called barcode."""
    for seed in range(12):
        cells, barcodes, counts, pairs, votes, exact, corrected, neighbours = _grouped_votes(seed)
        per_read_cells = np.repeat(cells, counts)
        per_read_barcodes = np.repeat(barcodes, counts)
        for edges in (None, np.empty((0, 2), np.int64), pairs):
            expected = legacy['_barcode_groups'](per_read_cells, per_read_barcodes, edges)
            observed = P._barcode_groups(per_read_cells, per_read_barcodes, edges)
            assert set(expected) == set(observed)
            for key in expected:
                np.testing.assert_array_equal(observed[key], expected[key])
                assert observed[key].dtype == expected[key].dtype


def test_shared_flags_preserve_ties_directed_edges_and_large_cell_identifiers():
    """Keep all tied top barcodes while avoiding arithmetic overflow in cell identifiers."""
    large = 2 ** 62
    cells = np.array([large, large, large + 1, large + 1, large + 2], np.int64)
    barcodes = np.array(['A', 'B', 'B', 'C', 'B'])
    counts = np.array([4, 1, 3, 3, 2])
    copies = [value.copy() for value in (cells, barcodes, counts)]
    flags = S.neighbour_bleed_flags(cells, barcodes, counts, [[large, large + 1]])
    np.testing.assert_array_equal(flags, [False, True, False, False, False])
    np.testing.assert_array_equal(S.neighbour_bleed_flags(cells, barcodes, counts,
                                                       [[large + 1, large]]), np.zeros(5, bool))
    for value, before in zip((cells, barcodes, counts), copies):
        np.testing.assert_array_equal(value, before)


def test_both_adapters_call_the_same_shared_flag_owner(monkeypatch):
    """Trace the common owner through both adapters without duplicating either decision rule."""
    calls = []
    owner = S.neighbour_bleed_flags

    def trace(*args):
        """Retain arguments and delegate all flag decisions to the shared numerical owner."""
        calls.append(args)
        return owner(*args)

    monkeypatch.setattr(S, 'neighbour_bleed_flags', trace)
    cells, barcodes, counts, pairs, votes, exact, corrected, neighbours = _grouped_votes(18)
    S._discount_neighbour_bleed(votes, exact, corrected, neighbours, 0.25)
    P._barcode_groups(np.repeat(cells, counts), np.repeat(barcodes, counts), pairs)
    assert len(calls) == 2


def test_shared_flags_handle_empty_groups_and_missing_neighbours():
    """Empty and neighbour-free inputs yield boolean flags in the original group order."""
    result = S.neighbour_bleed_flags([], [], [], [[1, 2]])
    assert result.shape == (0,) and result.dtype == bool
    for pairs in (None, [], [[99, 100]]):
        np.testing.assert_array_equal(S.neighbour_bleed_flags([1, 1], ['A', 'B'], [3, 1], pairs), [False, False])


@pytest.mark.parametrize('cells,barcodes,counts,pairs', [
    ([[1]], ['A'], [1], None), ([1], ['A', 'B'], [1], None),
    ([1], ['A'], [1, 2], None), ([1], ['A'], [1], [1, 2])])
def test_shared_flags_reject_inconsistent_group_or_pair_shapes(cells, barcodes, counts, pairs):
    """Reject malformed input before pandas can silently align inconsistent arrays."""
    with pytest.raises(ValueError, match='neighbour bleed'):
        S.neighbour_bleed_flags(cells, barcodes, counts, pairs)


def test_new_default_preserves_every_old_settings_era_without_rewriting_files(tmp_path):
    """Schema 959 keeps 1.0 for unstamped and 957/958 files while current files use 0.25."""
    from spacr.cli import resolve_module, resolve_settings
    from spacr import ops_decode as D
    from spacr.ops_settings import OPS_DEFAULTS
    from spacr.settings import DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION, ALPHA_FEATURES

    assert SETTINGS_SCHEMA_VERSION >= 959
    assert DEFAULT_CHANGES[959] == {'ops_read_neighbour_discount': 1.0}
    assert OPS_DEFAULTS['ops_read_neighbour_discount'] == D._DEFAULTS['ops_read_neighbour_discount'] == 0.25
    assert OPS_DEFAULTS['ops_prefix_bleed_in_recipe'] is False
    assert ALPHA_FEATURES[1013]['settings'] == ('ops_prefix_bleed_in_recipe',)
    assert 'ops_read_neighbour_discount' not in ALPHA_FEATURES[988]['settings']
    for era in (None, 955, 957, 958, 959):
        data = {} if era is None else {'spacr_settings_version': era}
        path = tmp_path / ('settings_' + str(era) + '.json')
        path.write_text(json.dumps(data))
        original = path.read_bytes()
        loaded = resolve_settings(resolve_module("ops"), str(path))
        assert D._number(loaded, 'ops_read_neighbour_discount') == (0.25 if era == 959 else 1.0)
        assert path.read_bytes() == original


def test_rules_off_skip_only_the_new_default_and_log_once(capsys):
    """A default discount cannot activate disabled rules or excuse an explicit other weight."""
    from spacr import ops_decode as D

    settings = {'ops_decode_preset': 'none'}
    first = D.decode_rule(settings)
    second = D.decode_rule(settings)
    assert not first['active'] and 'read_weight' not in first
    assert first['ops_read_neighbour_discount_applied']['full_barcode'] is False
    assert second['ops_read_neighbour_discount_applied']['full_barcode'] is False
    assert capsys.readouterr().out.count('reads are not discounted') == 1
    with pytest.raises(ValueError, match='decode rules'):
        D.decode_rule({'ops_read_neighbour_discount': 0.5})
    with pytest.raises(ValueError, match='decode rules'):
        D.decode_rule({'ops_read_distance_weight': 'decay'})


def test_positionless_cache_default_matches_discount_off_and_logs_once(capsys):
    """Replay old evidence unchanged with the default while retaining explicit-discount errors."""
    from spacr import ops_decode as D
    from tests.test_ops_field_cache_nucleus_positions import _evidence, _calls, _rule

    evidence = _evidence()
    rule = _rule(0.25)
    baseline = _calls(evidence, _rule(1.0))
    assert _calls(evidence, rule) == baseline
    assert _calls(evidence, rule) == baseline
    assert rule['read_weight']['neighbour_discount'] == 0.25
    assert rule['ops_read_neighbour_discount_applied']['cache_replay'] is False
    assert capsys.readouterr().out.count('reads are not discounted') == 1
    with pytest.raises(ValueError, match='written before'):
        _calls(evidence, _rule(0.5))


@pytest.mark.parametrize('method', ['spacr', 'recipe_simple'])
def test_prefix_default_no_op_preserves_all_decode_files(tmp_path, method, capsys):
    """The unavailable default leaves every output byte equal to explicit discount off."""
    from tests.test_ops_read_mode_952 import _prefix, _same_files

    default = _prefix(tmp_path, 'default', ops_prefix_quality_method=method)
    off = _prefix(tmp_path, 'off', ops_prefix_quality_method=method, ops_read_neighbour_discount=1.0)
    _same_files(default['folder'], off['folder'])
    assert capsys.readouterr().out.count('reads are not discounted') == 1
    with pytest.raises(ValueError, match='weighted majority'):
        P.prefix_settings({'ops_library': 'x.csv', 'ops_prefix_quality_method': method,
                           'ops_read_neighbour_discount': 0.5})


def test_application_flags_are_published_to_the_existing_run_journal(tmp_path, monkeypatch, capsys):
    """Write actual false flags and one notice through the existing journal publisher."""
    from spacr import ops_decode as D, run_journal as J

    run = J.Run(app_key='ops', settings={'ops_read_neighbour_discount': 0.25}, dir=tmp_path)
    monkeypatch.setattr(J._RUN_LOCAL, 'active', run, raising=False)
    settings = {'ops_decode_preset': 'none'}
    D.decode_rule(settings)
    D.decode_rule(dict(settings))
    saved = json.loads((tmp_path / 'settings.json').read_text())
    assert saved['ops_read_neighbour_discount_applied']['full_barcode'] is False
    assert capsys.readouterr().out.count('reads are not discounted') == 1


def _recipe_bleed_reads():
    """Keep a three-read minority above the existing floor so only bleed rescues it."""
    from tests.test_ops_read_weights_988 import _read

    return pd.DataFrame(_read(1, 'A', 4) + _read(1, 'B', 3) + _read(2, 'B', 5)
                        + _read(3, 'X', 3, gene=P.DECOY_GENE))


def _recipe_neighbours():
    """Connect only the minority-bearing cell and the cell with the matching top barcode."""
    return pd.DataFrame({'plate': 'P', 'well': 'W', 'site': 1,
                         'cell': [1, 2], 'neighbour': [2, 1]})


@pytest.mark.parametrize('discount', [0.0, 0.25, 0.5])
def test_opt_in_recipe_bleed_uses_the_existing_minority_filter(discount):
    """Any discount below one removes the flagged barcode in the categorical recipe rule."""
    reads = _recipe_bleed_reads()
    original = reads.copy(deep=True)
    plain = P.prefix_cell_calls(reads, method='recipe_simple', minority_min_reads=3)
    assert plain.loc[0, 'quality'] not in ('perfect', 'great')
    result = P.prefix_cell_calls(reads, method='recipe_simple', minority_min_reads=3,
        neighbour_discount=discount, neighbours=_recipe_neighbours(), bleed_in_recipe=True)
    assert result.loc[0, 'prefix'] == 'A' and result.loc[0, 'quality'] == 'perfect'
    assert result.loc[0, 'minority_ignored'] == 3
    assert bool(result.loc[0, 'minority_rescued'])
    assert result.loc[0, 'matched'] == 7 and result.loc[0, 'top_reads'] == 4
    assert result.loc[1, 'prefix'] == 'B' and result.loc[1, 'minority_ignored'] == 0
    pd.testing.assert_frame_equal(reads, original)


def test_recipe_switch_off_and_discount_off_preserve_all_cell_values():
    """The alpha gate or discount value one keeps all existing outputs and column layouts."""
    reads = _recipe_bleed_reads()
    plain = P.prefix_cell_calls(reads, method='recipe_simple', minority_min_reads=3)
    gate_off = P.prefix_cell_calls(reads, method='recipe_simple', minority_min_reads=3,
        neighbour_discount=0.25, neighbours=_recipe_neighbours(), bleed_in_recipe=False)
    discount_off = P.prefix_cell_calls(reads, method='recipe_simple', minority_min_reads=3,
        neighbour_discount=1.0, bleed_in_recipe=True)
    pd.testing.assert_frame_equal(gate_off, plain)
    pd.testing.assert_frame_equal(discount_off, plain)


def test_recipe_bleed_never_removes_a_tied_top_barcode():
    """A top-count tie stays ambiguous even if one barcode is the top of a neighbour."""
    from tests.test_ops_read_weights_988 import _read

    reads = pd.DataFrame(_read(1, 'A', 3) + _read(1, 'B', 3) + _read(2, 'B', 5))
    result = P.prefix_cell_calls(reads, method='recipe_simple', minority_min_reads=3,
        neighbour_discount=0.25, neighbours=_recipe_neighbours(), bleed_in_recipe=True)
    assert result.loc[0, 'minority_ignored'] == 0
    assert result.loc[0, 'quality'] not in ('perfect', 'great')
    assert not bool(result.loc[0, 'minority_rescued'])


def test_recipe_call_counts_and_control_labels_use_the_same_selected_reads():
    """Production and control frames pass through the same call owner and bleed gate."""
    chosen = P.prefix_settings({'ops_library': 'x.csv', 'ops_prefix_quality_method': 'recipe_simple',
        'ops_prefix_minority_min_reads': 3, 'ops_prefix_bleed_in_recipe': True})
    plan = P._weight_plan(chosen, [])
    assert plan['recipe_bleed'] and plan['neighbours'] and not plan['gaps']
    assert not plan['weighted']
    frame = _recipe_bleed_reads().drop(columns=['plate', 'well', 'site'])
    pairs = np.array([[1, 2], [2, 1]], np.int64)
    observed = P._setting_counts(frame, 'recipe_simple', 3, plan, chosen, pairs)
    control = frame.copy()
    control.loc[control['prefix'] == 'A', 'gene'] = P.DECOY_GENE
    control.loc[control['prefix'] == 'X', 'gene'] = 'GX'
    shuffled = P._setting_counts(control, 'recipe_simple', 3, plan, chosen, pairs)
    assert observed == (2, 1) and shuffled == (2, 1)
    off = {**chosen, 'ops_prefix_bleed_in_recipe': False}
    off_plan = P._weight_plan(off, [])
    assert not off_plan['recipe_bleed'] and not off_plan['neighbours']
    assert P._setting_counts(frame, 'recipe_simple', 3, off_plan, off, pairs) == (1, 1)


def test_opt_in_recipe_bleed_requires_actual_neighbour_information():
    """An enabled recipe filter cannot silently claim an application without geometry."""
    with pytest.raises(ValueError, match='touching-cell information'):
        P.prefix_cell_calls(_recipe_bleed_reads(), method='recipe_simple',
            neighbour_discount=0.25, bleed_in_recipe=True)


def test_prefix_run_exercises_recipe_flags_and_records_the_opt_in(tmp_path, monkeypatch):
    """The settings entry point reaches the shared owner without requiring nucleus distances."""
    from tests.test_ops_read_mode_952 import _prefix

    calls = []
    owner = S.neighbour_bleed_flags

    def trace(*args):
        """Observe genuine group inputs while preserving all shared flag calculations."""
        calls.append(args)
        return owner(*args)

    monkeypatch.setattr(S, 'neighbour_bleed_flags', trace)
    result = _prefix(tmp_path, 'recipe', ops_prefix_quality_method='recipe_simple',
                     ops_prefix_bleed_in_recipe=True, ops_prefix_minority_min_reads=3)
    assert calls
    saved = json.loads((Path(result['folder']) / 'quality_classes.json').read_text())
    assert saved['settings']['ops_prefix_bleed_in_recipe'] is True
    assert saved['settings']['ops_read_neighbour_discount'] == 0.25
    assert result['called_cells'] >= 1


def test_combinatorial_early_return_records_default_as_unapplied_once(tmp_path, monkeypatch):
    """The existing combinatorial decoder retains its output and reports the inactive default."""
    from spacr import ops_insitu as I

    sentinel = {'existing_combinatorial_output': True}
    seen = []

    def decode(settings, *, report):
        """Retain the existing early-return output without requiring unrelated image inputs."""
        seen.append(settings)
        return sentinel

    monkeypatch.setattr(I, '_run_combinatorial_decode', decode)
    messages = []
    settings = {'dst_root': str(tmp_path), 'ops_library': str(tmp_path / 'library.csv'),
                'ops_barcode_layout': 'combinatorial',
                'ops_barcode_layout_file': str(tmp_path / 'segments.csv')}
    assert P.run_prefix_decode(settings, report=messages.append) is sentinel
    assert P.run_prefix_decode(settings, report=messages.append) is sentinel
    assert settings['ops_read_neighbour_discount_applied'] == {'prefix_combinatorial': False}
    assert len(messages) == 1 and 'combinatorial' in messages[0]
    off = {**settings, 'ops_read_neighbour_discount': 1.0}
    assert P.run_prefix_decode(off, report=messages.append) is sentinel
    assert len(messages) == 1 and len(seen) == 3


def test_runtime_extractor_admits_the_notice_and_alpha_control(monkeypatch):
    """The shipped source extractor sees the notice identities and the actual generated control text."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    path = root / 'tools/build_i18n_catalogs.py'
    spec = importlib.util.spec_from_file_location('_ops_1013_runtime_sources', path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    sources = builder.canonical_sources()
    expected = {'rules off', 'cache replay without nucleus positions',
                'ops_read_neighbour_discount ({discount}) does not act under {method}; reads are not discounted'}
    assert expected <= set(sources['ui'])
    assert sources['setting_labels']['ops.ops_prefix_bleed_in_recipe'] == 'Prefix bleed in recipe'
    assert 'Default False. Alpha.' in sources['setting_tooltips']['ops_prefix_bleed_in_recipe']
    assert sources['setting_tooltips']['ops_read_neighbour_discount'].endswith('Default 0.25.')


@pytest.mark.parametrize('seed', [{}, {'x': 1}, {'plain': 'value'}])
def test_generic_resolver_keeps_unrelated_mappings_exact(seed):
    """A preset resolver retains plain mappings without adding OPS notice metadata."""
    from spacr import ops_decode as D

    before = deepcopy(seed)
    result = D.resolve_settings(seed)
    assert result == before and seed == before
    assert result is not seed


def test_ops_resolver_shares_one_notice_across_well_copies(capsys):
    """Actual OPS settings retain one notice context across repeated shallow resolutions."""
    from spacr import ops_decode as D

    settings = {'ops_decode_preset': 'none'}
    first = D.resolve_settings(settings)
    second = D.resolve_settings(settings)
    assert first['_meta_neighbour_discount'] is second['_meta_neighbour_discount']
    assert D.decode_rule(first)['ops_read_neighbour_discount_applied']['full_barcode'] is False
    assert D.decode_rule(second)['ops_read_neighbour_discount_applied']['full_barcode'] is False
    assert capsys.readouterr().out.count('reads are not discounted') == 1
