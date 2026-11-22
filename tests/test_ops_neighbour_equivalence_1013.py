"""Exact old-versus-shared bleed equivalence on existing OPS evidence owners."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import struct
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from spacr import ops_decode as D, ops_prefix as P, ops_sbs as S
from spacr.tabular import read_table
from tests.test_ops_neighbour_discount_default_1013 import legacy

DISCOUNTS = (1.0, 0.5, 0.25, 0.0)


def _exact(value):
    """Preserve ordering, scalar float bits, array dtypes and every table column."""
    if isinstance(value, pd.DataFrame):
        return ('frame', tuple(value.columns), tuple(value.index),
                tuple((str(value[name].dtype), _exact(value[name].to_numpy())) for name in value))
    if isinstance(value, np.ndarray):
        data = _exact(value.tolist()) if value.dtype.hasobject else value.tobytes()
        return ('array', value.dtype.str, value.shape, data)
    if isinstance(value, np.generic):
        return ('scalar', value.dtype.str, value.tobytes())
    if isinstance(value, float):
        return ('float', struct.pack('!d', value))
    if isinstance(value, dict):
        return ('dict', tuple((_exact(key), _exact(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_exact(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return (type(value).__name__, tuple(sorted(_exact(item) for item in value)))
    return (type(value).__name__, value)


def _digest(value):
    """Hash the exact comparison representation for an auditable per-case receipt."""
    return hashlib.sha256(pickle.dumps(_exact(value), protocol=5)).hexdigest()


def _arrays(path, scratch):
    """Read a supplied NPZ using the existing lazy OPS field-cache container owner."""
    directory = scratch / hashlib.sha256(str(path).encode()).hexdigest()[:16]
    folder = directory / 'case/decode_cache'
    folder.mkdir(parents=True, exist_ok=True)
    link = folder / 'site0001.npz'
    if not link.exists():
        link.symlink_to(path.resolve())
    records = D._cached_fields(str(directory), 'equivalence', ['case'])
    assert len(records) == 1
    saved = records[0]['z']
    try:
        return {key: saved[key] for key in saved.files}
    finally:
        saved.close()


@contextmanager
def _owner(module, name, replacement):
    """Swap only the frozen flag adapter while all calling and weight owners stay shared."""
    with patch.object(module, name, replacement):
        yield


def _funk_equivalence(evidence, rule, library, decoys, crosstalk, legacy, case):
    """Compare every full-barcode vote and assembled field call for the prescribed factors."""
    matcher, decoy_set = D.matcher_for(library, decoys)
    rows = []
    for discount in DISCOUNTS:
        chosen = deepcopy(rule)
        chosen['read_weight'] = D._read_weighting({'ops_read_neighbour_discount': discount})
        arguments = {key: chosen[key] for key in ('threshold', 'footprint', 'max_distance', 'min_fraction', 'single_read_quality')}
        arguments.update(pixel_qualities=(chosen['pixel_quality'],), crosstalk=crosstalk, evidence_rows=True)
        results = []
        for owner in (legacy['_discount_neighbour_bleed'], S._discount_neighbour_bleed):
            with _owner(S, '_discount_neighbour_bleed', owner):
                votes = D._field_votes(evidence, deepcopy(chosen), library, matcher, **arguments)
                pixels = votes['pixel'][chosen['pixel_quality']] if chosen['pixels'] else {}
                calls = D._assemble_calls(votes['primary'], pixels, votes, decoy_set)
                results.append((votes, calls))
        before, after = results
        assert _exact(before[0]) == _exact(after[0]), (case, discount, 'votes')
        assert _exact(before[1]) == _exact(after[1]), (case, discount, 'calls')
        rows.append({'dataset': case[0], 'site': int(case[1]), 'discount': discount,
                     'reads': after[0]['reads'], 'owned_reads': after[0]['reads_owned'],
                     'vote_cells': len(after[0]['tallies']['votes']), 'calls': len(after[1]['calls']),
                     'vote_sha256': _digest(after[0]), 'call_sha256': _digest(after[1]),
                     'votes_equal': True, 'calls_equal': True})
    return rows


def _prefix_equivalence(reads, pairs, legacy, case):
    """Compare all barcode weights and weighted-majority calls with distance weighting off."""
    matched = reads[(reads['cell'] > 0) & reads['status'].isin(('exact', 'corrected'))]
    neighbours = pd.DataFrame({'cell': pairs[:, 0], 'neighbour': pairs[:, 1]})
    rows = []
    for discount in DISCOUNTS:
        results = []
        for owner in (legacy['_barcode_groups'], P._barcode_groups):
            with _owner(P, '_barcode_groups', owner):
                groups = P._barcode_groups(matched['cell'].to_numpy(np.int64),
                                            matched['prefix'].to_numpy(), pairs if discount != 1 else None)
                term, used = P._weight_terms('off', True, 0.0)
                weights = P._read_weights(matched['quality'].to_numpy(float), None,
                    quality_term=term, decay_px=used, interior_px=0.0, min_weight=0.05)
                votes = P._discounted(np.bincount(groups['group'], weights=weights,
                                                 minlength=len(groups['n'])), groups, discount)
                calls = P.prefix_cell_calls(reads, keys=['cell'], method='weighted_majority',
                    distance_weight='off', weighted_quality=True,
                    neighbour_discount=discount, neighbours=neighbours)
                results.append(({'groups': groups, 'votes': votes}, calls))
        before, after = results
        assert _exact(before[0]) == _exact(after[0]), (case, discount, 'votes')
        assert _exact(before[1]) == _exact(after[1]), (case, discount, 'calls')
        rows.append({'dataset': case[0], 'site': int(case[1]), 'discount': discount,
                     'reads': len(reads), 'matched_reads': len(matched), 'vote_groups': len(after[0]['votes']),
                     'cells': len(after[1]), 'calls': int(after[1]['quality'].isin(('perfect', 'great')).sum()),
                     'vote_sha256': _digest(after[0]), 'call_sha256': _digest(after[1]),
                     'votes_equal': True, 'calls_equal': True, 'distance_weight': 'off'})
    return rows


def _write_receipt(rows, inputs, output):
    """Require unchanged input files and persist complete equality counts and source hashes."""
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    receipt = {'rows': rows, 'inputs_sha256': {str(path): digest for path, digest in inputs.items()},
               'sources_sha256': {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                  for name, module in (('ops_sbs', S), ('ops_prefix', P), ('ops_decode', D))}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def test_builtin_a1_core_equivalence(legacy, tmp_path):
    """Compare the four cached published A1 fields before accepting the external data swap."""
    receipt_path = os.environ.get('SPACR_1013_BUILTIN')
    if not receipt_path:
        pytest.skip('set SPACR_1013_BUILTIN to the published A1 generation receipt')
    context = json.loads(Path(receipt_path).read_text())
    library_path = Path(context['settings']['ops_library'])
    library = tuple(read_table(library_path, canonicalise=False, report=None)['prefix'].astype(str))
    rule = context['decode']['decode_rules']
    decoys = tuple(sorted(S.decoy_library(library, k=rule['decoy_k'])))
    fields = D._cached_fields(context['destination'], context['plate'], [context['well']])
    assert len(fields) == 4
    inputs = {library_path: hashlib.sha256(library_path.read_bytes()).hexdigest()}
    rows = []
    for field in fields:
        saved = field['z']
        try:
            evidence = {key: saved[key] for key in saved.files}
        finally:
            saved.close()
        assert all(key in evidence for key in D.CACHE_NUCLEUS_KEYS)
        path = Path(context['destination']) / context['well'] / 'decode_cache' / f"site{field['site']:04d}.npz"
        inputs[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows += _funk_equivalence(evidence, rule, library, decoys, context['decode']['crosstalk'], legacy,
                                  ('builtin_A1', field['site']))
    output = Path(os.environ.get('SPACR_1013_RECEIPTS', str(tmp_path))) / 'builtin_equivalence.json'
    _write_receipt(rows, inputs, output)
    assert len(rows) == 16


def test_external_funk_and_periscope_core_equivalence(legacy, tmp_path):
    """Compare every supplied field and site using only the approved external input root."""
    value = os.environ.get('SPACR_1013_CACHE')
    if not value:
        pytest.skip('set SPACR_1013_CACHE to the approved 20-field and 10-site input package')
    root = Path(value).resolve()
    context = json.loads((root / 'funk_A1/ops_report.json').read_text())['decode']
    rule = context['decode_rules']
    library = tuple(read_table(root / 'funk_A1/pool10_prefixes.csv', canonicalise=False, report=None)['prefix'].astype(str))
    decoys = tuple(sorted(S.decoy_library(library, k=rule['decoy_k'])))
    paths = [path for path in root.rglob('*') if path.is_file()]
    assert all(path.resolve().is_relative_to(root) for path in paths)
    inputs = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    fields = sorted((root / 'funk_A1/field_cache').glob('site*.npz'))
    sites = sorted((root / 'periscope_W1/candidates').glob('site*_log.npz'))
    assert len(fields) == 20 and len(sites) == 10
    rows = []
    for path in fields:
        site = int(path.stem[4:])
        evidence = _arrays(path, tmp_path)
        nuclei = read_table(root / f'funk_A1/nuclei/site{site:04d}_nuclei.csv', canonicalise=False, report=None)
        neighbours = S._disc_neighbours(nuclei['object_id'].to_numpy(np.int64),
            nuclei[['centroid_y', 'centroid_x']].to_numpy(float), nuclei['area'].to_numpy(float), rule['footprint'])
        assert neighbours
        rows += _funk_equivalence(evidence, {**rule, 'neighbours': neighbours}, library, decoys,
                                  context['crosstalk'], legacy, ('funk_A1', site))
    settings = json.loads((root / 'periscope_W1/pm_988.json').read_text())
    prefix = P.spacer_prefix_library(root / 'library/periscope_Barcodes.csv',
        sequence_col=settings['ops_library_sequence_col'], gene_col=settings['ops_library_gene_col'],
        prefix_length=settings['ops_prefix_length'], n_cycles=settings['ops_sbs_cycles'],
        cycle_mask=settings['ops_cycle_mask'], collision=settings['ops_prefix_collision'])
    expected = set(read_table(root / 'library/LIB_peri_full.csv', canonicalise=False, report=None)['barcode'].astype(str))
    assert set(prefix.keys) == expected
    matching = P.prefix_decoy_library(prefix, k=settings['ops_prefix_decoy_k'], seed=settings['ops_prefix_decoy_seed'])
    for path in sites:
        site = int(path.stem[4:-4])
        saved = _arrays(path, tmp_path)
        labels = P.site_labels(str(root / 'periscope_W1/cells_npz/site{site}_cells.npz'),
                               plate='CP186A', well='Well1', well_id='W1', site=site)
        peaks = saved['peaks']
        cell = labels[peaks[:, 0], peaks[:, 1]].astype(np.int64)
        pairs = P._touching_cells(labels)
        assert len(pairs)
        reads = P._site_tuning_reads((peaks, saved['strength'], saved['values'], cell), matching, {},
            bases=('G', 'T', 'A', 'C'), threshold=settings['ops_read_threshold'],
            crosstalk=settings['ops_prefix_crosstalk'], metric=settings['ops_prefix_metric'],
            max_distance=settings['ops_prefix_max_distance'], read_mode='log')
        rows += _prefix_equivalence(reads, pairs, legacy, ('periscope_W1_log', site))
    output = Path(os.environ.get('SPACR_1013_RECEIPTS', str(tmp_path))) / 'external_equivalence.json'
    _write_receipt(rows, inputs, output)
    assert len(rows) == 120 and all(row['votes_equal'] and row['calls_equal'] for row in rows)
