"""Actual-source negatives for the immutable caption pin arrival adapter."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import build_i18n_catalogs as builder
from i18n_caption_arrivals import load_arrival_contract, pinned_external_sources

@pytest.fixture(scope='module')
def state():
    declarations = {}
    for node in ast.parse((ROOT/'tests/qt/test_i18n_caption_ratchet.py').read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('EXTERNAL_SOURCE_COUNTS','EXTERNAL_SOURCE_KEY_SHA256'):
                    declarations[target.id] = ast.literal_eval(node.value)
    canonical = builder.canonical_sources()
    external = {table:canonical[name] for table,name in (
        ('SETTING_LABELS','setting_labels'),('SETTING_TOOLTIPS','setting_tooltips'),
        ('CATEGORY_HELP','categories'),('UI','ui'),('MODULE_SUMMARIES','module_summaries'))}
    return external, declarations, load_arrival_contract()

def project(external,declarations,contract=None):
    return pinned_external_sources(external,
        original_counts=declarations['EXTERNAL_SOURCE_COUNTS'],
        original_identity_sha256=declarations['EXTERNAL_SOURCE_KEY_SHA256'],contract=contract)

def test_actual_source_and_reviews_reproduce_original_catalog_pin(state):
    external,declarations,contract = state
    pinned = project(external,declarations)
    assert {table:len(rows) for table,rows in pinned.items()}==declarations['EXTERNAL_SOURCE_COUNTS']
    digest = hashlib.sha256('\0'.join(f'{table}\0{key}' for table,key in sorted(
        (table,key) for table,rows in pinned.items() for key in rows)).encode()).hexdigest()
    assert digest==declarations['EXTERNAL_SOURCE_KEY_SHA256']
    from spacr.qt.i18n_catalogs import en
    assert set(external['UI'])==set(en.UI_SOURCES)
    assert sum(map(len,external.values()))>sum(map(len,pinned.values()))

@pytest.mark.parametrize('change',('unknown_arrival','unknown_retirement','surviving_body_drift','admitted_body_drift'))
def test_actual_source_rejects_every_unrecorded_arrival_retirement_or_body_change(state,change):
    external,declarations,contract = state
    candidate = copy.deepcopy(external)
    if change=='unknown_arrival':
        candidate['UI'] = (*candidate['UI'],'A future unreviewed caption cannot inherit this proof.')
    elif change=='admitted_body_drift':
        entry = next(r for r in contract['added'] if r['table']=='SETTING_LABELS')
        candidate['SETTING_LABELS'][entry['key']] += ' unreviewed'
    else:
        entry = next(r for r in contract['baseline'] if r['table']=='SETTING_LABELS'
                     and r['key'] in candidate['SETTING_LABELS'])
        if change=='unknown_retirement':
            candidate['SETTING_LABELS'].pop(entry['key'])
        else:
            candidate['SETTING_LABELS'][entry['key']] += ' unreviewed'
    with pytest.raises(AssertionError,match='unreviewed caption'):
        project(candidate,declarations)

@pytest.mark.parametrize('change',('target_digest_drift','missing_review_record','file_hash_drift','record_hash_drift'))
def test_actual_source_rejects_unbound_or_drifted_review_evidence(state,change):
    external,declarations,contract = state
    candidate = copy.deepcopy(contract)
    entry = candidate['added'][0]
    if change=='target_digest_drift':
        entry['reviewed_evidence']['sv']['translation_sha256'] = '0'*64
        message = 'caption arrival target drift'
    elif change=='missing_review_record':
        entry['reviewed_evidence']['sv']['matching_review_records'] = []
        message = 'caption arrival lacks explicit reviewed record'
    elif change=='file_hash_drift':
        entry['reviewed_evidence']['sv']['matching_review_records'][0]['file_sha256'] = '0'*64
        message = 'caption arrival reviewed file bytes drift'
    else:
        entry['reviewed_evidence']['sv']['matching_review_records'][0]['record_sha256'] = '0'*64
        message = 'caption arrival reviewed record drift'
    with pytest.raises(AssertionError,match=message):
        project(external,declarations,candidate)


def test_original_historical_source_body_anchor_cannot_be_rewritten(state):
    external,declarations,contract = state
    candidate = copy.deepcopy(contract)
    entry = next(r for r in candidate['baseline'] if r['table']=='SETTING_LABELS')
    entry['source'] += ' silently repinned'
    entry['source_sha256'] = hashlib.sha256(entry['source'].encode()).hexdigest()
    with pytest.raises(AssertionError,match='historical caption bodies changed'):
        project(external,declarations,candidate)


def test_proof_file_bytes_are_checked_fresh_on_every_validation_call(state,monkeypatch):
    external,declarations,contract = state
    project(external,declarations)
    proof=contract['added'][0]['reviewed_evidence']['sv']['matching_review_records'][0]
    path=ROOT/proof['path']
    original=Path.read_bytes
    def changed(candidate):
        raw=original(candidate)
        return raw+b'\n' if candidate==path else raw
    monkeypatch.setattr(Path,'read_bytes',changed)
    with pytest.raises(AssertionError,match='caption arrival reviewed file bytes drift'):
        project(external,declarations)


def test_duplicate_source_bound_review_identity_is_rejected(state,monkeypatch):
    external,declarations,contract = state
    candidate=copy.deepcopy(contract)
    proof=candidate['added'][0]['reviewed_evidence']['sv']['matching_review_records'][0]
    path=ROOT/proof['path']
    original=Path.read_bytes
    payload=json.loads(original(path))
    payload['records'].append(copy.deepcopy(payload['records'][0]))
    raw=json.dumps(payload,ensure_ascii=False).encode()
    proof['file_sha256']=hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(Path,'read_bytes',lambda p:raw if p==path else original(p))
    with pytest.raises(AssertionError,match='duplicate caption reviewed record'):
        project(external,declarations,candidate)
