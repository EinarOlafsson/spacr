"""Validate explicit source/review arrivals before projecting the original pin.

The old identity pin belongs to the immutable English catalog at 3d9f02fe,
which differs from that commit's source by one stale installation hint. This
contract names that mixed baseline candidly. It never normalizes source text
or updates the original pin. The full actual inventory still owns manifest
equality and compact-layer exclusivity in the calling ratchet.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests' / 'data' / 'release_contracts' / 'runtime_caption_arrivals_2026_10_08.json.gz'
# Independently reproduced from the immutable pinned English catalog's
# SOURCE_HASHES; additions never alter this historical source-body anchor.
PINNED_BASELINE_SOURCE_HASH_SHA256 = '26a77519ac43c576346359e669772f630966a71f56b33a071d7d8084f35d8cd7'
TABLE_NAMES = {'SETTING_LABELS':'setting_labels','SETTING_TOOLTIPS':'setting_tooltips',
               'CATEGORY_HELP':'categories','UI':'ui','MODULE_SUMMARIES':'module_summaries'}
LANGUAGES = ('sv','de','es','zh_CN','pt','hi','ko','is','fr')

def _sha(value: object) -> str:
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()

def _records(rows: list[dict]) -> dict[tuple[str,str],dict]:
    result = {(row['table'],row['key']):row for row in rows}
    assert len(result)==len(rows), 'duplicate caption arrival identity'
    for identity,row in result.items():
        assert identity[0] in TABLE_NAMES
        assert row['source_sha256']==_sha(row['source']), 'invalid caption source hash'
        if identity[0] in ('UI','CATEGORY_HELP'):
            assert identity[1]==row['source'], 'invalid literal caption identity'
    return result

def _source_rows(external: Mapping[str,object]) -> dict[tuple[str,str],str]:
    assert set(external)==set(TABLE_NAMES), 'unexpected caption source table'
    return {(table,str(key)):str(value) for table,records in external.items()
            for key,value in (records.items() if isinstance(records,dict)
                              else ((value,value) for value in records))}

def load_arrival_contract() -> dict:
    return json.loads(gzip.decompress(FIXTURE.read_bytes()))

def pinned_external_sources(external: Mapping[str,object], *,
                            original_counts: Mapping[str,int],
                            original_identity_sha256: str,
                            contract: dict | None = None) -> dict[str,dict[str,str]]:
    """Require exact current source/review evidence, then restore the old view."""
    import importlib
    import build_i18n_catalogs as builder
    fixture = load_arrival_contract() if contract is None else contract
    assert fixture['schema']==1
    assert fixture['immutable_baseline_catalog_commit']=='3d9f02fe35860c5e1e0337aad56167fe49fa72f9'
    baseline = _records(fixture['baseline'])
    added = _records(fixture['added'])
    retired = _records(fixture['retired'])
    changed_before = _records([record['before'] for record in fixture['changed']])
    changed_after = _records([record['after'] for record in fixture['changed']])
    baseline_signature = _sha('\0'.join(f'{t}\0{k}\0{baseline[(t,k)]["source_sha256"]}'
                                        for t,k in sorted(baseline)))
    assert baseline_signature==PINNED_BASELINE_SOURCE_HASH_SHA256, 'historical caption bodies changed'
    baseline_identity = _sha('\0'.join(f'{t}\0{k}' for t,k in sorted(baseline)))
    assert baseline_identity==original_identity_sha256, 'historical caption identities changed'
    assert {t:sum(table==t for table,key in baseline) for t in TABLE_NAMES}==original_counts
    assert not (baseline.keys() & added.keys()), 'caption arrival already in baseline'
    assert retired.keys() <= baseline.keys(), 'caption retirement outside baseline'
    assert all(retired[key]==baseline[key] for key in retired), 'retired caption baseline changed'
    assert changed_before.keys()==changed_after.keys()
    assert changed_before.keys() <= baseline.keys()-retired.keys(), 'changed caption outside surviving baseline'
    assert all(changed_before[key]==baseline[key] for key in changed_before), 'changed caption baseline changed'
    expected = {key:row for key,row in baseline.items() if key not in retired}
    expected.update(added)
    expected.update(changed_after)
    actual = _source_rows(external)
    assert actual.keys()==expected.keys(), 'unreviewed caption arrival or retirement'
    assert all(_sha(actual[key])==row['source_sha256'] for key,row in expected.items()), 'unreviewed caption source body change'
    catalog_modules = {language:importlib.import_module(f'spacr.qt.i18n_catalogs.{language}') for language in LANGUAGES}
    # Normal review loading validates current source bindings and all ordinary
    # target gates. It is not an advisory compatibility hook.
    reviewed = {language:builder.reviewed_runtime_translations(language) for language in LANGUAGES}
    proof_files = {}
    for identity,row in list(added.items())+list(changed_after.items()):
        assert set(row['reviewed_evidence'])==set(LANGUAGES), 'incomplete caption arrival locales'
        for language,evidence in row['reviewed_evidence'].items():
            catalog = catalog_modules[language]
            assert catalog.SOURCE_HASHES.get(identity)==row['source_sha256'], 'stale caption arrival catalog hash'
            target = getattr(catalog,identity[0])[identity[1]]
            assert _sha(target)==evidence['translation_sha256'], 'caption arrival target drift'
            assert reviewed[language].get(row['source'])==target, 'caption arrival lacks current reviewed target'
            proofs = evidence['matching_review_records']
            assert proofs, 'caption arrival lacks explicit reviewed record'
            for proof in proofs:
                path = ROOT / proof['path']
                if path not in proof_files:
                    raw = path.read_bytes()
                    payload = json.loads(raw)
                    records = {_sha(json.dumps(record,ensure_ascii=False,sort_keys=True)):record
                               for record in payload['records']}
                    assert len(records)==len(payload['records']), 'duplicate caption reviewed record'
                    assert len({(r['table'],r['key']) for r in records.values()})==len(records), 'duplicate caption reviewed identity'
                    proof_files[path] = hashlib.sha256(raw).hexdigest(), payload, records
                file_hash, payload, records = proof_files[path]
                assert file_hash==proof['file_sha256'], 'caption arrival reviewed file bytes drift'
                assert payload['schema']==1 and payload['language']==language
                record = records.get(proof['record_sha256'])
                assert record is not None, 'caption arrival reviewed record drift'
                assert record['table']==TABLE_NAMES[identity[0]] and record['key']==identity[1]
                assert record['source']==row['source'] and record['source_sha256']==row['source_sha256']
                assert record['translation']==target
    # Only the pin assertion uses this reconstructed historical view.
    # Current manifests and compact ownership continue using ``external``.
    projected = {key:source for key,source in actual.items() if key not in added}
    projected.update({key:row['source'] for key,row in retired.items()})
    projected.update({key:row['source'] for key,row in changed_before.items()})
    assert projected.keys()==baseline.keys()
    assert all(projected[key]==row['source'] for key,row in baseline.items())
    return {table:{key:source for (name,key),source in projected.items() if name==table}
            for table in TABLE_NAMES}


def refresh_arrival_contract() -> dict:
    """Rebuild the current delta from canonical sources and exact locale reviews.

    The immutable baseline and its body hash are retained. Every arrival or
    changed body must have a matching current review record in all nine
    languages; generated catalog values and hashes must match those records.
    """
    import importlib
    import build_i18n_catalogs as builder

    previous = load_arrival_contract()
    baseline = _records(previous['baseline'])
    signature = _sha('\0'.join(
        f'{table}\0{key}\0{baseline[(table,key)]["source_sha256"]}'
        for table,key in sorted(baseline)))
    assert signature == PINNED_BASELINE_SOURCE_HASH_SHA256, 'historical caption bodies changed'
    canonical = builder.canonical_sources()
    external = {table: canonical[name] for table,name in TABLE_NAMES.items()}
    actual = _source_rows(external)
    proof_index = {}
    catalogs = {}
    for language in LANGUAGES:
        builder.reviewed_runtime_translations(language)
        catalogs[language] = importlib.import_module(f'spacr.qt.i18n_catalogs.{language}')
        for path in sorted((builder.REVIEWED_RUNTIME_DIR / language).glob('*.json')):
            raw = path.read_bytes()
            payload = json.loads(raw)
            for record in payload['records']:
                key = (language, record['table'], record['key'], record['source'], record['translation'])
                proof = {'path':str(path.relative_to(ROOT)),
                         'file_sha256':hashlib.sha256(raw).hexdigest(),
                         'record_sha256':_sha(json.dumps(record,ensure_ascii=False,sort_keys=True))}
                proof_index.setdefault(key, []).append(proof)

    def current_record(identity):
        table,key = identity
        source = actual[identity]
        row = {'table':table,'key':key,'source':source,'source_sha256':_sha(source),
               'reviewed_evidence':{}}
        for language in LANGUAGES:
            catalog = catalogs[language]
            assert catalog.SOURCE_HASHES.get(identity) == row['source_sha256'], 'stale caption arrival catalog hash'
            target = getattr(catalog,table)[key]
            proof_key = (language,TABLE_NAMES[table],key,source,target)
            proofs = proof_index.get(proof_key)
            assert proofs, f'caption arrival lacks explicit reviewed record: {language}/{identity}'
            row['reviewed_evidence'][language] = {
                'translation':target,'translation_sha256':_sha(target),
                'matching_review_records':proofs}
        return row

    updated = dict(previous)
    updated['added'] = [current_record(key) for key in sorted(actual.keys()-baseline.keys())]
    updated['retired'] = [baseline[key] for key in sorted(baseline.keys()-actual.keys())]
    updated['changed'] = [
        {'before':baseline[key],'after':current_record(key)}
        for key in sorted(actual.keys() & baseline.keys())
        if actual[key] != baseline[key]['source']]
    counts = {table:sum(name==table for name,key in baseline) for table in TABLE_NAMES}
    identity = _sha('\0'.join(f'{table}\0{key}' for table,key in sorted(baseline)))
    pinned_external_sources(external, original_counts=counts,
                            original_identity_sha256=identity, contract=updated)
    assert updated['baseline'] == previous['baseline']
    FIXTURE.write_bytes(gzip.compress(
        (json.dumps(updated,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf-8'),
        mtime=0))
    return {name:len(updated[name]) for name in ('baseline','added','retired','changed')}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', required=True,
                        help='regenerate the delta after authoring all required locale reviews')
    parser.parse_args()
    print(json.dumps(refresh_arrival_contract(), sort_keys=True))
