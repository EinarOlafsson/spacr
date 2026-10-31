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
    sources = builder.canonical_sources()
    reviewed = {language:builder._reviewed_runtime_translations(language, sources=sources)
                for language in LANGUAGES}
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


def refresh_compact_caption_pin(*, approved_additions: tuple[str, ...],
                                approved_retirements: tuple[str, ...] = (),
                                pin_path: Path | None = None) -> dict:
    """Refresh only an explicitly approved compact-caption delta.

    The owning semantic discovery remains authoritative. Removing the approved
    arrivals and restoring the approved retirements must reproduce the complete
    old fingerprint, so an unapproved arrival, retirement or replacement cannot
    silently move this pin. A retirement is a caption the source no longer
    shows; its literal row is kept, so older translated text is not lost.
    """
    import ast
    import importlib
    import os
    import tempfile

    contract = importlib.import_module('tests.qt.test_i18n_caption_ratchet')
    compact = importlib.import_module('spacr.qt.i18n')
    assert Path(contract.ROOT).resolve() == ROOT.resolve(), 'foreign compact discovery root'
    assert Path(compact.__file__).resolve() == ROOT / 'spacr/qt/i18n.py', 'foreign compact rows'
    path = ROOT / 'tests/qt/test_i18n_caption_ratchet.py' if pin_path is None else pin_path
    original = path.read_text(encoding='utf-8')
    assignments = {}
    for node in ast.parse(original).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                        'COMPACT_CAPTION_COUNT', 'COMPACT_CAPTION_SHA256'):
                    assert target.id not in assignments, 'duplicate compact pin'
                    assignments[target.id] = node
    assert len(assignments) == 2, 'missing compact pin'
    old_count = ast.literal_eval(assignments['COMPACT_CAPTION_COUNT'].value)
    old_digest = ast.literal_eval(assignments['COMPACT_CAPTION_SHA256'].value)
    assert type(old_count) is int and old_count >= 0, 'invalid compact count'
    assert isinstance(old_digest, str) and len(old_digest) == 64, 'invalid compact digest'
    approved = frozenset(approved_additions)
    retired = frozenset(approved_retirements)
    assert approved or retired, 'empty compact approvals'
    assert len(approved) == len(approved_additions), 'duplicate compact approvals'
    assert len(retired) == len(approved_retirements), 'duplicate compact retirements'
    assert not approved & retired, 'compact caption both added and retired'
    assert all(isinstance(source, str) and source.strip() for source in approved | retired)
    discovered = contract.compact_user_facing_captions()
    assert approved <= discovered, 'approved compact caption is not discovered'
    assert not retired & discovered, 'approved compact retirement is still discovered'
    previous = (discovered - approved) | retired
    assert len(previous) == old_count, 'unapproved compact arrival or retirement'
    assert _sha('\0'.join(sorted(previous))) == old_digest, 'historical compact surface changed'
    assert not discovered - compact._ROWS.keys(), 'compact caption lacks an exact row'
    assert tuple(compact._TRANSLATED_CODES) == LANGUAGES, 'compact locale order changed'
    for source in approved:
        row = compact._ROWS[source]
        assert len(row) == len(LANGUAGES), 'incomplete compact locale row'
        assert all(isinstance(value, str) and value.strip() for value in row), 'empty compact target'
    count = len(discovered)
    digest = _sha('\0'.join(sorted(discovered)))
    lines = original.splitlines(keepends=True)
    replacements = {
        'COMPACT_CAPTION_COUNT': f'COMPACT_CAPTION_COUNT = {count}\n',
        'COMPACT_CAPTION_SHA256': f'COMPACT_CAPTION_SHA256 = (\n    "{digest}"\n)\n',
    }
    for name, node in sorted(assignments.items(), key=lambda item: item[1].lineno, reverse=True):
        lines[node.lineno - 1:node.end_lineno] = [replacements[name]]
    updated = ''.join(lines)
    assert path.read_text(encoding='utf-8') == original, 'compact pin changed during refresh'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.compact-pin-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    result = {'count': count, 'sha256': digest, 'approved_additions': sorted(approved)}
    if retired:
        result['approved_retirements'] = sorted(retired)
    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--refresh', action='store_true',
                        help='regenerate the delta after authoring all required locale reviews')
    action.add_argument('--refresh-compact', action='store_true',
                        help='refresh the compact pin after explicitly approving exact new rows')
    parser.add_argument('--approved-compact-addition', action='append', default=[],
                        help='one explicitly reviewed new compact caption; repeat for each addition')
    parser.add_argument('--approved-compact-retirement', action='append', default=[],
                        help='one compact caption the source no longer shows; repeat for each retirement')
    args = parser.parse_args()
    if args.refresh_compact:
        from PySide6.QtWidgets import QApplication

        application = QApplication.instance() or QApplication([])
        result = refresh_compact_caption_pin(
            approved_additions=tuple(args.approved_compact_addition),
            approved_retirements=tuple(args.approved_compact_retirement),
        )
    else:
        if args.approved_compact_addition or args.approved_compact_retirement:
            parser.error('compact approvals require --refresh-compact')
        result = refresh_arrival_contract()
    print(json.dumps(result, sort_keys=True))
