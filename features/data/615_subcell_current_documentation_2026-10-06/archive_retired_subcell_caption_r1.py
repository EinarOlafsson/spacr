from pathlib import Path
import hashlib
import json
import build_i18n_catalogs as runtime
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
inventory = json.loads((scratch / 'subcell-rybg-documentation-inventory-r3.json').read_text())
retired = inventory['retired_runtime']
assert len(retired) == 1
source = retired[0]
assert source not in runtime.canonical_sources()['ui']
report = {}
for language in ('sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'):
    folder = Path('docs/i18n/reviewed/runtime') / language
    found = []
    for path in folder.glob('*.json'):
        raw = path.read_bytes()
        doc = json.loads(raw)
        removed = [row for row in doc['records'] if row.get('table') == 'ui' and row.get('key') == source]
        if not removed:
            continue
        assert len(removed) == 1
        assert removed[0]['source'] == source
        assert removed[0]['source_sha256'] == hashlib.sha256(source.encode()).hexdigest()
        archive = folder / 'archive/2026-10-06-subcell-rybg' / path.name
        archive.parent.mkdir(parents=True, exist_ok=True)
        assert not archive.exists()
        archive.write_bytes(raw)
        assert archive.read_bytes() == raw
        doc['records'] = [row for row in doc['records'] if row not in removed]
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')
        active = json.loads(path.read_text())
        original = json.loads(archive.read_text())
        assert active == {**original, 'records': [row for row in original['records'] if row not in removed]}
        found.append({'original_review_file': str(path), 'complete_original_bytes_archive': str(archive),
                      'original_sha256': hashlib.sha256(raw).hexdigest(), 'retired_record': removed[0],
                      'every_other_complete_record_and_header_and_order_exact': True})
    assert len(found) == 1, (language, found)
    report[language] = found[0]
    print('PASS obsolete two-plane-only caption review retired; complete original archived', language)
(scratch / 'subcell-rybg-retired-caption-archive-r1.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
