from pathlib import Path
import json
import hashlib
import build_i18n_catalogs as builder

root = Path.cwd()
out = Path('/media/carruthers/mnt3/codex/scratch/magnifier-acceptance-20261007/original-runtime-reviews-r1')
out.mkdir(exist_ok=False)
sources = builder.canonical_sources()
retired = []
for path in sorted((root / 'docs/i18n/reviewed/runtime').glob('*/*.json')):
    payload = json.loads(path.read_text())
    remove = []
    keep = []
    for row in payload.get('records', []):
        table = sources[row['table']]
        current = table.get(row['key']) if isinstance(table, dict) else row['key'] if row['key'] in table else None
        if current == row['source']:
            keep.append(row)
            continue
        assert row['table'] == 'ui'
        assert row['key'] == 'Detection method' or row['key'].startswith('What a new object does where the mask already has an object.') or row['key'].startswith('Run the CPU method chosen under Detection method') or row['key'].startswith('Segment each field four times, as it is, flipped two ways')
        remove.append(row)
    if remove:
        relative = path.relative_to(root)
        backup = out / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(path.read_bytes())
        payload['records'] = keep
        payload.setdefault('retired_records', []).extend({'date': '2026-10-07', 'reason': 'Maintainer requested Object detection/Magnification settings and whole-object Fuse/Replace semantics; the prior English row is absent from the current canonical source.', 'record': row} for row in remove)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
        retired.append({'file': str(relative), 'original_sha256': hashlib.sha256(backup.read_bytes()).hexdigest(), 'keys': [row['key'] for row in remove]})
for language in builder.MODEL_SPECS:
    print(language, len(builder.reviewed_runtime_translations(language)), 'active reviews valid', flush=True)
(out / 'receipt.json').write_text(json.dumps(retired, ensure_ascii=False, indent=2) + '\n')
