"""Read the one reviewed caption through the ordinary runtime translation API."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from spacr.qt.i18n import tr

source = ('This row was the first without a database. Filename matching '
          'did not attach the database to a plate row.')
assert tr(source, language='en') == source
results = {}
for path in sorted((ROOT / 'docs/i18n/reviewed/runtime').glob(
        '*/2026-10-15-database-pair-fallback.json')):
    payload = json.loads(path.read_text())
    assert len(payload['records']) == 1
    record = payload['records'][0]
    assert record['source'] == source
    actual = tr(source, language=payload['language'])
    assert actual == record['translation'], (payload['language'], actual)
    results[payload['language']] = hashlib.sha256(actual.encode()).hexdigest()
assert len(results) == 9
print(json.dumps({'passed': True, 'api': 'spacr.qt.i18n.tr',
                  'english_preserved': True, 'target_hashes': results}, indent=2))
