"""Read the reviewed DINO captions through the ordinary translation API."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from spacr.qt.i18n import tr
results = {}
for path in sorted((ROOT / 'docs/i18n/reviewed/runtime').glob('*/2026-10-15-own-dino-controls-home-b.json')):
    payload = json.loads(path.read_text())
    for record in payload['records']:
        assert tr(record['source'], language='en') == record['source']
        assert tr(record['source'], language=payload['language']) == record['translation']
    results[payload['language']] = len(payload['records'])
assert len(results) == 9 and sum(results.values()) == 108
print(json.dumps({'api': 'spacr.qt.i18n.tr', 'exact_target_counts': results, 'english_preserved': True}, indent=2))
