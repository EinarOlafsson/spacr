import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
payloads = {}
for path in sorted(here.iterdir()):
    if path.is_file() and path.name != 'MANIFEST.json':
        raw = path.read_bytes()
        payloads[path.name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
(here / 'MANIFEST.json').write_text(json.dumps({'schema': 1, 'payloads': payloads}, indent=2, sort_keys=True) + '\n')
print(len(payloads), sum(row['bytes'] for row in payloads.values()))
