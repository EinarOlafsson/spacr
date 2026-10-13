"""Generate hashes for the compact independent review payloads."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).parent
payloads = {}
for path in sorted(root.iterdir()):
    if path.is_file() and path.name != 'MANIFEST.json':
        raw = path.read_bytes()
        payloads[path.name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
(root / 'MANIFEST.json').write_text(json.dumps({'schema': 1, 'payloads': payloads}, indent=2) + '\n')
