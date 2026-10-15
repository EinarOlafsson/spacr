"""One bounded read-only source-bound Space runtime/build-log snapshot."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import requests
from huggingface_hub import get_token
S = Path('/mnt/wd4tb/scratch/n698-managed-provisioning-20261015')
repo = 'einarolafsson/spacr-mask-generator'
commit = '3371bce879bb15a6597d6f268071ebfb2df1e23c'
session = requests.Session(); session.headers['Authorization'] = 'Bearer ' + get_token()
base = 'https://huggingface.co/api/spaces/' + repo
result = {'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'repo': repo, 'expected_commit': commit, 'actions': 'GET only; no restart/settings/upload/download/inference'}
for name, url in [('info', base), ('runtime', base + '/runtime')]:
    response = session.get(url, timeout=(10, 25), allow_redirects=False); response.raise_for_status(); data = response.json()
    if name == 'info':
        assert data.get('sha') == commit
        result['source'] = {k: data.get(k) for k in ('id', 'sha', 'sdk', 'private')}
    else:
        result['runtime'] = {k: data.get(k) for k in ('stage', 'hardware', 'errorMessage', 'gcTimeout', 'resources')}
log = S / 'managed-build-after-packaging.log'; cap = 4 * 1024 * 1024; received = 0; deadline = time.monotonic() + 20
try:
    with session.get(base + '/logs/build', stream=True, timeout=(10, 8), allow_redirects=False) as response:
        result['build_log_http_status'] = response.status_code
        result['build_log_content_type'] = response.headers.get('content-type')
        response.raise_for_status()
        with log.open('xb') as handle:
            for chunk in response.iter_content(chunk_size=4096):
                if time.monotonic() >= deadline:
                    result['build_log_bounded_stop'] = '20-second window'; break
                if chunk:
                    allowed = chunk[:cap-received]; handle.write(allowed); received += len(allowed)
                if received >= cap:
                    result['build_log_bounded_stop'] = '4MiB limit'; break
except requests.RequestException as error:
    result['build_log_capture_stop'] = type(error).__name__
    if getattr(error, 'response', None) is not None: result['build_log_error_status'] = error.response.status_code
result['build_log_bytes'] = received
if log.exists(): result['build_log_sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
p = S / 'managed-build-after-packaging.json'; p.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result, indent=2))
