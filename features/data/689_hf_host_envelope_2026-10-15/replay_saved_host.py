"""Replay the recorded public response through the current verifier offline."""
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from urllib.parse import urlparse

evidence = Path(__file__).resolve().parent
root = evidence.parents[2]
spec = importlib.util.spec_from_file_location('publisher_under_review', root / 'tools/publish_docs_channels.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)
def saved(name):
    return gzip.decompress((evidence / (name + '.body.gz')).read_bytes())
responses = {'publication.json': saved('hf-publication'),
             'channels.json': saved('hf-channels'), 'index.html': saved('hf-index')}
requests = []
def fetch(request, timeout):
    name = urlparse(request.full_url).path.rsplit('/', 1)[-1]
    requests.append(name)
    return io.BytesIO(responses[name])
with tempfile.TemporaryDirectory() as folder:
    site = Path(folder)
    (site / 'channels.json').write_bytes(responses['channels.json'])
    (site / 'index.html').write_bytes(saved('hf-immutable-index'))
    with patch.object(publisher, 'urlopen', fetch):
        commit = publisher.verify_nightly(site, attempts=1, interval=0)
assert requests == ['publication.json', 'channels.json', 'index.html']
print(json.dumps({'result': 'PASS', 'commit': commit, 'network_calls': 0}))
