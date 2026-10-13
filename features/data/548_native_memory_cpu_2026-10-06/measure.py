import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

from spacr import io
from tests.test_native_tzyx_batch_f548 import _settings

parser = argparse.ArgumentParser(description='Measure a source-verified native ingest.')
parser.add_argument('source', type=Path)
parser.add_argument('--source-sha256', required=True)
args = parser.parse_args()
source = args.source.resolve()
actual_sha = hashlib.sha256(Path(io.__file__).read_bytes()).hexdigest()
if actual_sha != args.source_sha256:
    parser.error('loaded io.py differs from the required frozen source SHA256')
started = time.perf_counter()
settings, returned = io.preprocess_img_data(_settings(source))
print(json.dumps({
    'source': str(source),
    'io_source_sha256': actual_sha,
    'returned': returned,
    'seconds': round(time.perf_counter() - started, 3),
    'peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
    'channels': settings['channels'],
}), flush=True)
