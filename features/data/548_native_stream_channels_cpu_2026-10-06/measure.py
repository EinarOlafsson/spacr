import json
import resource
import sys
import time
from pathlib import Path

from spacr import io
from tests.test_native_tzyx_batch_f548 import _settings

source = Path(sys.argv[1])
started = time.perf_counter()
settings, returned = io.preprocess_img_data(_settings(source))
print(json.dumps({
    'source': str(source),
    'returned': returned,
    'seconds': round(time.perf_counter() - started, 3),
    'peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
    'channels': settings['channels'],
}), flush=True)
