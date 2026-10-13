import hashlib
import resource
import tempfile
from pathlib import Path

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
import spacr.io as IO
from tests.test_watch_timelapse_series_f548 import _converted_series
from tests.test_native_tzyx_batch_f548 import _settings

print('IO source', hashlib.sha256(Path(IO.__file__).read_bytes()).hexdigest(), flush=True)

def fail(plane, **kwargs):
    raise RuntimeError('retained rescale diagnostic')

IO.exposure.rescale_intensity = fail
with tempfile.TemporaryDirectory(dir='/mnt/wd4tb/scratch/f548-io-map-lifetime-20261006') as folder:
    source, rows = _converted_series(Path(folder))
    failure = None
    try:
        IO.preprocess_img_data(_settings(source))
    except RuntimeError as observed:
        failure = observed
    assert failure is not None
    trace = failure.__traceback__
    while trace.tb_frame.f_code.co_name != 'fail':
        trace = trace.tb_next
    borrowed = trace.tb_frame.f_locals['plane']
    print('Reading retained rescale plane after failure; core dumps disabled', flush=True)
    print(float(borrowed.sum()), flush=True)
