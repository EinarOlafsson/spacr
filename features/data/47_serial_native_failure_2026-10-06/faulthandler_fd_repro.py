import ctypes
import faulthandler
import json
import os
import resource
from pathlib import Path

from spacr import logging_util
from spacr.qt import app

root = Path(os.environ['SPACR_FAULT_REPRO_ROOT'])
places = iter((root / 'first', root / 'second'))
logging_util.log_dir = lambda: next(places)
assert app._install_crash_dump()
real_enable = faulthandler.enable
faulthandler.enable = lambda **_kw: None
assert app._install_crash_dump()
faulthandler.enable = real_enable
app.__dict__.pop('_CRASH_DUMP_FILE').close()
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
print(json.dumps({'handler_enabled': faulthandler.is_enabled(), 'first': str(root/'first'/'spacr-crash.log'), 'second': str(root/'second'/'spacr-crash.log')}), flush=True)
ctypes.string_at(0)
