import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

folder = Path(os.environ['SPACR_PROBE_EVIDENCE'])
executable = Path(sys.executable).resolve()
stamp = time.time_ns()
row = {
    'event': 'session_start', 'mode': 'identity_only', 'pid': os.getpid(),
    'time_ns': stamp, 'source_sha': os.environ['GITHUB_SHA'],
    'root': os.environ['GITHUB_WORKSPACE'], 'executable': str(executable),
    'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
    'process_start_ticks': Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19],
    'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    'worker': 'controlled-native-probe',
}
(folder / f'process-{os.getpid()}-{stamp}.jsonl').write_text(json.dumps(row) + '\n')
os.kill(os.getpid(), signal.SIGSEGV)
