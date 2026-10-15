from pathlib import Path
import subprocess
pid = 2405901
expected_start = '52692424'
scope = 'run-u64915.scope'
path = Path('/proc') / str(pid)
if path.exists():
    actual_start = (path / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    membership = (path / 'cgroup').read_text()
    if actual_start == expected_start and '/' + scope in membership:
        subprocess.run(['systemctl', '--user', 'stop', scope], check=True, timeout=20)
