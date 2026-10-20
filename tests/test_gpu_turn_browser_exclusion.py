"""The real GPU queue ignores browser contexts while retaining its admission gates."""

import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest


fcntl = pytest.importorskip('fcntl', reason='The GPU queue uses POSIX flock')
pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Process identity uses Linux /proc')
SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'gpu_turn.sh'


@contextmanager
def _process(tmp_path, name, args=(), *, deleted=False):
    executable = tmp_path / name
    shutil.copyfile('/bin/sleep', executable)
    executable.chmod(0o700)
    process = subprocess.Popen([str(executable), '60', *args], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    try:
        if deleted:
            executable.unlink()
        yield process
    finally:
        process.terminate()
        process.wait(timeout=5)


def _scheduler(tmp_path, pid, *, blocked=False, unreadable=False):
    home = tmp_path / 'isolated-home'
    directory = home / '.spacr' / 'gpu'
    directory.mkdir(parents=True)
    clock = tmp_path / 'clock'
    clock.write_text('1000', encoding="utf-8")
    (directory / 'last_release').write_text('1000', encoding="utf-8")
    query_log = tmp_path / 'queries'
    binary = tmp_path / 'bin'
    binary.mkdir()
    programs = {
        'date': 'if [ "$#" -eq 1 ] && [ "$1" = "+%s" ]; then\n'
                '  cat "$CLOCK"\n'
                'else\n'
                '  exec /bin/date "$@"\n'
                'fi\n',
        'sleep': 'now=$(cat "$CLOCK")\n'
                 'printf "%s" "$((now + $1))" > "$CLOCK"\n',
        'nvidia-smi': 'now=$(cat "$CLOCK")\n'
                      'printf "%s\\n" "$now" >> "$QUERIES"\n'
                      'if [ "$now" -lt "$CLEAR_AT" ]; then\n'
                      '  printf "%s\\n" "$TEST_PID"\n'
                      'fi\n',
    }
    if unreadable:
        programs['readlink'] = 'exit 1\n'
        programs['tr'] = ('if [ "$#" -eq 2 ] && [ "$1" = "-d" ] && [ "$2" = " " ]; then\n'
                          '  exec /usr/bin/tr "$@"\n'
                          'fi\n'
                          'exit 1\n')
    for name, source in programs.items():
        path = binary / name
        path.write_text('#!/bin/sh\n' + source, encoding="utf-8")
        path.chmod(0o700)
    environment = {key: value for key, value in os.environ.items()
                   if key not in {'GPU_IDLE_S', 'GPU_GAP_S', 'GPU_POLL_S'}}
    environment.update(HOME=str(home), PATH=str(binary) + os.pathsep + os.environ['PATH'],
                       CLOCK=str(clock), QUERIES=str(query_log), TEST_PID=str(pid),
                       CLEAR_AT='1300' if blocked else '999999')
    marker = tmp_path / 'started'
    command = ['bash', str(SCRIPT), 'cpu-only-browser-admission', sys.executable, '-c',
               'from pathlib import Path; import sys; '
               'Path(sys.argv[1]).write_text(Path(sys.argv[2]).read_text(encoding="utf-8"))',
               str(marker), str(clock)]
    return command, environment, marker, directory, query_log


def _run_scheduler(settings):
    command, environment, marker, directory, queries = settings
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (directory / 'holder').read_text(encoding="utf-8") == ''
    assert (directory / 'last_release').read_text(encoding="utf-8").strip() == marker.read_text(encoding="utf-8")
    return int(marker.read_text(encoding="utf-8")), queries.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize('name', ['chrome', 'chromium', 'chromium-browser',
                                 'google-chrome', 'google-chrome-stable'])
def test_browser_only_admission_still_observes_default_idle_and_gap(tmp_path, name):
    with _process(tmp_path, name) as process:
        started, queries = _run_scheduler(_scheduler(tmp_path, process.pid))
    assert started == 1600
    assert queries[0] == '1000' and queries[-1] == '1600'


def test_deleted_browser_executable_is_still_excluded(tmp_path):
    with _process(tmp_path, 'chrome', deleted=True) as process:
        assert os.readlink(f'/proc/{process.pid}/exe').endswith(' (deleted)')
        started, _ = _run_scheduler(_scheduler(tmp_path, process.pid))
    assert started == 1600


@pytest.mark.parametrize('argument,blocked', [('--type=gpu-process', False),
                                            ('--type=gpu-process-extra', True),
                                            ('prefix--type=gpu-process', True)])
def test_graphics_flag_is_an_exact_argument_not_a_compute_substring(tmp_path, argument, blocked):
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', argument])
    try:
        started, _ = _run_scheduler(_scheduler(tmp_path, process.pid, blocked=blocked))
    finally:
        process.terminate()
        process.wait(timeout=5)
    assert started == (1645 if blocked else 1600)


@pytest.mark.parametrize('identity', ['compute', 'missing', 'unreadable'])
def test_compute_or_unknown_identity_keeps_resetting_idle_until_it_leaves(tmp_path, identity):
    with _process(tmp_path, 'not-chrome-compute') as process:
        pid = 99999999 if identity == 'missing' else process.pid
        started, queries = _run_scheduler(_scheduler(tmp_path, pid, blocked=True,
                                                     unreadable=identity == 'unreadable'))
    assert started == 1645
    assert '1285' in queries and '1300' in queries


def test_browser_exclusion_does_not_bypass_the_shared_queue_lock(tmp_path):
    with _process(tmp_path, 'chrome') as process:
        settings = _scheduler(tmp_path, process.pid)
        command, environment, marker, directory, queries = settings
        with (directory / 'lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            scheduler = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 5
                while not (directory / 'log').exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert (directory / 'log').exists()
                assert not marker.exists() and not queries.exists()
                assert scheduler.poll() is None
                fcntl.flock(lock, fcntl.LOCK_UN)
                output, error = scheduler.communicate(timeout=25)
                assert scheduler.returncode == 0, output + error
                assert marker.read_text(encoding="utf-8") == '1600'
            finally:
                if scheduler.poll() is None:
                    scheduler.terminate()
                    scheduler.wait(timeout=5)
