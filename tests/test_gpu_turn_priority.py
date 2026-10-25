"""The GPU queue lets priority jobs (label prefix) go first without stopping running work."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


pytest.importorskip('fcntl', reason='The GPU queue uses POSIX flock')
pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Priority detection reads Linux /proc')
SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'gpu_turn.sh'
PREFIX = 'prioritytest-'


def _environment(tmp_path, prefixes=PREFIX):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith('GPU_')}
    environment.update(HOME=str(tmp_path / 'home'), GPU_IDLE_S='0', GPU_GAP_S='0',
                       GPU_POLL_S='0.2', GPU_PRIORITY_PREFIXES=prefixes)
    return environment


def _job(tmp_path, label, marker, environment):
    command = ['bash', str(SCRIPT), label, sys.executable, '-c',
               'import pathlib, sys; pathlib.Path(sys.argv[1]).write_text("ran")', str(marker)]
    return subprocess.Popen(command, env=environment, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def _fake_queued(tmp_path, label, home=None):
    environment = dict(os.environ, HOME=str(home or tmp_path / 'home'))
    (Path(environment['HOME']) / '.spacr' / 'gpu').mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(['bash', '-c', 'sleep 30; true', '/fake/tools/gpu_turn.sh', label],
                            env=environment)


def _wait_for(path, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.1)
    return path.exists()


def _log(tmp_path):
    path = tmp_path / 'home' / '.spacr' / 'gpu' / 'log'
    return path.read_text(encoding='utf-8') if path.exists() else ''


def test_ordinary_job_waits_while_a_priority_job_is_queued(tmp_path):
    marker = tmp_path / 'started'
    fake = _fake_queued(tmp_path, PREFIX + 'segmentation')
    try:
        job = _job(tmp_path, 'ordinary-screen', marker, _environment(tmp_path))
        assert not _wait_for(marker, 2.0)
        assert 'hands the queue to a waiting priority job' in _log(tmp_path)
    finally:
        fake.terminate(); fake.wait(timeout=5)
    assert _wait_for(marker, 5.0)
    assert job.wait(timeout=5) == 0
    assert 'holds the queue again' in _log(tmp_path)


def test_priority_job_does_not_yield_to_another_priority_job(tmp_path):
    marker = tmp_path / 'started'
    fake = _fake_queued(tmp_path, PREFIX + 'other')
    try:
        job = _job(tmp_path, PREFIX + 'mine', marker, _environment(tmp_path))
        assert _wait_for(marker, 5.0)
        assert job.wait(timeout=5) == 0
    finally:
        fake.terminate(); fake.wait(timeout=5)
    assert 'hands the queue' not in _log(tmp_path)


def test_empty_prefix_list_turns_priority_off(tmp_path):
    marker = tmp_path / 'started'
    fake = _fake_queued(tmp_path, PREFIX + 'segmentation')
    try:
        job = _job(tmp_path, 'ordinary-screen', marker, _environment(tmp_path, prefixes=''))
        assert _wait_for(marker, 5.0)
        assert job.wait(timeout=5) == 0
    finally:
        fake.terminate(); fake.wait(timeout=5)


def test_ordinary_job_starts_when_no_priority_job_is_queued(tmp_path):
    marker = tmp_path / 'started'
    job = _job(tmp_path, 'ordinary-screen', marker, _environment(tmp_path))
    assert _wait_for(marker, 5.0)
    assert job.wait(timeout=5) == 0
    assert 'hands the queue' not in _log(tmp_path)


def test_priority_job_on_another_queue_is_ignored(tmp_path):
    marker = tmp_path / 'started'
    fake = _fake_queued(tmp_path, PREFIX + 'elsewhere', home=tmp_path / 'other-home')
    try:
        job = _job(tmp_path, 'ordinary-screen', marker, _environment(tmp_path))
        assert _wait_for(marker, 5.0)
        assert job.wait(timeout=5) == 0
    finally:
        fake.terminate(); fake.wait(timeout=5)
    assert 'hands the queue' not in _log(tmp_path)
