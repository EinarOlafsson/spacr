"""Measure keeps its workers inside free RAM.

The clamp math, the pause-and-resume throttle and the process list are all
driven by a fake psutil, so no test depends on this machine's memory and no
real process is ever signalled.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from spacr import measure, resource_log

GIB = 1024 ** 3


class _FakeProcess:
    def __init__(self, pid, name, user, rss, registry):
        self.pid = pid
        self.info = {'pid': pid, 'name': name, 'username': user,
                     'memory_info': types.SimpleNamespace(rss=rss)}
        self._registry = registry
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def username(self):
        return self.info['username']

    def parents(self):
        return [self._registry[1]]

    def children(self, recursive=False):
        return []


class _FakePsutil:
    """Just the psutil surface the guard reads."""

    class NoSuchProcess(Exception):
        pass

    def __init__(self, readings=(), processes=()):
        self._readings = list(readings)
        self.processes = {}
        for pid, name, user, rss in processes:
            self.processes[pid] = _FakeProcess(pid, name, user, rss,
                                               self.processes)
        self.processes.setdefault(100, _FakeProcess(100, 'python', 'me', 0,
                                                    self.processes))
        self.processes.setdefault(1, _FakeProcess(1, 'systemd', 'root', 0,
                                                  self.processes))

    def virtual_memory(self):
        available, total = (self._readings.pop(0) if len(self._readings) > 1
                            else self._readings[0])
        return types.SimpleNamespace(available=available, total=total)

    def Process(self, pid=None):
        return self.processes[100 if pid is None else pid]

    def process_iter(self, _attrs):
        return list(self.processes.values())


@pytest.fixture
def merged(tmp_path):
    folder = tmp_path / 'plate' / 'merged'
    folder.mkdir(parents=True)
    np.save(folder / 'field_1.npy', np.zeros((1024, 1024, 4), np.float32))
    return folder


def test_max_safe_workers_keeps_the_reserve():
    assert measure._max_safe_workers(64 * GIB, 128 * GIB, 4 * GIB) == 12
    assert measure._max_safe_workers(1 * GIB, 128 * GIB, 4 * GIB) == 1


def test_the_plan_reads_one_field_and_the_free_ram(merged):
    fake = _FakePsutil([(20 * GIB, 32 * GIB)])
    plan = measure._ram_guard_plan(str(merged), 30, multiplier=100,
                                   psutil_module=fake)
    assert plan['nbytes'] == 16 * 1024 ** 2
    assert plan['per_worker'] == 1600 * 1024 ** 2
    assert plan['max_safe'] == int((20 - 4) * GIB // plan['per_worker'])
    assert plan['exceeds']


def test_the_plan_also_finds_merged_under_the_plate(merged):
    fake = _FakePsutil([(20 * GIB, 32 * GIB)])
    assert measure._ram_guard_plan(str(merged.parent), 2,
                                   psutil_module=fake) is not None


def test_a_headless_run_is_clamped_with_a_warning(capsys):
    plan = {'per_worker': 2 * GIB, 'available': 10 * GIB,
            'reserve': 2 * GIB, 'max_safe': 4}
    assert measure._clamp_workers_to_ram({}, 16, plan) == 4
    assert 'using 4 workers' in capsys.readouterr().out
    assert measure._clamp_workers_to_ram({'ram_guard': False}, 16, plan) == 16
    assert measure._clamp_workers_to_ram({}, 3, plan) == 3
    assert measure._clamp_workers_to_ram({}, 16, None) == 16


def test_calibration_shrinks_the_wave_only_with_the_guard_on(capsys):
    plan = {'per_worker': GIB, 'nbytes': GIB // 8, 'available': 20 * GIB,
            'total': 32 * GIB}
    per_worker, wave = measure._calibrated_wave({}, plan, 4 * GIB, 10)
    assert (per_worker, wave) == (4 * GIB, 4)
    assert '32.0x' in capsys.readouterr().out
    assert measure._calibrated_wave({'ram_guard': False}, plan, 4 * GIB,
                                    10) == (4 * GIB, 10)
    assert measure._calibrated_wave({}, plan, 0, 10) == (GIB, 10)


def test_the_throttle_pauses_then_resumes(capsys):
    fake = _FakePsutil([(3 * GIB, 16 * GIB), (3 * GIB, 16 * GIB),
                        (10 * GIB, 16 * GIB)])
    sleeps = []
    waited = measure._wait_for_ram(2 * GIB, lambda: True, field='f.npy',
                                   psutil_module=fake, sleep=sleeps.append,
                                   poll=1.0)
    out = capsys.readouterr().out
    assert waited == 2.0 and len(sleeps) == 2
    assert 'holding f.npy' in out and 'resuming with f.npy' in out
    assert out.count('holding') == 1


def test_the_throttle_never_waits_when_nothing_is_running(capsys):
    fake = _FakePsutil([(1 * GIB, 16 * GIB)])
    sleeps = []
    assert measure._wait_for_ram(2 * GIB, lambda: False, field='f.npy',
                                 psutil_module=fake,
                                 sleep=sleeps.append) == 0.0
    assert sleeps == []


def test_the_throttle_does_not_wait_with_room():
    fake = _FakePsutil([(12 * GIB, 16 * GIB)])
    assert measure._wait_for_ram(2 * GIB, lambda: True,
                                 psutil_module=fake,
                                 sleep=pytest.fail) == 0.0


def test_a_running_field_is_seen_through_ready():
    done = types.SimpleNamespace(ready=lambda: True)
    running = types.SimpleNamespace(ready=lambda: False)
    assert not measure._any_field_running([('a', 0, done)])
    assert measure._any_field_running([('a', 0, done), ('b', 1, running)])


def _processes():
    return _FakePsutil([(1, 1)], processes=[
        (100, 'python', 'me', 9 * GIB),
        (200, 'firefox', 'me', 3 * GIB),
        (201, 'slack', 'me', 2 * GIB),
        (300, 'postgres', 'other', 8 * GIB),
        (301, 'Xorg', 'root', 1 * GIB),
        (302, 'gnome-shell', 'me', 1 * GIB),
        (303, 'spacr-worker', 'me', 5 * GIB),
    ])


def test_only_the_users_own_programs_are_listed_largest_first():
    rows = resource_log._closable_processes(_processes())
    assert [row['name'] for row in rows] == ['firefox', 'slack']


def test_closing_terminates_only_listed_programs():
    fake = _processes()
    outcomes = resource_log._close_processes([200, 100, 301], fake)
    assert outcomes == {200: 'closed', 100: 'refused', 301: 'refused'}
    assert fake.processes[200].terminated
    assert not fake.processes[100].terminated
    assert not fake.processes[301].terminated
