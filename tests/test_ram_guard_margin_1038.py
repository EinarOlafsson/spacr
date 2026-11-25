"""Item 1038: the batch loader's free-RAM margin follows the RAM limit, and a wait is never silent.

The cgroup files, the environment and the meminfo readers are all faked, so
no test depends on this machine's memory or on the sched scope it runs in.
"""
from __future__ import annotations

import logging

import pytest

from spacr import _mask_batches as MB
from spacr import resource_log

GIB = 2**30


@pytest.mark.parametrize('limit, margin', [
    (2 * GIB, int(0.2 * GIB)),
    (8 * GIB, int(0.8 * GIB)),
    (64 * GIB, 2 * GIB),
])
def test_the_margin_is_ten_percent_of_the_limit_up_to_two_gib(monkeypatch, limit, margin):
    monkeypatch.setattr(MB, '_cgroup_limit_bytes', lambda: limit)
    monkeypatch.setattr(MB, '_sched_declared_ram_bytes', lambda: None)
    monkeypatch.setattr(MB, '_host_total_ram_bytes', lambda: 125 * GIB)
    assert MB._ram_limit() == (limit, 'cgroup')
    assert MB._ram_margin_bytes() == margin
    assert resource_log._free_ram_margin_bytes(limit) == margin


def test_without_any_limit_the_host_total_sets_the_margin(monkeypatch):
    monkeypatch.setattr(MB, '_cgroup_limit_bytes', lambda: None)
    monkeypatch.setattr(MB, '_sched_declared_ram_bytes', lambda: None)
    monkeypatch.setattr(MB, '_host_total_ram_bytes', lambda: 125 * GIB)
    assert MB._ram_limit() == (125 * GIB, 'host')
    assert MB._ram_margin_bytes() == 2 * GIB
    monkeypatch.setattr(MB, '_host_total_ram_bytes', lambda: None)
    assert MB._ram_limit() == (None, None)
    assert MB._ram_margin_bytes() == 2 * GIB


def test_the_sched_declaration_and_the_cgroup_tightest_wins(monkeypatch):
    declared = MB._sched_declared_ram_bytes
    monkeypatch.setattr(MB, '_cgroup_limit_bytes', lambda: 8 * GIB)
    monkeypatch.setattr(MB, '_sched_declared_ram_bytes', lambda: 2 * GIB)
    assert MB._ram_limit() == (2 * GIB, 'sched')
    assert declared({'SCHED_RAM_GB': '2'}) == 2 * GIB
    assert declared({'SCHED_RAM_GB': '1.5'}) == int(1.5 * GIB)
    assert declared({'SCHED_RAM_GB': 'x'}) is None
    assert declared({}) is None


def test_the_cgroup_limit_reads_memory_high_as_well_as_memory_max(tmp_path):
    root = tmp_path / 'cg'
    scope = root / 'user.slice' / 'job.scope'
    scope.mkdir(parents=True)
    (root / 'memory.max').write_text('max\n')
    (root / 'user.slice' / 'memory.max').write_text(f'{60 * GIB}\n')
    (scope / 'memory.max').write_text(f'{8 * GIB}\n')
    (scope / 'memory.high').write_text(f'{6 * GIB}\n')
    proc = tmp_path / 'cgroup'
    proc.write_text('0::/user.slice/job.scope\n')
    assert MB._cgroup_limit_bytes(str(proc), str(root)) == 6 * GIB
    (scope / 'memory.high').write_text('max\n')
    assert MB._cgroup_limit_bytes(str(proc), str(root)) == 8 * GIB
    assert MB._cgroup_limit_bytes(str(tmp_path / 'absent'), str(root)) is None


def test_every_wait_is_logged_with_free_margin_limit_and_time(caplog, capsys):
    levels = iter([int(0.1 * GIB), int(0.1 * GIB), 4 * GIB])
    with caplog.at_level(logging.WARNING, logger='spacr.mask_batches'):
        MB._wait_for_ram(GIB, available=lambda: next(levels), sleep=lambda s: None,
                         limit=lambda: (8 * GIB, 'cgroup'))
    messages = [record.getMessage() for record in caplog.records]
    assert any('waiting for RAM' in m and '0.10 GiB free' in m and '0.80 GiB margin' in m
               and '8.0 GiB (cgroup)' in m and 'waited 0 of' in m for m in messages)
    assert any('free again' in m for m in messages)
    assert 'WARNING: RAM guard' in capsys.readouterr().out


def test_a_long_wait_repeats_its_warning(monkeypatch, caplog):
    monkeypatch.setattr(MB, '_RAM_WAIT_SECONDS', 70.0)
    with caplog.at_level(logging.WARNING, logger='spacr.mask_batches'):
        MB._wait_for_ram(0, available=lambda: 0, sleep=lambda s: None,
                         limit=lambda: (2 * GIB, 'sched'))
    waits = [r.getMessage() for r in caplog.records if 'waiting for RAM' in r.getMessage()]
    assert len(waits) == 3                                  # at 0, 30 and 60 s


def test_after_the_timeout_it_proceeds_when_the_batch_still_fits(monkeypatch, caplog):
    monkeypatch.setattr(MB, '_RAM_WAIT_SECONDS', 3.0)
    with caplog.at_level(logging.WARNING, logger='spacr.mask_batches'):
        waited = MB._wait_for_ram(GIB, available=lambda: int(1.1 * GIB), sleep=lambda s: None,
                                  limit=lambda: (4 * GIB, 'cgroup'))
    assert waited == 3.0
    assert any('without the margin' in r.getMessage() for r in caplog.records)


def test_after_the_timeout_it_raises_under_a_hard_limit_it_cannot_fit(monkeypatch):
    monkeypatch.setattr(MB, '_RAM_WAIT_SECONDS', 3.0)
    with pytest.raises(MB.RamLimitError, match=r'Declare at least 4 GiB') as caught:
        MB._wait_for_ram(2 * GIB, available=lambda: int(0.5 * GIB), sleep=lambda s: None,
                         limit=lambda: (2 * GIB, 'sched'))
    assert isinstance(caught.value, MemoryError)


def test_after_the_timeout_it_proceeds_on_a_host_without_a_limit(monkeypatch):
    monkeypatch.setattr(MB, '_RAM_WAIT_SECONDS', 3.0)
    assert MB._wait_for_ram(10 * GIB, available=lambda: 0, sleep=lambda s: None,
                            limit=lambda: (125 * GIB, 'host')) == 3.0


def test_a_job_declared_at_peak_plus_a_little_no_longer_stalls(monkeypatch):
    """A 2 GiB sched job peaking at 1.6 GiB has 0.4 GiB free between fields.

    The old fixed 2 GiB margin could never be met and every field waited the
    full timeout; the 0.2 GiB margin of a 2 GiB limit is met at once.
    """
    monkeypatch.setattr(MB, '_cgroup_limit_bytes', lambda: 2 * GIB)
    monkeypatch.setattr(MB, '_sched_declared_ram_bytes', lambda: 2 * GIB)
    monkeypatch.setattr(MB, '_available_ram_bytes', lambda: int(0.4 * GIB))
    slept = []
    for _ in range(5):
        assert MB._wait_for_ram(0, sleep=slept.append) == 0
    assert slept == []
    monkeypatch.setattr(MB, '_RAM_WAIT_SECONDS', 5.0)
    assert MB._wait_for_ram(0, margin=2 * GIB, sleep=slept.append) == 5.0


def test_the_batch_planner_uses_the_same_margin(monkeypatch):
    monkeypatch.setattr(MB, '_ram_limit', lambda: (2 * GIB, 'sched'))   # margin 0.2 GiB
    size = MB._largest_fitting_size(40, 0, int(0.5 * GIB), free_vram=None, free_ram=int(1.2 * GIB),
                                    margin_gb=0.0, tile_batch=0, model_loaded=True)
    assert size == 2
    assert MB._largest_fitting_size(40, 0, int(0.5 * GIB), free_vram=None, free_ram=12 * GIB,
                                    margin_gb=0.0, tile_batch=0, model_loaded=True,
                                    ram_margin=2 * GIB) == 20
