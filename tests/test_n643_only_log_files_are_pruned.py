"""The age pruner never deletes data with a log-like substring in its name."""
import os
import time
from pathlib import Path

import pytest

from spacr import run_journal as journal


@pytest.fixture
def logs(tmp_path, monkeypatch):
    folder = tmp_path / 'private-logs'
    folder.mkdir()
    monkeypatch.setattr(journal, '_prune_root', lambda kind: folder)
    return folder


def aged(folder, name):
    path = folder / name
    path.write_bytes(b'preserved private bytes')
    old = time.time() - 90 * 24 * 60 * 60
    os.utime(path, (old, old))
    return path


@pytest.mark.parametrize('name', ['spacr-results.log.csv', 'spacr.log-notes.txt', 'spacr.log.backup', 'spacr.log.1.csv'])
def test_log_like_data_never_enters_age_plan_and_is_refused_at_deletion(logs, name):
    data = aged(logs, name)
    plan = journal._prune_plan('logs', 30, 0)
    assert plan['delete'] == []
    plan['delete'].append((str(data), data.stat().st_size))
    deleted, freed, refused = journal._prune(plan)
    assert (deleted, freed) == (0, 0)
    assert refused
    assert data.read_bytes() == b'preserved private bytes'


def test_real_logs_and_numeric_rotations_remain_prunable(logs):
    names = ['spacr.log', 'spacr-info.log', 'spacr-20200101.log', 'spacr.log.1', 'spacr-info.log.25']
    paths = [aged(logs, name) for name in names]
    plan = journal._prune_plan('logs', 30, 0)
    assert {Path(raw).name for raw, size in plan['delete']} == set(names)
    deleted, freed, refused = journal._prune(plan)
    assert deleted == len(names) and freed > 0 and not refused
    assert all(not path.exists() for path in paths)
