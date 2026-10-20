"""Preserved-media sharing keeps verified bytes and atomic replacement isolation."""
import errno
import hashlib
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_appended_candidate import copy_preserved_checked, copy_preserved_web
from build_release_candidate import copy_checked


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('enabled', [False, True])
def test_preserved_copy_records_verified_bytes_and_opt_in_inode(tmp_path, enabled):
    baseline = tmp_path / 'baseline.mp4'
    baseline.write_bytes(b'published immutable media')
    root = tmp_path / 'candidate'
    target = root / 'media_host/05_home/video/master.mp4'
    records = []
    copy_preserved_checked(baseline, target, records, root, sha(baseline), hardlink=enabled)
    assert (baseline.stat().st_ino == target.stat().st_ino) is enabled
    assert target.read_bytes() == baseline.read_bytes()
    assert records == [{'path': 'media_host/05_home/video/master.mp4',
                        'sha256': sha(baseline), 'bytes': baseline.stat().st_size}]


@pytest.mark.parametrize('replace_baseline', [False, True])
def test_atomic_copy_replacement_does_not_mutate_other_link(tmp_path, replace_baseline):
    baseline = tmp_path / 'baseline.mp4'
    baseline.write_bytes(b'verified baseline')
    root = tmp_path / 'candidate'
    target = root / 'media_host/master.mp4'
    copy_preserved_checked(baseline, target, [], root, sha(baseline), hardlink=True)
    replacement = tmp_path / 'new.mp4'
    replacement.write_bytes(b'new rendition bytes')
    replaced, preserved = (baseline, target) if replace_baseline else (target, baseline)
    copy_checked(replacement, replaced, [], tmp_path, sha(replacement))
    assert replaced.read_bytes() == b'new rendition bytes'
    assert preserved.read_bytes() == b'verified baseline'
    assert replaced.stat().st_ino != preserved.stat().st_ino


def test_cross_device_falls_back_to_verified_copy(tmp_path, monkeypatch):
    baseline = tmp_path / 'baseline.mp4'
    baseline.write_bytes(b'verified baseline')
    root = tmp_path / 'candidate'
    target = root / 'media_host/master.mp4'
    calls = []
    def cross_device(source, destination):
        calls.append((source, destination))
        raise OSError(errno.EXDEV, 'different filesystems')
    monkeypatch.setattr(os, 'link', cross_device)
    records = []
    copy_preserved_checked(baseline, target, records, root, sha(baseline), hardlink=True)
    assert calls == [(baseline, target)]
    assert target.read_bytes() == baseline.read_bytes()
    assert target.stat().st_ino != baseline.stat().st_ino
    assert records[0]['sha256'] == sha(baseline)


def test_stale_baseline_is_rejected_before_linking(tmp_path):
    baseline = tmp_path / 'baseline.mp4'
    baseline.write_bytes(b'changed after receipt')
    root = tmp_path / 'candidate'
    target = root / 'media_host/master.mp4'
    records = []
    with pytest.raises(ValueError, match='Source changed'):
        copy_preserved_checked(baseline, target, records, root, 'stale', hardlink=True)
    assert not target.exists()
    assert records == []


def test_only_verified_production_is_linked_and_replaced_lesson_is_excluded(tmp_path):
    baseline = tmp_path / 'baseline'
    published = tmp_path / 'published'
    published.mkdir()
    (published / 'index.html').write_bytes(b'generated player')
    records = []
    for identity in ('05_home', '07_mask'):
        relative = f'web/production/{identity}/poster.jpg'
        source = baseline / relative
        source.parent.mkdir(parents=True)
        source.write_bytes(identity.encode())
        records.append({'path': relative, 'sha256': sha(source), 'bytes': source.stat().st_size})
    root = tmp_path / 'candidate'
    copy_preserved_web(published, baseline, root, {'files': records},
                       replacements=['07_mask'], hardlink_preserved=True)
    poster = root / records[0]['path']
    assert poster.stat().st_ino == (baseline / records[0]['path']).stat().st_ino
    assert not (root / records[1]['path']).exists()
    assert (root / 'web/index.html').stat().st_ino != (published / 'index.html').stat().st_ino


def test_linked_target_is_independently_verified(tmp_path, monkeypatch):
    baseline = tmp_path / 'baseline.mp4'
    baseline.write_bytes(b'verified baseline')
    wrong = tmp_path / 'wrong.mp4'
    wrong.write_bytes(b'wrong inode')
    root = tmp_path / 'candidate'
    target = root / 'media_host/master.mp4'
    real_link = os.link
    monkeypatch.setattr(os, 'link', lambda source, destination: real_link(wrong, destination))
    records = []
    with pytest.raises(ValueError, match='Linked bytes differ'):
        copy_preserved_checked(baseline, target, records, root, sha(baseline), hardlink=True)
    assert records == []
    assert baseline.read_bytes() == b'verified baseline'
