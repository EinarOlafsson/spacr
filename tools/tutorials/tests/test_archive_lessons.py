"""Archiving hides lessons losslessly; restoring returns them to their places."""
from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import archive_lessons  # noqa: E402
import archived  # noqa: E402

DEPLOYED = archive_lessons.TREES['docs']
CHECKPOINT = HERE / 'release_candidate' / 'web'


def test_the_deployed_library_is_its_publication_record_minus_the_archived_lessons():
    record = archive_lessons.load_tree(CHECKPOINT)
    deployed = archive_lessons.load_tree(DEPLOYED)
    assert archive_lessons.archived_view(record) == deployed
    for name, data in deployed.items():
        if name == archive_lessons.NAVIGATION_JS:
            continue
        ids = [lesson['id'] for lesson in data['lessons']]
        assert not set(archived.ARCHIVED_IDS) & set(ids), name
        for lesson in data['lessons']:
            for scene in lesson['scenes']:
                assert not set(archived.ARCHIVED_IDS) & set(scene.get('related_lessons', []))


def test_putting_every_archived_lesson_back_rebuilds_the_publication_record():
    record = archive_lessons.load_tree(CHECKPOINT)
    full = archive_lessons.full_view(archive_lessons.load_tree(DEPLOYED))
    for name in record:
        if name == archive_lessons.NAVIGATION_JS:
            continue
        assert full[name] == record[name], name


def test_every_archived_lesson_has_its_record_and_media_paths():
    index = archived.load_index()
    assert tuple(sorted(index['lessons'], key=lambda i: index['lessons'][i]['position'])) \
        == archived.ARCHIVED_IDS
    for identity, entry in index['lessons'].items():
        record = json.loads((archived.ARCHIVE_LESSONS / f'{identity}.json').read_text(encoding='utf-8'))
        assert record['id'] == identity
        catalogs = record['trees']['docs']['catalogs']
        assert len(catalogs) == 15 and all(c['entry']['id'] == identity for c in catalogs.values())
        assert entry['hf']['archive_path'] == f'archive/{identity}/'
        assert entry['local_copy']


def _sandbox(tmp_path, monkeypatch):
    tree = tmp_path / 'tutorials'
    names = [archive_lessons.CATALOG_JS, archive_lessons.NAVIGATION_JS, *archive_lessons.CATALOG_JSON]
    for name in names:
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CHECKPOINT / name, tree / name)
    monkeypatch.setattr(archive_lessons, 'TREES', {'docs': tree})
    monkeypatch.setattr(archived, 'ARCHIVE_DIR', tmp_path / 'archive')
    monkeypatch.setattr(archived, 'ARCHIVE_INDEX', tmp_path / 'archive' / 'archived_lessons.json')
    monkeypatch.setattr(archived, 'ARCHIVE_LESSONS', tmp_path / 'archive' / 'lessons')
    monkeypatch.setattr(archived, 'ARCHIVED_IDS', ())
    monkeypatch.setattr(archive_lessons, '_move_posters', lambda *args: [])
    monkeypatch.setattr(archive_lessons, '_prose', lambda *args: [])
    monkeypatch.setattr(archive_lessons, '_regenerate', lambda: None)
    monkeypatch.setattr(archive_lessons, '_git', lambda *args: 'test')
    return tree, {name: (tree / name).read_bytes() for name in names}


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_restoring_in_any_order_returns_every_byte(tmp_path, monkeypatch, capsys, seed):
    tree, before = _sandbox(tmp_path, monkeypatch)
    assert archive_lessons.main(['archive', '--apply']) == 0
    archived.ARCHIVED_IDS = archived.archived_ids()
    english = json.loads((tree / 'catalog/lessons_en.json').read_text(encoding='utf-8'))
    assert len(english['lessons']) == 84 - len(archive_lessons.APPROVED_1_5_1_6)
    pending = list(archived.ARCHIVED_IDS)
    random.Random(seed).shuffle(pending)
    while pending:
        batch = [pending.pop() for _ in range(min(len(pending), 3))]
        assert archive_lessons.main(['restore', *batch, '--apply']) == 0
        archived.ARCHIVED_IDS = archived.archived_ids()
    for name, data in before.items():
        if name == archive_lessons.NAVIGATION_JS:
            restored = archive_lessons._read_js(tree / name)
            original = json.loads(data.decode()[data.decode().index('(') + 1:data.decode().rindex(')')])
            assert {**restored, 'routes': dict(sorted(restored['routes'].items()))} == \
                {**original, 'routes': dict(sorted(original['routes'].items()))}
        else:
            assert (tree / name).read_bytes() == data, name


def test_the_default_run_is_a_dry_run(tmp_path, monkeypatch, capsys):
    tree, before = _sandbox(tmp_path, monkeypatch)
    assert archive_lessons.main(['archive']) == 0
    assert 'dry run' in capsys.readouterr().out
    assert all((tree / name).read_bytes() == data for name, data in before.items())
    assert not archived.ARCHIVE_INDEX.exists()
