"""Unlist redirected lessons without discarding their archival Pages bytes."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from publish_release_candidate import retire_withdrawn_pages


@pytest.fixture
def archived_lesson(tmp_path):
    candidate=tmp_path/'candidate';pages=tmp_path/'pages'
    (candidate/'web').mkdir(parents=True)
    data=dict(lessons=[dict(id='02_install_spacr')],
              lesson_aliases={'02_conda_install':'02_install_spacr'})
    (candidate/'web/lesson_catalog.js').write_text(
        '"use strict";\nwindow.SPACR_LESSON_CATALOG = Object.freeze('+json.dumps(data)+');\n')
    old=pages/'production/02_conda_install'
    (old/'audio/en').mkdir(parents=True)
    (old/'poster.jpg').write_bytes(b'accepted archival poster')
    (old/'audio/en/af_heart.json').write_bytes(b'accepted archival timing')
    current=pages/'production/02_install_spacr';current.mkdir()
    (current/'poster.jpg').write_bytes(b'current poster')
    return candidate,pages,old,{'withdrawn_lessons':['02_conda_install']}


def test_redirected_pages_are_archived_and_active_media_remains(archived_lesson):
    candidate,pages,old,manifest=archived_lesson
    original={p.relative_to(old).as_posix():p.read_bytes() for p in old.rglob('*') if p.is_file()}
    records=retire_withdrawn_pages(candidate,pages,manifest)
    assert not old.exists()
    backup=candidate/'retired-pages/02_conda_install'
    assert {p.relative_to(backup).as_posix():p.read_bytes() for p in backup.rglob('*') if p.is_file()}==original
    assert len(records)==2
    assert (pages/'production/02_install_spacr/poster.jpg').read_bytes()==b'current poster'
    assert retire_withdrawn_pages(candidate,pages,manifest)==records


@pytest.mark.parametrize('identity',['02_install_spacr','03_unlinked','../02_conda_install'])
def test_only_declared_redirected_retirements_are_allowed(archived_lesson,identity):
    candidate,pages,old,manifest=archived_lesson
    manifest['withdrawn_lessons']=[identity]
    with pytest.raises(ValueError,match='redirects'):
        retire_withdrawn_pages(candidate,pages,manifest)
    assert old.exists()


def test_conflicting_archival_bytes_are_not_overwritten(archived_lesson):
    candidate,pages,old,manifest=archived_lesson
    backup=candidate/'retired-pages/02_conda_install';backup.mkdir(parents=True)
    (backup/'poster.jpg').write_bytes(b'earlier archival bytes')
    with pytest.raises(ValueError,match='different bytes'):
        retire_withdrawn_pages(candidate,pages,manifest)
    assert old.exists() and (old/'poster.jpg').read_bytes()==b'accepted archival poster'
    assert (backup/'poster.jpg').read_bytes()==b'earlier archival bytes'


def test_symlinked_retirement_is_rejected(archived_lesson):
    candidate,pages,old,manifest=archived_lesson
    (old/'unsafe').symlink_to(pages/'production/02_install_spacr',target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        retire_withdrawn_pages(candidate,pages,manifest)
    assert (pages/'production/02_install_spacr/poster.jpg').exists()
