"""Which tutorial lessons are archived: hidden from the library, kept whole.

An archived lesson keeps its identity, folder and media paths. Its deployed
catalog entries, navigation places and related-lesson links are stored under
``tools/tutorials/archive/`` by ``archive_lessons.py``, which also restores
them to their original positions. Tests and generators read
:data:`ARCHIVED_IDS` from here rather than listing lesson identities
themselves, so archiving or restoring one lesson needs no other edit.

The displayed lesson number is the position in the visible list, never the
stored ``number`` field, so the library numbers stay contiguous.
"""
from __future__ import annotations

import json
from pathlib import Path

ARCHIVE_DIR = Path(__file__).resolve().with_name('archive')
ARCHIVE_INDEX = ARCHIVE_DIR / 'archived_lessons.json'
ARCHIVE_LESSONS = ARCHIVE_DIR / 'lessons'


def load_index(path: Path | None = None) -> dict:
    """The archive index, or an empty one when nothing has been archived."""
    path = path or ARCHIVE_INDEX
    if not path.is_file():
        return {'schema': 1, 'lessons': {}}
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') != 1 or not isinstance(data.get('lessons'), dict):
        raise ValueError(f'{path}: unsupported tutorial archive index')
    return data


def archived_ids(path: Path | None = None) -> tuple[str, ...]:
    """Archived lesson identities, in their original catalog order."""
    lessons = load_index(path)['lessons']
    return tuple(sorted(lessons, key=lambda identity: (lessons[identity].get('position', 0),
                                                       identity)))


ARCHIVED_IDS: tuple[str, ...] = archived_ids()


def archived_app_keys(path: Path | None = None) -> dict[str, str]:
    """``{app_key: lesson id}`` for archived lessons that teach one module."""
    lessons = load_index(path)['lessons']
    return {entry['app_key']: identity for identity, entry in lessons.items()
            if entry.get('app_key')}


def is_archived(identity: str) -> bool:
    """Whether ``identity`` (a lesson id such as ``24_plaque``) is archived."""
    return identity in ARCHIVED_IDS


def visible(identities):
    """``identities`` without the archived ones, order kept."""
    return [identity for identity in identities if identity not in ARCHIVED_IDS]
