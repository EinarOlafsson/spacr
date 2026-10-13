"""Retention requires both positive identity evidence and rejection of drift."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retain_narration import (CAPTION_LANGUAGES, digest, require_timing,
                             require_unchanged_lesson, retained_sources)
from stage_lesson import write


def test_equal_catalogs_pass_and_each_changed_source_fails():
    original = {'id': 'lesson', 'scenes': [{'narration': 'Original.'}]}
    require_unchanged_lesson(original, deepcopy(original), deepcopy(original))
    for index in range(3):
        sources = [deepcopy(original) for _ in range(3)]
        sources[index]['scenes'][0]['narration'] = 'Changed.'
        with pytest.raises(ValueError, match='Retained catalog differs'):
            require_unchanged_lesson(*sources)


@pytest.fixture
def timing():
    lesson = {'id': 'lesson', 'scenes': [{'narration': 'Original.'}]}
    record = {'language': 'en', 'voice': 'v', 'media_sha256': 'a' * 64,
              'render_inputs': {'lesson': 'lesson', 'language': 'en', 'voice': 'v',
                                'scenes': deepcopy(lesson['scenes'])}}
    return lesson, record


def test_timing_matches_exact_narration_and_media(timing):
    lesson, record = timing
    require_timing(record, lesson, 'en', 'v', 'a' * 64)


@pytest.mark.parametrize('change', [
    lambda r: r.update(language='fr'), lambda r: r.update(voice='other'),
    lambda r: r.update(media_sha256='b' * 64),
    lambda r: r['render_inputs'].update(lesson='wrong'),
    lambda r: r['render_inputs'].update(language='fr'),
    lambda r: r['render_inputs'].update(voice='other'),
    lambda r: r['render_inputs']['scenes'][0].update(narration='Changed.'),
    lambda r: r['render_inputs'].update(scenes=[]),
])
def test_wrong_narration_or_identity_is_rejected(timing, change):
    lesson, record = timing
    change(record)
    with pytest.raises(ValueError, match='Retained timing identity'):
        require_timing(record, lesson, 'en', 'v', 'a' * 64)


@pytest.fixture
def files(tmp_path, timing):
    stage, original, baseline = (tmp_path / name for name in ('stage', 'original', 'published'))
    lesson, record = timing
    for lang in ['en', *CAPTION_LANGUAGES]:
        filename = f"{'lessons' if lang == 'en' else 'captions'}_{lang}.json"
        for root in [stage / 'catalog', original / 'catalog', baseline]:
            write(root / filename, {'lessons': [lesson]})
    for root in [original, stage]:
        path = root / 'production/lesson/audio/en/v.m4a'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'actual fixture media')
        record['media_sha256'] = digest(path)
        write(path.with_suffix('.json'), record)
    return stage, original, baseline, 'lesson', {'en': ['v']}


def test_real_file_identity_positive_counterpart(files):
    result = retained_sources(*files, require_staged=True)
    assert len(result['tracks']) == 1 and len(result['catalogs']) == 7
    assert result['resynthesized'] is False


@pytest.mark.parametrize('suffix', ['.m4a', '.json'])
def test_changed_staged_bytes_rejected(files, suffix):
    path = files[0] / ('production/lesson/audio/en/v' + suffix)
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed in staging'):
        retained_sources(*files, require_staged=True)


def test_missing_staged_file_only_allowed_in_pre_copy_check(files):
    path = files[0] / 'production/lesson/audio/en/v.m4a'
    path.unlink()
    retained_sources(*files, require_staged=False)
    with pytest.raises(FileNotFoundError):
        retained_sources(*files, require_staged=True)


def test_missing_caption_lesson_rejected(files):
    write(files[0] / 'catalog/captions_sv.json', {'lessons': []})
    with pytest.raises(ValueError, match='exactly one'):
        retained_sources(*files, require_staged=True)


@pytest.mark.parametrize('identity', ['../lesson', '.', '..'])
def test_path_escape_rejected(files, identity):
    with pytest.raises(ValueError, match='private stage'):
        retained_sources(*files[:3], identity, files[4], require_staged=True)


def test_original_cannot_be_the_destination(files):
    with pytest.raises(ValueError, match='private stage'):
        retained_sources(files[1], *files[1:], require_staged=True)
