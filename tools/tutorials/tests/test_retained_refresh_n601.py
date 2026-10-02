"""Visual refresh may retain only exact, fully checked published narration."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_appended_candidate as candidate


@pytest.fixture
def retained(tmp_path, monkeypatch):
    stage, baseline = tmp_path / 'stage', tmp_path / 'baseline'
    lesson = dict(id='04_platform_installers', number=4, app_key=None,
                  scenes=[dict(narration='Original words.', hold=3)])
    voices = {'en': ['af_heart'], 'ja': [f'voice{i}' for i in range(26)]}
    published_lesson = dict(lesson, narration_voices=voices)
    catalogs = {name: {'lessons': [deepcopy(published_lesson)]} for name in candidate.CATALOGS}
    for name, catalog in catalogs.items():
        candidate.write(baseline / 'web/catalog' / name, catalog)
        candidate.write(stage / 'catalog' / name, catalog)
    manifest = {'files': []}
    for language, items in voices.items():
        for voice in items:
            audio = f'Original {language}/{voice}'.encode()
            for root in (baseline / 'media_host', stage / 'production'):
                path = root / lesson['id'] / 'audio' / language / (voice + '.m4a')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(audio)
                candidate.write(path.with_suffix('.json'), dict(language=language, voice=voice,
                    media_sha256=candidate.digest(path), render_inputs=dict(lesson=lesson['id'],
                    language=language, voice=voice, scenes=lesson['scenes'], runtime='historical')))
            for suffix in ('.m4a', '.json'):
                path = (baseline / 'media_host' / lesson['id'] / 'audio' / language / (voice + suffix))
                manifest['files'].append(dict(path=path.relative_to(baseline).as_posix(),
                    sha256=candidate.digest(path), bytes=path.stat().st_size))
    decoded = []
    monkeypatch.setitem(sys.modules, 'verify_audio_release', SimpleNamespace(
        LANGUAGES={key: ('x', value) for key, value in voices.items()},
        check_track=lambda path: decoded.append(path) or []))
    return SimpleNamespace(stage=stage, baseline=baseline, lesson=lesson, catalogs=catalogs,
                           manifest=manifest, voices=voices, decoded=decoded)


def verify(data):
    return candidate.verify_retained_tracks(data.stage, data.baseline, data.lesson,
                                           data.catalogs, data.manifest)


def test_exact_27_track_retention_decodes_without_claiming_new_runtime(retained):
    voices, checks = verify(retained)
    assert voices == retained.voices
    assert len(checks) == len(retained.decoded) == 27
    assert all(isinstance(path, Path) for path in retained.decoded)
    assert all(row['audio_sha256'] and row['timing_sha256'] for row in checks)


@pytest.mark.parametrize('change', ['audio', 'timing', 'missing', 'extra', 'extra_timing', 'extra_nested',
                                    'catalog', 'canonical', 'hold', 'manifest', 'baseline'])
def test_retention_rejects_changed_content_and_inventory(retained, change):
    path = retained.stage / 'production/04_platform_installers/audio/en/af_heart.m4a'
    if change == 'audio':
        path.write_bytes(b'changed')
    elif change == 'timing':
        path.with_suffix('.json').write_text('{}')
    elif change == 'missing':
        path.unlink()
    elif change == 'extra_nested':
        nested = path.parent.parent / 'nested/en/af_heart.m4a'
        nested.parent.mkdir(parents=True)
        nested.write_bytes(path.read_bytes())
    elif change.startswith('extra'):
        path.with_name('unpublished' + ('.json' if change == 'extra_timing' else '.m4a')).write_text('x')
    elif change == 'catalog':
        filename = retained.stage / 'catalog/captions_ko.json'
        data = candidate.read(filename)
        data['lessons'][0]['scenes'][0]['narration'] = 'Changed translation'
        candidate.write(filename, data)
    elif change == 'canonical':
        retained.lesson['scenes'][0]['narration'] = 'Changed source'
    elif change == 'hold':
        retained.lesson['scenes'][0]['hold'] = 20
    elif change == 'manifest':
        retained.manifest['files'][0]['sha256'] = 'wrong'
    elif change == 'baseline':
        (retained.baseline / 'media_host/04_platform_installers/audio/en/af_heart.m4a').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        verify(retained)


def test_decode_failure_is_never_waived(retained, monkeypatch):
    monkeypatch.setattr(sys.modules['verify_audio_release'], 'check_track', lambda path: ['dead air'])
    with pytest.raises(ValueError, match='dead air'):
        verify(retained)


def test_equal_bytes_with_wrong_narration_identity_are_rejected(retained):
    for root in (retained.baseline / 'media_host', retained.stage / 'production'):
        path = root / '04_platform_installers/audio/en/af_heart.json'
        timing = candidate.read(path)
        timing['render_inputs']['scenes'][0]['narration'] = 'Other words'
        candidate.write(path, timing)
    record = next(row for row in retained.manifest['files'] if row['path'].endswith('/en/af_heart.json'))
    path = retained.baseline / record['path']
    record.update(sha256=candidate.digest(path), bytes=path.stat().st_size)
    with pytest.raises(ValueError, match='narration'):
        verify(retained)


def test_retained_catalog_objects_are_preserved_and_fresh_routing_unchanged(retained, monkeypatch):
    updated = dict(id='05_home')
    calls = []
    def update(*args, **kwargs):
        calls.append((args, kwargs))
        return 'fresh', ['review']
    monkeypatch.setattr(candidate, 'update_catalogs', update)
    result = candidate.selected_catalogs(retained.catalogs, [retained.lesson, updated], {}, {},
        ['04_platform_installers', '05_home'], ['04_platform_installers'], {})
    assert result == ('fresh', ['review'])
    assert calls[0][0][1] == [updated]
    assert calls[0][0][4] == ['05_home']
    assert candidate.selected_catalogs(retained.catalogs, [retained.lesson], {}, {},
        ['04_platform_installers'], ['04_platform_installers'], {}) == (retained.catalogs, [])
    assert len(calls) == 1


@pytest.mark.parametrize('retained_ids', [['04_platform_installers'], ['05_home', '05_home']])
def test_retention_cannot_bypass_fresh_append_or_duplicate_selection(tmp_path, retained_ids):
    with pytest.raises(ValueError, match='explicitly refreshed'):
        candidate.build(tmp_path, tmp_path, ['05_home'], retain_narration=retained_ids)
