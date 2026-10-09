"""Audio correction must not change scripts or existing offered voice sets."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_audio_patch_candidate as patch
from audit_staged_catalogs import CATALOGS
from stage_lesson import write


@pytest.fixture
def staged(tmp_path, monkeypatch):
    baseline, stage, repo = [tmp_path / name for name in ('baseline', 'stage', 'repo')]
    canonical = {'id': '07_mask', 'number': 7, 'title': 'Masks',
                 'scenes': [{'narration': 'Inspect the live preview.', 'visual': 'preview'}]}
    lesson = deepcopy(canonical)
    lesson['narration_voices'] = {'en': ['af_heart', 'bf_alice'], 'fr': ['ff_siwis']}
    for name in CATALOGS:
        write(baseline / 'web/catalog' / name, {'lessons': [lesson]})
        write(stage / 'catalog' / name, {'lessons': [lesson]})
    write(repo / 'tools/tutorials/lessons/07_mask.json', canonical)
    write(stage / 'production/07_mask/lesson.en.json', canonical)
    records = []
    for voice in lesson['narration_voices']['en']:
        audio = stage / 'production/07_mask/audio/en' / (voice + '.m4a')
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b'fixture corrected audio ' + voice.encode())
        timing = {'language': 'en', 'voice': voice,
                  'media_sha256': hashlib.sha256(audio.read_bytes()).hexdigest(),
                  'render_inputs': {'lesson': '07_mask', 'language': 'en', 'voice': voice,
                                    'scenes': canonical['scenes']}}
        write(audio.with_suffix('.json'), timing)
        for suffix in ('.m4a', '.json'):
            records.append({'path': f'media_host/07_mask/audio/en/{voice}{suffix}',
                            'sha256': 'original-' + voice, 'bytes': 1})
    write(baseline / 'release-manifest.json', {'files': records})
    monkeypatch.setattr(patch, 'REPO', repo)
    return baseline, stage, repo


def test_uses_existing_english_offerings_and_leaves_other_language_unstaged(staged):
    baseline, stage, _ = staged
    plan = patch.patch_plan(baseline, stage, ['07_mask'])
    assert [row['voice'] for row in plan] == ['af_heart', 'bf_alice']
    assert not (stage / 'production/07_mask/audio/fr').exists()
    assert all(row['audio_sha256'] != row['original_audio_sha256'] for row in plan)


@pytest.mark.parametrize('mutation', ['translation', 'canonical', 'extra_voice', 'missing_voice',
                                    'audio_identity', 'timing_narration', 'missing_original'])
def test_rejects_drift_before_any_candidate_is_created(staged, mutation):
    baseline, stage, repo = staged
    if mutation == 'translation':
        path = stage / 'catalog/lessons_fr.json'
        data = json.loads(path.read_text())
        data['lessons'][0]['title'] = 'Modified translation'
        write(path, data)
    elif mutation == 'canonical':
        path = repo / 'tools/tutorials/lessons/07_mask.json'
        data = json.loads(path.read_text())
        data['scenes'][0]['narration'] = 'Different workflow instructions.'
        write(path, data)
    elif mutation == 'extra_voice':
        (stage / 'production/07_mask/audio/en/am_new.m4a').write_bytes(b'extra')
    elif mutation == 'missing_voice':
        (stage / 'production/07_mask/audio/en/bf_alice.json').unlink()
    elif mutation in {'audio_identity', 'timing_narration'}:
        path = stage / 'production/07_mask/audio/en/af_heart.json'
        data = json.loads(path.read_text())
        if mutation == 'audio_identity':
            data['media_sha256'] = 'wrong'
        else:
            data['render_inputs']['scenes'][0]['narration'] = 'Altered speech.'
        write(path, data)
    elif mutation == 'missing_original':
        path = baseline / 'release-manifest.json'
        data = json.loads(path.read_text())
        data['files'] = data['files'][1:]
        write(path, data)
    with pytest.raises(ValueError):
        patch.patch_plan(baseline, stage, ['07_mask'])
    assert not list(stage.glob('release-candidate-*'))


@pytest.mark.parametrize('identities', [[], ['07_mask', '07_mask'], ['../07_mask'], ['88_absent']])
def test_rejects_invalid_selection(staged, identities):
    baseline, stage, _ = staged
    with pytest.raises(ValueError):
        patch.patch_plan(baseline, stage, identities)
