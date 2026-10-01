"""Only the first refreshed Heart CUDA sentence gets the listener's repair."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'authoring/tools'))
import render_all_voices as renderer
import verify_audio_release as verifier

TARGET = 'Here the request was auto and the selected backend is CUDA on NVIDIA hardware.'


def lesson(language='en'):
    catalog = json.loads((ROOT / 'release_candidate/web/catalog' / f'lessons_{language}.json').read_text())
    return next(item for item in catalog['lessons'] if item['id'] == '04_platform_installers')


def plans(item, language, voice=None):
    code = 'b' if language == 'en' and voice and voice.startswith('b') else renderer.LANGUAGES[language][0]
    dialect = renderer.narration_dialect(language, code, voice)
    speed = renderer.resolve_voice_speed(voice) if voice else 1.0
    return renderer.prepare_scene_plans(item, language, dialect, speed, voice=voice)


def lesson_speaking_target():
    """The installer lesson with the repaired sentence put back in one scene.

    The published lesson was rewritten as a CPU-only walkthrough and no longer
    says CUDA, so the repair is dormant there; this copy keeps its rule tested.
    """
    item = deepcopy(lesson())
    item['scenes'][4]['narration'] = TARGET + ' The installed Torch build is CUDA thirteen point two.'
    return item


def test_the_published_installer_lesson_no_longer_speaks_cuda_so_the_repair_is_dormant():
    spoken = [sentence for scene in plans(lesson(), 'en', 'af_heart') for sentence in scene['sentences']]
    assert spoken and not [sentence for sentence in spoken if 'CUDA' in sentence['text']]


def test_first_cuda_is_the_explicit_repair_target_when_the_lesson_speaks_it():
    spoken = [sentence for scene in plans(lesson_speaking_target(), 'en', 'af_heart')
              for sentence in scene['sentences']]
    cuda = [sentence for sentence in spoken if 'CUDA' in sentence['text']]
    assert len(cuda) == 2
    assert cuda[0]['text'] == TARGET
    assert '[CUDA](/kˈudə/)' in cuda[0]['speech_text']
    assert '[CUDA](/kˈuːdᵊ/)' in cuda[1]['speech_text']


def test_all_fifty_voice_plans_keep_every_other_sentence_and_setting():
    """No published voice plan is changed by the repair while the lesson does not say CUDA."""
    changed = []
    for language, (code, voices) in renderer.LANGUAGES.items():
        item = lesson(language)
        for voice in voices:
            dialect_code = 'b' if language == 'en' and voice.startswith('b') else code
            dialect = renderer.narration_dialect(language, dialect_code, voice)
            speed = renderer.resolve_voice_speed(voice)
            before = renderer.prepare_scene_plans(item, language, dialect, speed)
            after = renderer.prepare_scene_plans(item, language, dialect, speed, voice=voice)
            assert after == before, (language, voice)
            if after != before:
                changed.append((language, voice))
    assert changed == []


def test_the_repair_changes_only_the_target_sentence_of_the_heart_plan():
    item = lesson_speaking_target()
    dialect = renderer.narration_dialect('en', renderer.LANGUAGES['en'][0], 'af_heart')
    speed = renderer.resolve_voice_speed('af_heart')
    before = renderer.prepare_scene_plans(item, 'en', dialect, speed)
    after = renderer.prepare_scene_plans(item, 'en', dialect, speed, voice='af_heart')
    expected = deepcopy(before)
    target = next(s for p in expected for s in p['sentences'] if s['text'] == TARGET)
    target['speech_text'] = target['speech_text'].replace('[CUDA](/kˈuːdᵊ/)', '[CUDA](/kˈudə/)')
    assert after == expected != before


@pytest.mark.parametrize('field,value', [
    (0, '03_pip_install'), (1, 'fr'), (2, 'af_aoede'),
    (3, 'The installed Torch build is CUDA thirteen point two.'),
])
def test_neighbouring_lesson_language_voice_and_second_mention_are_unchanged(field, value):
    request = ['04_platform_installers', 'en', 'af_heart', TARGET]
    source = 'selected back end is [CUDA](/kˈuːdᵊ/) on NVIDIA hardware.'
    assert renderer.track_speech_text(*request, source) != source
    request[field] = value
    assert renderer.track_speech_text(*request, source) == source


def test_a_changed_pronunciation_premise_is_refused():
    with pytest.raises(ValueError, match='premise changed'):
        renderer.track_speech_text('04_platform_installers', 'en', 'af_heart', TARGET, 'CUDA')


@pytest.mark.parametrize('voices', [('af_heart', 'am_puck'), ('am_puck', 'af_heart')])
def test_release_verifier_does_not_reuse_a_different_voices_cached_plan(tmp_path, voices):
    # These real voices share dialect AND speed: omitting voice from the
    # cache key must actually collide, not pass because their speeds differ.
    assert renderer.resolve_voice_speed(voices[0]) == renderer.resolve_voice_speed(voices[1])
    path = tmp_path / 'lessons_en.json'
    path.write_text(json.dumps({'lessons': [lesson_speaking_target()]}))
    specs = verifier.supported_track_specs(production=tmp_path, catalog_path=path,
                                          languages={'en': ('a', list(voices))})
    by_voice = {spec.voice: spec for spec in specs}
    for voice in voices:
        assert by_voice[voice].scene_plans == plans(lesson_speaking_target(), 'en', voice)
    first = lambda spec: next(s['speech_text'] for p in spec.scene_plans for s in p['sentences'] if s['text'] == TARGET)
    assert first(by_voice['af_heart']) != first(by_voice['am_puck'])


def test_candidate_uses_its_own_heart_audio_and_all_native_sentence_cues():
    """The 2026-09-12 CUDA repair was retired with the CUDA sentence; the
    published Heart track is the candidate's own render, checked in the browser
    sentence by sentence against the lesson it narrates."""
    repair = ROOT / 'audio_repairs/platform-heart-refreshed-20260912'
    retired = hashlib.sha256((repair / 'af_heart.m4a').read_bytes()).hexdigest()
    manifest = json.loads((ROOT / 'release_candidate/release-manifest.json').read_text())
    path = 'media_host/04_platform_installers/audio/en/af_heart.m4a'
    record = next(item for item in manifest['files'] if item['path'] == path)
    assert record['sha256'] != retired
    report = json.loads((ROOT / 'release_candidate/candidate-browser-checks.json').read_text())
    case = next(item for item in report['ready_playback_cases'] if item['lesson'] == '04_platform_installers')
    assert case['passed'] is True and case['audio_sha256'] == record['sha256']
    sentences = [s['text'] for scene in plans(lesson(), 'en', 'af_heart') for s in scene['sentences']]
    checks = case['sentence_cue_checks']
    assert sentences and [observed['text'] for observed in checks] == sentences
    for observed in checks:
        assert abs(observed['audio'] - observed['requested_audio_time']) < 1
        assert observed['text'] in observed['cues']
