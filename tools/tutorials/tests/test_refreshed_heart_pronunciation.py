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


def test_first_cuda_in_current_refreshed_english_is_the_explicit_repair_target():
    item = lesson()
    spoken = [sentence for scene in plans(item, 'en', 'af_heart') for sentence in scene['sentences']]
    cuda = [sentence for sentence in spoken if 'CUDA' in sentence['text']]
    assert len(cuda) == 2
    assert cuda[0]['text'] == TARGET
    assert '[CUDA](/kˈudə/)' in cuda[0]['speech_text']
    assert '[CUDA](/kˈuːdᵊ/)' in cuda[1]['speech_text']


def test_all_fifty_voice_plans_keep_every_other_sentence_and_setting():
    changed = []
    for language, (code, voices) in renderer.LANGUAGES.items():
        item = lesson(language)
        for voice in voices:
            dialect_code = 'b' if language == 'en' and voice.startswith('b') else code
            dialect = renderer.narration_dialect(language, dialect_code, voice)
            speed = renderer.resolve_voice_speed(voice)
            before = renderer.prepare_scene_plans(item, language, dialect, speed)
            after = renderer.prepare_scene_plans(item, language, dialect, speed, voice=voice)
            expected = deepcopy(before)
            if (language, voice) == ('en', 'af_heart'):
                target = next(s for p in expected for s in p['sentences'] if s['text'] == TARGET)
                target['speech_text'] = target['speech_text'].replace('[CUDA](/kˈuːdᵊ/)', '[CUDA](/kˈudə/)')
            assert after == expected, (language, voice)
            if after != before:
                changed.append((language, voice))
    assert changed == [('en', 'af_heart')]


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
    path.write_text(json.dumps({'lessons': [lesson()]}))
    specs = verifier.supported_track_specs(production=tmp_path, catalog_path=path,
                                          languages={'en': ('a', list(voices))})
    by_voice = {spec.voice: spec for spec in specs}
    for voice in voices:
        assert by_voice[voice].scene_plans == plans(lesson(), 'en', voice)
    first = lambda spec: next(s['speech_text'] for p in spec.scene_plans for s in p['sentences'] if s['text'] == TARGET)
    assert first(by_voice['af_heart']) != first(by_voice['am_puck'])


def test_candidate_uses_the_verified_refreshed_audio_and_all_native_sentence_cues():
    repair = ROOT / 'audio_repairs/platform-heart-refreshed-20260912'
    metadata = json.loads((repair / 'af_heart.json').read_text())
    audio_hash = hashlib.sha256((repair / 'af_heart.m4a').read_bytes()).hexdigest()
    assert metadata['media_sha256'] == audio_hash
    manifest = json.loads((ROOT / 'release_candidate/release-manifest.json').read_text())
    for suffix in ('m4a', 'json'):
        path = f'media_host/04_platform_installers/audio/en/af_heart.{suffix}'
        record = next(item for item in manifest['files'] if item['path'] == path)
        assert record['sha256'] == hashlib.sha256((repair / f'af_heart.{suffix}').read_bytes()).hexdigest()
    report = json.loads((ROOT / 'release_candidate/candidate-browser-checks.json').read_text())
    case = next(item for item in report['ready_playback_cases'] if item['lesson'] == '04_platform_installers')
    assert case['audio_sha256'] == audio_hash
    sentences = [s for scene in metadata['scenes'] for s in scene['sentences']]
    checks = case['sentence_cue_checks']
    assert sentences and len(checks) == len(sentences)
    for sentence, observed in zip(sentences, checks):
        midpoint = (sentence['speech_start'] + sentence['speech_end']) / 2
        assert observed['requested_audio_time'] == pytest.approx(midpoint)
        assert abs(observed['audio'] - midpoint) < 1
        assert observed['text'] == sentence['text']
        assert sentence['text'] in observed['cues']
