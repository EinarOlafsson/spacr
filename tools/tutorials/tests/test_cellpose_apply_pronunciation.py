"""The measured Cellpose Japanese onset repair changes one exact spoken unit."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / 'authoring/tools'))
import render_all_voices as renderer
from sentence_audio_cache import cached_sentence

TARGET = 'Applyタブを開きます。Custom modelは、学習させたチェックポイントを指しています。Sourceを例のapplyフォルダーにします。その画像は、学習に使っていないウェルから撮ったものです。Channelsをゼロ、Percentilesを二と九十九にします。接している細胞が分かれたままになるようFill inをオフにし、SaveとVerboseをオンにします。'


def test_cellpose_apply_replaces_only_the_leading_label():
    actual = renderer.track_speech_text('19_train_cellpose', 'ja', 'jf_tebukuro', TARGET, TARGET)
    assert actual == 'アプライ' + TARGET[len('Apply'):]
    assert '例のapplyフォルダー' in actual


@pytest.mark.parametrize('identity,language,voice,text', [
    ('58_graph_builder', 'ja', 'jf_tebukuro', TARGET),
    ('19_train_cellpose', 'en', 'jf_tebukuro', TARGET),
    ('19_train_cellpose', 'ja', 'jf_alpha', TARGET),
    ('19_train_cellpose', 'ja', 'jf_tebukuro', TARGET + '変更'),
])
def test_cellpose_apply_other_requests_remain_identical(identity, language, voice, text):
    assert renderer.track_speech_text(identity, language, voice, text, text) == text


def test_cellpose_apply_rejects_a_changed_spoken_premise():
    with pytest.raises(ValueError, match='pronunciation premise changed'):
        renderer.track_speech_text('19_train_cellpose', 'ja', 'jf_tebukuro', TARGET, '別の発音')


def test_cellpose_apply_changes_only_one_unit_across_the_actual_fifty_voice_plans():
    english = json.loads((TOOLS / 'lessons/19_train_cellpose.json').read_text())
    changed = []
    for language, (code, voices) in renderer.LANGUAGES.items():
        lesson = deepcopy(english)
        if language != 'en':
            review = json.loads((TOOLS / f'lessons/reviews/19_train_cellpose.{language}.json').read_text())
            for scene, text in zip(lesson['scenes'], review['scenes']):
                scene['narration'] = text
        for voice in voices:
            vcode = 'b' if language == 'en' and voice.startswith('b') else code
            dialect = renderer.narration_dialect(language, vcode, voice)
            speed = renderer.resolve_voice_speed(voice)
            before = renderer.prepare_scene_plans(lesson, language, dialect, speed)
            after = renderer.prepare_scene_plans(lesson, language, dialect, speed, voice=voice)
            expected = deepcopy(before)
            if (language, voice) == ('ja', 'jf_tebukuro'):
                assert expected[10]['sentences'][0]['text'] == TARGET
                expected[10]['sentences'][0]['speech_text'] = 'アプライ' + TARGET[len('Apply'):]
            assert after == expected
            for number, (old, new) in enumerate(zip(before, after), 1):
                for unit, (a, b) in enumerate(zip(old['sentences'], new['sentences']), 1):
                    if a != b:
                        changed.append((language, voice, number, unit))
    assert changed == [('ja', 'jf_tebukuro', 11, 1)]


def test_cellpose_apply_normal_cache_reuses_the_neighbour_and_only_synthesizes_the_new_unit(tmp_path):
    calls = []
    def synthesize():
        calls.append(True)
        return np.array([0.1, -0.1], dtype=np.float32), 'actual-test-unit'
    common = dict(schema=1, speed=1.0, language='ja', lang_code='j', dialect='ja',
                  voice='jf_tebukuro', sample_rate=24000, synthesis_runtime={'test': 'exact-runtime'})
    old = dict(common, speech_text=TARGET)
    neighbour = dict(common, speech_text='変更しない次の文。')
    for inputs in (old, neighbour):
        assert cached_sentence(tmp_path, inputs, synthesize, np)[2] is False
    calls.clear()
    repaired = dict(common, speech_text=renderer.track_speech_text(
        '19_train_cellpose', 'ja', 'jf_tebukuro', TARGET, TARGET))
    assert cached_sentence(tmp_path, repaired, synthesize, np)[2] is False
    assert cached_sentence(tmp_path, neighbour, synthesize, np)[2] is True
    assert cached_sentence(tmp_path, old, synthesize, np)[2] is True
    assert len(calls) == 1
