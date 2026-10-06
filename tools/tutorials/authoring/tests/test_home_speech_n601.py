"""Home pronunciation fixes preserve captions and previously accepted tracks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import render_all_voices as renderer


def lesson(identity, language):
    source = json.loads((ROOT.parent / 'lessons' / (identity + '.json')).read_text())
    if language != 'en':
        review = json.loads((ROOT.parent / 'lessons/reviews' / f'{identity}.{language}.json').read_text())
        assert len(review['scenes']) == len(source['scenes'])
        for scene, text in zip(source['scenes'], review['scenes']):
            scene['narration'] = text
    return source


@pytest.mark.parametrize('language,voice', [('ja','jf_alpha'), ('zh-CN','zf_xiaobei')])
def test_all_current_source_bound_scenes_keep_captions_and_speak_controls(language, voice):
    source = lesson('05_home', language)
    original = deepcopy(source)
    plans = renderer.prepare_scene_plans(source, language, 'us', voice=voice)
    assert len(plans) == 22 and source == original
    assert [p['display_text'] for p in plans] == [s['narration'] for s in source['scenes']]
    speech = ' '.join(s['speech_text'] for p in plans for s in p['sentences'])
    assert not re.search('[A-Za-z]', speech)
    for required in (['プリファレンシズ','テーマ','アニメーション','タビュラー'] if language == 'ja'
                     else ['偏好设置','主题','动画','制作掩膜编辑器','批量掩膜模块','弓形虫']):
        assert required in speech
    # Plasmodium and Candida are alpha species (634): Home no longer names them.
    for withdrawn in ('疟原虫', '念珠菌', 'プラスモディウム', 'カンジダ'):
        assert withdrawn not in speech


@pytest.mark.parametrize('language,expected', [
    ('ja', 'ラン コンペア、トレーニング ランズ、ラン。'),
    ('zh-CN', '运行比较、训练记录、运行。')])
def test_longest_labels_precede_exact_standalone_run(language, expected):
    text = 'Run Compare、Training Runs、Run。'
    assert renderer.track_speech_text('05_home', language, None, text, text) == expected


@pytest.mark.parametrize('language,text,expected', [
    ('ja', 'Embeddings', 'エンベディングス'),
    ('ja', 'Power / Design', 'パワー アンド デザイン'),
    ('ja', 'Dose-Response', 'ドーズ レスポンス'),
    ('zh-CN', 'Embeddings', '嵌入特征'),
    ('zh-CN', 'Power / Design', '功效与设计'),
    ('zh-CN', 'Dose-Response', '剂量反应'),
])
def test_new_home_data_controls_have_reviewed_spoken_names(language, text, expected):
    assert renderer.track_speech_text('05_home', language, None, text, text) == expected


@pytest.mark.parametrize('display,speech', [('Runway', 'Runway'), ('Run', 'unrelated'),
                                          ('ThemeNew', 'ThemeNew')])
def test_unknown_labels_or_changed_pronunciation_premise_require_review(display, speech):
    with pytest.raises(ValueError, match='premise changed|Unreviewed Latin'):
        renderer.track_speech_text('05_home', 'ja', 'jf_alpha', display, speech)


@pytest.mark.parametrize('identity,count,expected', [
    ('05_home', 37, '904f1ac00722e9d443c52f41a37615ffc69a005fc139221e0dc2ebd941477862')])
def test_unaffected_track_fingerprints_match_before_home_change(identity, count, expected):
    # Captured before this Home-only branch; no runtime/model imports needed.
    historical = json.loads((Path(__file__).parent / 'fixtures' / 'home-n601-before.json').read_text())
    records = []
    for language, (code, voices) in renderer.LANGUAGES.items():
        if identity == '05_home' and language in {'ja','zh-CN'}:
            continue
        source = historical['lessons'][language]
        for voice in voices:
            actual_code = 'b' if language == 'en' and voice.startswith('b') else code
            dialect = renderer.narration_dialect(language, actual_code, voice)
            speed = renderer.resolve_voice_speed(voice)
            plans = renderer.prepare_scene_plans(source, language, dialect, speed, voice=voice)
            fingerprint, _ = renderer.track_fingerprint(source, language, actual_code, dialect, voice, speed, plans)
            records.append(dict(lesson=identity, language=language, voice=voice, fingerprint=fingerprint))
    assert len(records) == count
    assert hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest() == expected
