"""The actual PSF speech plan keeps captions and other lessons unchanged."""
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import render_all_voices as renderer  # noqa: E402


@pytest.mark.parametrize("language,voice", [("ja", "jf_alpha"), ("zh-CN", "zf_xiaobei")])
def test_actual_seven_scenes_speak_controls_without_latin_leakage(language, voice):
    review = json.loads((ROOT.parent / "lessons/reviews" / f"86_psf_workflow.{language}.json").read_text())
    lesson = {"id": "86_psf_workflow", "scenes": [{"narration": text} for text in review["scenes"]]}
    before = deepcopy(lesson)
    plans = renderer.prepare_scene_plans(lesson, language, "us", voice=voice)
    assert lesson == before
    assert [plan["display_text"] for plan in plans] == review["scenes"]
    speech = " ".join(sentence["speech_text"] for plan in plans for sentence in plan["sentences"])
    assert not re.search(r"[A-Za-z]", speech)
    assert ("コンペア" if language == "ja" else "比较") in speech
    assert ("オリジナル" if language == "ja" else "原始图像") in speech
    assert ("プロセスト" if language == "ja" else "处理后图像") in speech
    assert ("リチャードソン・ルーシー" if language == "ja" else "理查德森露西") in speech
    if language == "zh-CN":
        assert "制作掩膜编辑器" in speech
        assert "批量掩膜模块" in speech


@pytest.mark.parametrize("language,code,voice,text,expected", [
    ("ja", "j", "jf_alpha", "Make Masks で Compare をクリックします。",
     "039aca1e96e673373be898805017e967ad582688c1896582b626295b8866fee3"),
    ("zh-CN", "z", "zf_xiaobei", "在 Make Masks 中点击 Compare。",
     "7d1147c590705520201c51a7a7511ebd4bc6b3e336b43788768135ce14bfbaec"),
])
def test_existing_lesson_fingerprint_matches_before_psf_change(language, code, voice, text, expected):
    # Captured from the normal renderer before adding the lesson86 branch.
    lesson = {"id": "14_make_masks", "scenes": [{"narration": text, "hold_after": 0.7}]}
    plans = renderer.prepare_scene_plans(lesson, language, "us", voice=voice)
    actual, _ = renderer.track_fingerprint(lesson, language, code, "us", voice, 1.0, plans)
    assert actual == expected


def test_other_language_and_other_lesson_keep_their_existing_speech():
    text = "Choose Convolution in Make Masks and click Compare."
    assert renderer.track_speech_text("86_psf_workflow", "en", "af_heart", text, text) == text
    assert renderer.track_speech_text("14_make_masks", "zh-CN", "zf_xiaobei", text, text) == text


@pytest.mark.parametrize("display,speech", [("Compare", "unrelated"), ("NewControl", "NewControl")])
def test_changed_control_or_speech_requires_review(display, speech):
    with pytest.raises(ValueError, match="pronunciation premise|Unreviewed Latin"):
        renderer.track_speech_text("86_psf_workflow", "zh-CN", "zf_xiaobei", display, speech)
