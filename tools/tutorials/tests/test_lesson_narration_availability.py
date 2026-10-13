"""Offer only verified tracks when a tutorial is published incrementally."""

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

PLAYER = Path(__file__).resolve().parents[1] / "authoring/web/app_v2.js"


def selectors(lesson, language="en", voice="af_bella"):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to exercise the actual player functions")
    source = PLAYER.read_text()
    functions = []
    for name in ("lessonVoices", "populateVoiceSelector", "populateNarrationLanguages"):
        match = re.search(r"function " + name + r"\([^)]*\)\s*\{[\s\S]*?\n\}", source)
        assert match, name
        functions.append(match.group())
    script = """
const input = INPUT;
let activeLesson = input.lesson;
const VOICE_CATALOG = [
 {id:'en', label:'English', voices:[
  {id:'af_heart', name:'Heart', variant:'American female'},
  {id:'af_bella', name:'Bella', variant:'American female'}]},
 {id:'es', label:'Spanish', voices:[
  {id:'ef_dora', name:'Dora', variant:'Female'}]}
];
const languageById = id => VOICE_CATALOG.find(item => item.id === id) || VOICE_CATALOG[0];
const document = {createElement: () => ({})};
function select() {return {value:'', options:[],
 set innerHTML(value) {this.options=[];}, appendChild(option) {this.options.push(option);}};}
const elements = {language:select(), voice:select()};
FUNCTIONS
populateNarrationLanguages(input.language);
populateVoiceSelector(input.voice);
console.log(JSON.stringify({language:elements.language, voice:elements.voice}));
""".replace("INPUT", json.dumps({"lesson": lesson, "language": language, "voice": voice}))
    script = script.replace("FUNCTIONS", "\n".join(functions))
    return json.loads(subprocess.check_output([node, "-e", script], text=True))


def test_existing_lessons_retain_all_voices_and_selection():
    result = selectors({"id": "existing"})
    assert [row["value"] for row in result["voice"]["options"]] == ["af_heart", "af_bella", "silent"]
    assert result["voice"]["value"] == "af_bella"
    assert result["language"]["options"][0]["textContent"] == "English · 2 voices"


def test_partial_lesson_offers_only_declared_voice_and_correct_count():
    result = selectors({"id": "new", "narration_voices": {"en": ["af_heart"]}})
    assert [row["value"] for row in result["voice"]["options"]] == ["af_heart", "silent"]
    assert result["voice"]["value"] == "af_heart"
    assert result["language"]["options"][0]["textContent"] == "English · 1 voice"


def test_unavailable_language_keeps_text_language_and_uses_silent_master():
    result = selectors({"id": "new", "narration_voices": {"en": ["af_heart"]}}, "es", "ef_dora")
    assert result["language"]["value"] == "es"
    assert result["voice"]["value"] == "silent"
    assert [row["value"] for row in result["voice"]["options"]] == ["silent"]
    assert result["language"]["options"][1]["textContent"] == "Spanish · silent"


@pytest.mark.parametrize("declaration", [None, {}, {"en": "af_heart"}, {"en": ["missing"]}])
def test_explicit_empty_or_invalid_availability_does_not_offer_missing_audio(declaration):
    result = selectors({"id": "new", "narration_voices": declaration})
    assert result["voice"]["value"] == "silent"
    assert [row["value"] for row in result["voice"]["options"]] == ["silent"]


def test_explicit_silent_preference_is_preserved():
    result = selectors({"id": "new", "narration_voices": {"en": ["af_heart"]}}, voice="silent")
    assert result["voice"]["value"] == "silent"
