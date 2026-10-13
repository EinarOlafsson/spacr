"""Exercise the shipped chapter builder against audio/catalog text drift."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
PLAYER = ROOT / 'docs/source/_extra/tutorials/app_v2.js'


def chapters(*, language='en', timing_language='en', timing_voice='af_heart', text='Spoken Windows instructions.'):
    source = PLAYER.read_text()
    function = source.split('function rebuildChapterData()', 1)[1].split('\nfunction relatedLessonsForScene', 1)[0]
    setup = {'captionLanguage': language, 'timingLanguage': timing_language,
             'timingVoice': timing_voice, 'text': text}
    script = '''
const input = INPUT;
const activeLesson = {id: '04_platform_installers'};
const elements = {voice: {value: 'af_heart'}};
const audioTimings = {language: input.timingLanguage, voice: input.timingVoice,
 scenes: [{scene: 1, speech_start: 26.3, speech_end: 30.05, text: input.text}]};
const effectiveCaptionLanguage = () => input.captionLanguage;
const captionLesson = () => ({scenes: [{narration: 'Catalog or translated instructions.'}]});
const relatedLessonsForScene = () => [];
const chapterLabel = text => text;
let chapterData = [];
function rebuildChapterData() BODY
rebuildChapterData();
console.log(JSON.stringify(chapterData));
'''.replace('INPUT', json.dumps(setup)).replace('BODY', function)
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed to execute the actual tutorial player')
    return json.loads(subprocess.check_output([node, '-e', script], text=True))


def test_same_language_captions_and_transcript_follow_the_selected_audio():
    result = chapters()[0]
    assert result['text'] == 'Spoken Windows instructions.'
    assert result['label'] == result['text']
    assert (result['start'], result['end']) == (26.3, 30.05)


@pytest.mark.parametrize('kwargs', [
    {'language': 'de'}, {'timing_language': 'fr'},
    {'timing_voice': 'af_bella'}, {'text': ''}, {'text': None},
])
def test_translations_or_unmatched_metadata_do_not_become_english_transcripts(kwargs):
    assert chapters(**kwargs)[0]['text'] == 'Catalog or translated instructions.'
