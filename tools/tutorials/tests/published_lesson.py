"""What a lesson's media must satisfy in the current release candidate.

The 2026-09-12/13 final-check evidence for Model Compare, Model Zoo, OPS,
Embeddings and Map Barcodes describes recordings that were later rewritten
and republished; their tracks and browser cases are no longer in the
candidate, so tests that bound that evidence to "the final candidate" went
stale. The checks here bind the lesson to the candidate that is published
now (``publication-receipt.json``), from the source lesson to each voice and
each Heart sentence the browser observed.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'release_candidate'
#: The newest publication evidence; it names the voices every lesson ships.
PUBLICATION = ROOT / 'evidence/2026-10-03-rerecord-wave2-publication.json'
sys.path.insert(0, str(ROOT / 'authoring/tools'))
import render_all_voices as renderer  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_published_lesson(identity, scenes):
    """Assert ``identity`` is published, source-current and fully verified.

    :param identity: the lesson id.
    :param scenes: the scene count of the current English source.
    """
    source = read(ROOT / 'lessons' / (identity + '.json'))
    assert len(source['scenes']) == scenes
    manifest_path = CANDIDATE / 'release-manifest.json'
    manifest_sha = sha(manifest_path)
    assert read(CANDIDATE / 'checkpoint.json')['manifest_sha256'] == manifest_sha
    receipt = read(CANDIDATE / 'publication-receipt.json')
    assert receipt['manifest_sha256'] == manifest_sha and receipt['readback']['passed'] is True
    publication = read(PUBLICATION)
    assert publication['commit'] == receipt['commit'] and publication['tag'] == receipt['tag']

    catalog = read(CANDIDATE / 'web/catalog/lessons_en.json')
    lesson = next(item for item in catalog['lessons'] if item['id'] == identity)
    assert lesson.get('status') != 'coming_soon'
    assert [scene['narration'] for scene in lesson['scenes']] == [
        scene['narration'] for scene in source['scenes']]

    records = {item['path']: item for item in read(manifest_path)['files']}
    tracks = {tuple(path.split('/')[3:5]) for path in records
              if path.startswith(f'media_host/{identity}/audio/') and path.endswith('.m4a')}
    voices = {(language, voice + '.m4a')
              for language, voice in publication['narration_voices']['08_measure']}
    assert tracks == voices and ('en', 'af_heart.m4a') in tracks
    for language, name in tracks:
        timing = f'media_host/{identity}/audio/{language}/{name[:-4]}.json'
        assert timing in records, timing
    assert f'media_host/{identity}/video/{identity}_silent.mp4' in records

    browser = read(CANDIDATE / 'candidate-browser-checks.json')
    assert browser['manifest_sha256'] == manifest_sha and browser['passed'] is True
    cases = [item for item in browser['ready_playback_cases'] if item['lesson'] == identity]
    assert len(cases) == 1 and cases[0]['passed'] is True
    case = cases[0]
    heart = records[f'media_host/{identity}/audio/en/af_heart.m4a']
    assert case['audio_sha256'] == heart['sha256']
    dialect = renderer.narration_dialect('en', renderer.LANGUAGES['en'][0], 'af_heart')
    plans = renderer.prepare_scene_plans(lesson, 'en', dialect,
                                         renderer.resolve_voice_speed('af_heart'), voice='af_heart')
    sentences = [sentence['text'] for plan in plans for sentence in plan['sentences']]
    checks = case['sentence_cue_checks']
    assert len(sentences) > scenes and [item['text'] for item in checks] == sentences
    for observed in checks:
        assert abs(observed['audio'] - observed['requested_audio_time']) < 1
        assert observed['text'] in observed['cues']
