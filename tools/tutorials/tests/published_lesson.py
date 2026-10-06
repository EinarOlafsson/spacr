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
#: The unchanged lessons in this cohort retain this accepted voice set.
VOICE_BASELINE = ROOT / 'evidence/2026-10-04-rerecord-wave4-publication.json'
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
    checkpoint = read(CANDIDATE / 'checkpoint.json')
    assert checkpoint['manifest_sha256'] == manifest_sha
    receipt = read(CANDIDATE / 'publication-receipt.json')
    assert receipt['manifest_sha256'] == manifest_sha and receipt['readback']['passed'] is True
    assert checkpoint['media_uploaded'] and checkpoint['pages_tree_ready']
    assert checkpoint['release_hold'] is False
    for key in ('repository', 'branch', 'tag', 'commit', 'media_root'):
        assert checkpoint['media_revision'][key] == receipt[key]
    assert receipt['readback']['commit'] == receipt['commit']
    assert receipt['media_root'].endswith('/resolve/' + receipt['commit'])
    publication = read(VOICE_BASELINE)

    catalog = read(CANDIDATE / 'web/catalog/lessons_en.json')
    lesson = next(item for item in catalog['lessons'] if item['id'] == identity)
    assert lesson.get('status') != 'coming_soon'
    assert [scene['narration'] for scene in lesson['scenes']] == [
        scene['narration'] for scene in source['scenes']]

    records = {item['path']: item for item in read(manifest_path)['files']}
    tracks = {tuple(path.split('/')[3:5]) for path in records
              if path.startswith(f'media_host/{identity}/audio/') and path.endswith('.m4a')}
    # These unchanged lessons retain the accepted baseline voices.
    sets = {tuple(map(tuple, pairs)) for pairs in publication['narration_voices'].values()}
    assert len(sets) == 1, 'publication lessons ship different voice sets'
    voices = {(language, voice + '.m4a') for language, voice in sets.pop()}
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
    hosted = read(CANDIDATE / 'published-media-browser-checks.json')
    assert hosted['manifest_sha256'] == manifest_sha and hosted['passed'] is True
    assert hosted['media_root'] == receipt['media_root']
    assert hosted['index_sha256'] == sha(ROOT.parents[1] / 'docs/source/_extra/tutorials/index.html')
    hosted_cases = [item for item in hosted['ready_playback_cases'] if item['lesson'] == identity]
    assert len(hosted_cases) == 1 and hosted_cases[0]['passed'] is True
    assert hosted_cases[0]['audio_sha256'] == heart['sha256']
    dialect = renderer.narration_dialect('en', renderer.LANGUAGES['en'][0], 'af_heart')
    plans = renderer.prepare_scene_plans(lesson, 'en', dialect,
                                         renderer.resolve_voice_speed('af_heart'), voice='af_heart')
    sentences = [sentence['text'] for plan in plans for sentence in plan['sentences']]
    checks = case['sentence_cue_checks']
    assert len(sentences) > scenes and [item['text'] for item in checks] == sentences
    for observed in checks:
        assert abs(observed['audio'] - observed['requested_audio_time']) < 1
        assert observed['text'] in observed['cues']
