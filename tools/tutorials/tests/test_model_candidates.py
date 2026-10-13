"""Both model lessons have actual scoped recordings and fully checked media.

The 2026-09-12 evidence describes the first recordings; both lessons were
rewritten as walkthroughs and republished, so their published media is
checked against the current candidate.
"""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_promotion import validate_scope
sys.path.insert(0, str(Path(__file__).resolve().parent))
from published_lesson import check_published_lesson  # noqa: E402

LESSONS = [('21_model_compare', 6), ('22_model_zoo', 13)]
FIRST_RECORDINGS = [('21_model_compare', 8), ('22_model_zoo', 7)]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('identity,scenes', FIRST_RECORDINGS)
def test_first_recordings_kept_their_actual_scope_and_disclosures(identity, scenes):
    label = identity.split('_', 1)[1]
    report = read(ROOT / 'evidence' / f'2026-09-12_{label}_final_checks.json')
    assert report['lesson'] == identity and report['published'] is False
    assert report['gui_inference_or_benchmark_completed'] is False
    assert report['native_speaker_signoff'] is False and report['human_listening_signoff'] is False
    validate_scope(identity, report['capture'])
    matrix = report['matrix']
    assert matrix['passed'] is True and matrix['scene_count'] == scenes
    assert matrix['unique_final_tracks'] == len(matrix['tracks']) == 50


@pytest.mark.parametrize('identity,scenes', LESSONS)
def test_published_model_lesson_voices_and_heart_captions_are_the_candidates(identity, scenes):
    check_published_lesson(identity, scenes)
