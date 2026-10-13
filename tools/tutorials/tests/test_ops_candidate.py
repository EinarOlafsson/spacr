"""OPS is the verified four-tile geometry lesson, not a claimed full pipeline."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from published_lesson import check_published_lesson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ops_capture_keeps_the_real_geometry_and_unfinished_workflow_boundary():
    report = read(ROOT / 'evidence/2026-09-12_ops_final_checks.json')
    assert report['published'] is False and report['full_ops_pipeline_completed'] is False
    assert report['gui_workflow_completed'] is False
    capture = report['capture']
    assert capture['accepted'] is True and capture['gui']['run_clicked'] is False
    run = capture['terminal']['run']
    assert run['accepted'] is True
    assert run['placed'] == run['accepted_edges'] == 4
    assert run['canvas'] == [2756, 2756]
    assert run['full_pipeline_completed'] is False
    assert run['segmentation_or_decoding_performed'] is False


def test_ops_published_voices_and_heart_captions_are_the_candidates():
    """The twelve-scene native walkthrough replaced the 2026-09-12 recording."""
    check_published_lesson('76_ops', 12)
