"""Require actual final-media and recording evidence before promoting OPS.

Catalog metadata alone cannot turn the placeholder into a ready lesson.
This is a private packaging gate, not approval to deploy the tutorial.
"""
import hashlib
import json
from pathlib import Path

from check_completed_matrix import check, digest
from stage_lesson import read


def require_recorded_ops(stage, language, lesson):
    if stage is None:
        raise ValueError('OPS promotion requires verified recording and final media, not metadata alone')
    stage = Path(stage).resolve()
    identity = '76_ops'
    prefix = 'captions' if language in {'da', 'de', 'is', 'ko', 'nb', 'sv'} else 'lessons'
    current = [item for item in read(stage / 'catalog' / f'{prefix}_{language}.json')['lessons']
               if item['id'] == identity]
    if current != [lesson]:
        raise ValueError('OPS promotion must match the exact verified language catalog')
    # This checks all 50 final tracks, complete decode/master receipts, source-
    # pinned scripts, captured pixels, and all fourteen browser language cases.
    matrix = check(stage, identity, stage.parent / 'tools/render_all_voices.py')
    english = read(stage / 'production' / identity / 'lesson.en.json')
    canonical = hashlib.sha256(json.dumps(english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if (matrix.get('passed') is not True or matrix.get('unique_final_tracks') != 50
            or matrix.get('canonical_english_sha256') != canonical
            or len(matrix.get('browser_reports', [])) != 14):
        raise ValueError('OPS promotion lacks complete current final-media verification')
    capture = read(stage / 'captures/ops_1507_verified/scientific_acceptance.json')
    gui, terminal = capture['gui'], capture['terminal']
    if (capture.get('accepted') is not True or gui.get('accepted') is not True
            or terminal.get('accepted') is not True
            or gui.get('run_clicked') is not False
            or terminal.get('source_unchanged') is not True
            or any(item.get('gui_workflow_completed') is not False for item in (capture, gui, terminal))):
        raise ValueError('OPS promotion needs actual GUI navigation and the separately scoped API recording')
    hashes = capture['source_hashes']
    if not hashes:
        raise ValueError('OPS recording has no source evidence')
    for name, expected in hashes.items():
        path = Path(name).resolve(strict=True)
        if not path.is_relative_to(stage) or digest(path) != expected:
            raise ValueError('OPS recorded evidence changed or leaves staging')
    run = terminal['run']
    if (run.get('accepted') is not True or run.get('placed') != 4
            or run.get('accepted_edges') != 4 or run.get('canvas') != [2756, 2756]
            or run.get('full_pipeline_completed') is not False
            or run.get('segmentation_or_decoding_performed') is not False):
        raise ValueError('OPS recording does not demonstrate the authored four-tile scope')
    return matrix
