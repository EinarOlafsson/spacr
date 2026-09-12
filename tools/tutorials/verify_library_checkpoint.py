#!/usr/bin/env python3
"""Reconcile every staged lesson, keeping explicit workflow holds visible.

This is not publication approval, a new full decode, a new GUI recording, or
native-language/listening review. It reuses the existing per-lesson verifier
and retained-media checks; unreported missing lessons are errors, not skips.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from audit_staged_catalogs import audit
from check_completed_matrix import check, digest, reconcile_browser, voice_matrix
from stage_lesson import DEFAULT_STAGE, REPO, read, write
from verify_retained_media import retained_media_sources


def require_partition(identities, held, retained):
    """Require explicit, non-overlapping dispositions for real catalog IDs."""
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate catalog lesson identity')
    if (not held <= set(identities) or not retained <= set(identities)
            or held & retained):
        raise ValueError('Held/retained lessons must be distinct catalog entries')
    return [key for key in identities if key not in held | retained]


def require_retention_receipt(saved, current):
    """A past decode must describe today's exact retained tracks and master."""
    if (saved.get('passed') is not True
            or saved.get('video_fully_decoded') is not True
            or saved.get('tracks_checked') != len(current['tracks'])
            or saved.get('media') != current['media']
            or saved.get('catalogs') != current['catalogs']):
        raise ValueError('Incomplete or stale retained media receipt')
    tracks = saved.get('tracks', [])
    if (len(tracks) != len(current['tracks'])
            or any(t.get('errors') != [] for t in tracks)
            or [{k: v for k, v in t.items() if k != 'errors'} for t in tracks]
            != current['tracks']):
        raise ValueError('Retained audio identities, hashes or decode results differ')


def verify(stage, held, *, retained=frozenset({'33_plate_viewer'}), baseline=None):
    stage = Path(stage).resolve()
    baseline = Path(baseline) if baseline is not None else REPO / 'docs/source/_extra/tutorials/catalog'
    catalog = read(stage / 'catalog/lessons_en.json')['lessons']
    by_id = {item['id']: item for item in catalog}
    selected = require_partition([item['id'] for item in catalog], held, retained)
    catalog_proof = audit(baseline, stage, set(selected),
                          visual_only={'34_database'} & set(selected))
    from build_navigation import build
    navigation = build({'lessons': catalog})
    inventory = voice_matrix(stage.parent / 'tools/render_all_voices.py')
    lessons = []
    for identity in selected:
        result = check(stage, identity, stage.parent / 'tools/render_all_voices.py',
                       retained_narration=identity == '34_database', baseline=baseline)
        lessons.append({'lesson': identity, 'scope': 'staged final bytes',
                        'tracks': result['unique_final_tracks'],
                        'browser_cases': len(result['browser_reports']),
                        'reconciliation': result})
        print(identity, 'PASS', flush=True)
    for identity in sorted(retained):
        current = retained_media_sources(stage, stage.parent, baseline, identity, inventory)
        saved = read(stage / 'retention' / identity / 'media-checks.json')
        require_retention_receipt(saved, current)
        browsers = []
        hashes = {(t['language'], t['voice']): t['audio_sha256'] for t in current['tracks']}
        cases = [(lang, voices[0], None) for lang, voices in inventory.items()]
        cases += [('en', 'af_heart', lang) for lang in ('da', 'de', 'is', 'ko', 'nb', 'sv')]
        for language, voice, caption in cases:
            tag = f'{language}-{voice}' + (f'-captions-{caption}' if caption else '')
            path = stage / 'browser' / identity / tag / 'playback-checks.json'
            report = read(path)
            reconcile_browser(report, identity, language, voice, caption,
                              hashes[(language, voice)], len(by_id[identity]['scenes']))
            if (report.get('whole_media_retained') is not True
                    or report.get('original_sources_unchanged') is not True):
                raise ValueError('Retained browser case did not verify unchanged sources')
            browsers.append({'case': tag, 'report_sha256': digest(path)})
        lessons.append({'lesson': identity, 'scope': 'unchanged original media',
                        'tracks': len(current['tracks']), 'browser_cases': len(browsers),
                        'retained_sources': current, 'browser_reports': browsers})
        print(identity, 'PASS (retained)', flush=True)
    return {'scope': 'Cross-library source and final-byte reconciliation only',
            'checked_lessons': len(lessons), 'catalog_lessons': len(catalog),
            'checked_tracks': sum(item['tracks'] for item in lessons),
            'browser_cases': sum(item['browser_cases'] for item in lessons),
            'held_lesson_ids': sorted(held), 'lessons': lessons,
            'catalog_preservation': catalog_proof, 'checked_subset_passed': True,
            'preservation_baseline': {'path': str(baseline.resolve()),
                'catalog_sha256': {p.name: digest(p) for p in sorted(baseline.glob('*.json'))}},
            'missing_registered_routes': navigation['missing_tutorials'],
            'navigation_source_commit': navigation['source_commit'],
            'whole_library_complete': not held and not navigation['missing_tutorials'],
            'new_decode_performed': False,
            'current_gui_revalidated': False, 'native_speaker_reviewed': False,
            'listening_reviewed': False, 'publication_approved': False,
            'published': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--held', nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path,
                        help='Frozen pre-refresh catalog directory; checks remain byte-exact')
    args = parser.parse_args()
    proof = verify(args.stage, set(args.held), baseline=args.baseline)
    write(args.output, proof)
    print(f"Checked {proof['checked_lessons']}/{proof['catalog_lessons']} lessons, "
          f"{proof['checked_tracks']} tracks, {proof['browser_cases']} browser cases; "
          f"held: {', '.join(proof['held_lesson_ids']) or 'none'}. Not publication approval.")
