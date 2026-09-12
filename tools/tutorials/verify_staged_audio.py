#!/usr/bin/env python3
"""Fully decode selected staged tracks and verify current synthesis fingerprints.

Use the existing release verifier with one lesson, not the full production
matrix. This is technical acceptance, not a listening or translation sign-off.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from stage_lesson import DEFAULT_STAGE, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lesson', required=True)
    parser.add_argument('--languages', nargs='+', required=True)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--retained-narration', action='store_true',
                        help='Require byte-identical original tracks and catalogs instead of a new runtime')
    args = parser.parse_args()
    os.environ['SPACR_TUTORIAL_WORKSPACE'] = str(args.stage.resolve())
    os.environ.setdefault('USE_TF', '0')
    sys.path.insert(0, str(DEFAULT_STAGE.parent / 'tools'))
    import verify_audio_release as verify

    if args.retained_narration:
        from retain_narration import retained_sources
        from stage_lesson import REPO
        retained_sources(args.stage, DEFAULT_STAGE.parent,
                         REPO / 'docs/source/_extra/tutorials/catalog', args.lesson,
                         {lang: values[1] for lang, values in verify.LANGUAGES.items()},
                         require_staged=True)

    reports = []
    for language in args.languages:
        code, voices = verify.LANGUAGES[language]
        catalog = read(args.stage / 'catalog' / f'lessons_{language}.json')
        lesson = next(item for item in catalog['lessons'] if item['id'] == args.lesson)
        for voice in voices:
            path = args.stage / 'production' / args.lesson / 'audio' / language / f'{voice}.m4a'
            if args.retained_narration:
                errors = verify.check_track(path)
            else:
                dialect_code = 'b' if language == 'en' and voice.startswith('b') else code
                dialect = verify.narration_dialect(language, dialect_code, voice)
                speed = verify.resolve_voice_speed(voice)
                plans = verify.prepare_scene_plans(lesson, language, dialect, speed, voice=voice)
                spec = verify.TrackSpec(path, lesson, language, dialect_code, voice, dialect, speed, plans)
                errors = verify.check_track(spec)
            record = {'language': language, 'voice': voice, 'errors': errors,
                      'audio_sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None}
            reports.append(record)
            print(f'{language}/{voice}: {len(errors)} errors', flush=True)
            if errors:
                print('\n'.join(errors), flush=True)
    result = {'lesson': args.lesson, 'scope_languages': args.languages,
              'retained_narration': args.retained_narration,
              'current_runtime_freshness_claimed': not args.retained_narration,
              'tracks_checked': len(reports), 'passed': all(not r['errors'] for r in reports),
              'human_listening_review': False, 'published': False, 'tracks': reports}
    write(args.stage / 'production' / args.lesson /
          f"audio-checks.{'+'.join(args.languages)}.json", result)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
