#!/usr/bin/env python3
"""Reconcile existing technical reports against the final staged media bytes.

This does not replace decoding, browser checks, editorial review or listening.
It prevents an incomplete/stale combination of those artifacts being counted as
a completed matrix. Run after writers and verifiers have stopped for a lesson.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

from stage_lesson import DEFAULT_STAGE, REPO, read, write

CAPTION_LANGUAGES = ('da', 'de', 'is', 'ko', 'nb', 'sv')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def voice_matrix(renderer):
    """Read the renderer's literal inventory without importing TTS/GPU code."""
    tree = ast.parse(Path(renderer).read_text(encoding="utf-8"))
    matches = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'LANGUAGES'
                       for target in node.targets)]
    if len(matches) != 1:
        raise ValueError('Expected exactly one renderer language inventory')
    inventory = ast.literal_eval(matches[0])
    result = {language: item[1] for language, item in inventory.items()}
    if (set(result) != {'en', 'es', 'fr', 'hi', 'it', 'ja', 'pt-BR', 'zh-CN'}
            or sum(map(len, result.values())) != 50
            or any(len(voices) != len(set(voices)) for voices in result.values())):
        raise ValueError('The preserved eight-language/fifty-voice matrix changed')
    return result


def reconcile_audio(reports, lesson, expected, hashes):
    found = {}
    for report in reports:
        if (report.get('lesson') != lesson or report.get('passed') is not True
                or report.get('tracks_checked') != len(report.get('tracks', []))):
            raise ValueError('Incomplete, failed or wrong-lesson audio report')
        for record in report['tracks']:
            key = (record['language'], record['voice'])
            if (key not in expected or record.get('errors') != []
                    or record.get('audio_sha256') != hashes.get(key)):
                raise ValueError('Failed, unexpected or stale audio track')
            found[key] = record['audio_sha256']
    if set(found) != set(expected):
        raise ValueError('The audio reports do not cover every preserved voice')
    return found


def reconcile_browser(report, lesson, language, voice, caption, audio_hash, scenes):
    if (report.get('lesson') != lesson or report.get('passed') is not True
            or report.get('loaded_audio_sha256') != audio_hash
            or report.get('scene_count') != scenes
            or report.get('scope') != f'{language}/{voice} playback and scene links only'
            or report.get('navigation_contains_staged_lesson') is not True):
        raise ValueError('Incomplete, wrong-route or stale browser report')
    if (report.get('caption_language') != caption or (caption is not None and
            (report.get('caption_scenes_match_staging') is not True
             or len(report.get('caption_webvtt_sha256', '')) != 64))):
        raise ValueError('Missing or incorrect independent caption evidence')
    clock = report['seek_playback_clocks']
    if (clock.get('mediaError') is not None
            or not all(isinstance(clock.get(key), (int, float)) and math.isfinite(clock[key])
                       for key in ('audio', 'video', 'expectedVideo', 'audioDuration', 'videoDuration'))
            or min(clock['audioDuration'], clock['videoDuration']) <= 0
            or abs(clock['video'] - clock['expectedVideo']) >= .5):
        raise ValueError('Invalid or unsynchronised browser clocks')


def reconcile_english_catalog(translated, english, inventory, *, retained_narration=False, published_voice_order=False):
    """Allow only the normal complete planned-voice declaration as metadata."""
    compared, authored = dict(translated), dict(english)
    if (retained_narration or published_voice_order) and 'narration_voices' not in compared:
        raise ValueError('Retained English needs its bound fifty-voice matrix')
    for item in (compared, authored):
        if 'narration_voices' in item:
            declared = item.pop('narration_voices')
            if retained_narration or published_voice_order:
                # The caller has already checked immutable published catalogs,
                # all source tracks and exact current canonical lesson bytes.
                valid = (isinstance(declared, dict) and set(declared) == set(inventory)
                    and all(isinstance(voices, list)
                            and all(isinstance(voice, str) for voice in voices)
                            and len(voices) == len(set(voices))
                            and set(voices) == set(inventory[language])
                            for language, voices in declared.items()))
            else:
                valid = isinstance(declared, dict) and declared == inventory
            if not valid:
                raise ValueError('English planned voices differ from the normal fifty-voice matrix')
    if compared != authored:
        raise ValueError('Staged English differs from the rendered lesson')


def reconcile_visual_sources(stage, lesson_id, english):
    """Keep raw frame checks strict; reuse only a fully bound published master."""
    stage = Path(stage).resolve()
    folder = stage / 'production' / lesson_id
    specification = folder / 'scenes.json'
    staged_record=folder/'staged-native-edit.json'
    staged_mode=specification.is_file() and read(specification).get('kind')=='edited_staged_native_intervals'
    if staged_mode or staged_record.exists():
        if (not staged_mode or not staged_record.is_file() or any((folder/name).exists()
                for name in ['published-native-edit.json','published-native-mix.json','published-native-reuse.json'])):
            raise ValueError('Edited staged footage requires its exclusive accepted-source record')
        from published_native_edit import checked_staged_native_timing
        return read(checked_staged_native_timing(stage,lesson_id))['staged_native_edit']
    mix_mode = (specification.is_file() and
        read(specification).get('kind') == 'mixed_published_native_intervals')
    mix_record = folder / 'published-native-mix.json'
    if mix_mode or mix_record.exists():
        if (not mix_mode or not mix_record.is_file()
                or any((folder / name).exists() for name in ['published-native-edit.json','published-native-reuse.json'])):
            raise ValueError('Mixed published sources need their exclusive source/card/interval record')
        from published_native_mix import checked_mixed_native_timing
        return read(checked_mixed_native_timing(stage, lesson_id))['published_native_mix']
    edit_mode = (specification.is_file() and
        read(specification).get('kind') in {'edited_published_native_intervals',
                                           'terminal_prefix_published_native_intervals'})
    edit_record = folder / 'published-native-edit.json'
    if edit_mode or edit_record.exists():
        if (not edit_mode or not edit_record.is_file()
                or (folder / 'published-native-reuse.json').exists()):
            raise ValueError('Edited published footage requires its exclusive source/interval record')
        from published_native_edit import checked_edited_native_timing
        # This revalidates the exact current 50 render inputs/audio hashes,
        # immutable published source, reviewed frame edits and rendered master.
        timing = read(checked_edited_native_timing(stage, lesson_id))
        return timing['published_native_edit']
    interval_mode = (specification.is_file() and
        read(specification).get('kind') == 'published_native_intervals')
    reuse_record = folder / 'published-native-reuse.json'
    if interval_mode or reuse_record.exists():
        if not interval_mode or not reuse_record.is_file():
            raise ValueError('Published-native reuse needs its explicit intervals and source record')
        from native_live_timing import checked_native_timing
        timing = read(checked_native_timing(stage, lesson_id))
        return timing['published_native_reuse']
    visual = read(folder / 'visual.json')
    if len(visual['scenes']) != len(english['scenes']):
        raise ValueError('Visual scene count differs')
    for scene in visual['scenes']:
        path = (folder / scene['image']).resolve()
        if not path.is_relative_to(stage) or digest(path) != scene['capture_sha256']:
            raise ValueError('The captured image changed or leaves staging')
    return None



def _terminal_published_voice_order(stage, lesson_id, *, retained_narration=False):
    """Admit published voice ordering only after the explicit terminal proof."""
    root=Path(stage)/'production'/lesson_id
    audio=root/'published-terminal-scene-removal.json'
    spec=root/'scenes.json';visual=root/'published-native-edit.json'
    kind='terminal_prefix_published_native_intervals'
    requested=(audio.exists() or audio.is_symlink()
        or (spec.is_file() and read(spec).get('kind')==kind)
        or (visual.is_file() and read(visual).get('kind')==kind))
    if not requested:return False
    if (lesson_id!='05_home' or retained_narration or not audio.is_file() or audio.is_symlink()
            or not spec.is_file() or spec.is_symlink() or read(spec).get('kind')!=kind
            or not visual.is_file() or visual.is_symlink() or read(visual).get('kind')!=kind):
        raise ValueError('Terminal published ordering requires its exclusive Home audio/visual proof')
    from published_terminal_prefix import checked_terminal_narration
    tracks,proofs=checked_terminal_narration(stage,lesson_id,decode=False)
    if len(tracks)!=50 or len(proofs)!=50:
        raise ValueError('Terminal published ordering requires all fifty checked tracks')
    return True


def check(stage, lesson_id, renderer, *, retained_narration=False, baseline=None, retained_from=None):
    if retained_from is not None and not retained_narration:
        raise ValueError('Use retained-from only with retained-narration')
    if Path(lesson_id).name != lesson_id or lesson_id in {'.', '..'}:
        raise ValueError('Expected one lesson identity')
    stage = Path(stage).resolve()
    folder = stage / 'production' / lesson_id
    english = read(folder / 'lesson.en.json')
    canonical = hashlib.sha256(json.dumps(english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    inventory = voice_matrix(renderer)
    published_voice_order = _terminal_published_voice_order(stage, lesson_id, retained_narration=retained_narration)
    retention = None
    if retained_narration:
        from retain_narration import retained_sources
        from stage_lesson import REPO
        original = Path(retained_from) if retained_from is not None else Path(renderer).resolve().parents[1]
        from published_retained_narration import RECORD, checked_published_narration
        if (folder / RECORD).exists() or (folder / RECORD).is_symlink():
            if retained_from is not None:
                raise ValueError('Published retention has its own immutable source, not retained-from')
            checked_published_narration(stage, lesson_id)
            retention = read(folder / RECORD)
        else:
            retention = retained_sources(stage, original,
                Path(baseline) if baseline is not None else REPO / 'docs/source/_extra/tutorials/catalog', lesson_id, inventory,
                require_staged=True)
    expected = {(language, voice) for language, voices in inventory.items() for voice in voices}
    hashes = {key: digest(folder / 'audio' / key[0] / (key[1] + '.m4a')) for key in expected}
    for language in [*inventory, *CAPTION_LANGUAGES]:
        prefix = 'captions' if language in CAPTION_LANGUAGES else 'lessons'
        matches = [item for item in read(stage / 'catalog' / f'{prefix}_{language}.json')['lessons']
                   if item['id'] == lesson_id]
        if len(matches) != 1 or len(matches[0]['scenes']) != len(english['scenes']):
            raise ValueError('Catalog identity or scene count differs')
        translated = matches[0]
        if language == 'en':
            reconcile_english_catalog(translated, english, inventory, retained_narration=retained_narration,
                                      published_voice_order=published_voice_order)
        elif not retained_narration:
            review = read(folder / f'review.{language}.json')
            if (review.get('english_sha256') != canonical or review.get('lesson') != lesson_id
                    or review.get('language') != language
                    or review.get('scenes') != [scene['narration'] for scene in translated['scenes']]
                    or any(review.get(key) != translated.get(key) for key in
                           ('title', 'description', 'prerequisite', 'objectives'))):
                raise ValueError('Translation review is stale or differs from the catalog')
        for voice in inventory.get(language, []):
            timing = read(folder / 'audio' / language / (voice + '.json'))
            inputs = timing['render_inputs']
            if (inputs.get('lesson') != lesson_id or inputs.get('voice') != voice
                    or inputs.get('language') != language or timing.get('media_sha256') != hashes[(language, voice)]
                    or [scene['narration'] for scene in inputs['scenes']] !=
                       [scene['narration'] for scene in translated['scenes']]):
                raise ValueError('Audio timing identity or narration is stale')
    reports = sorted(folder.glob('audio-checks.*.json'))
    tracks = reconcile_audio([read(path) for path in reports], lesson_id, expected, hashes)
    browser = []
    cases = [(language, voices[0], None) for language, voices in inventory.items()]
    cases += [('en', 'af_heart', language) for language in CAPTION_LANGUAGES]
    for language, voice, caption in cases:
        tag = f'{language}-{voice}' + (f'-captions-{caption}' if caption else '')
        path = stage / 'browser' / lesson_id / tag / 'playback-checks.json'
        reconcile_browser(read(path), lesson_id, language, voice, caption,
                          hashes[(language, voice)], len(english['scenes']))
        browser.append({'case': tag, 'report_sha256': digest(path)})
    reuse = reconcile_visual_sources(stage, lesson_id, english)
    master = folder / 'video' / f'{lesson_id}_silent.mp4'
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration,size:stream=codec_type,width,height,r_frame_rate,nb_frames',
        '-of', 'json', str(master)], text=True))
    streams = probe['streams']
    if (len(streams) != 1 or streams[0]['codec_type'] != 'video'
            or [streams[0]['width'], streams[0]['height']] != [3840, 2160]
            or streams[0]['r_frame_rate'] != '30/1'):
        raise ValueError('Expected the single silent 4K/30 master')
    result = {'lesson': lesson_id, 'passed': True, 'canonical_english_sha256': canonical,
            'scene_count': len(english['scenes']), 'unique_final_tracks': len(tracks),
            'tracks': [{'language': key[0], 'voice': key[1], 'sha256': value}
                       for key, value in sorted(tracks.items())],
            'browser_reports': browser, 'master_sha256': digest(master), 'master_probe': probe,
            'poster_sha256': digest(folder / 'poster.jpg'),
            'scope': 'Final artifact reconciliation, not a new decode, editorial or listening review',
            'retained_narration': retention,
            'native_speaker_signoff': False, 'human_listening_review': False, 'published': False}
    if reuse is not None:
        key = 'staged_native_edit' if (folder / 'staged-native-edit.json').exists() else 'published_native_mix' if (folder / 'published-native-mix.json').exists() else 'published_native_edit' if (folder / 'published-native-edit.json').exists() else 'published_native_reuse'
        result[key] = reuse
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lesson', required=True)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--retained-narration', action='store_true',
                        help='Require exact original/published catalogs and unchanged original audio/timings')
    parser.add_argument('--retained-from', type=Path,
                        help='Original workspace for retained catalogs/audio; requires retained-narration')
    args = parser.parse_args(argv)
    result = check(args.stage, args.lesson, REPO / 'tools/tutorials/authoring/tools/render_all_voices.py',
                   retained_narration=args.retained_narration, retained_from=args.retained_from)
    write(args.stage / 'production' / args.lesson / 'final-artifact-checks.json', result)
    print(f"{args.lesson}: {result['unique_final_tracks']} final tracks, "
          f"{len(result['browser_reports'])} browser checks, matching sources; not published.")


if __name__ == '__main__':
    main()
