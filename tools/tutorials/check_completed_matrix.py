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

from stage_lesson import DEFAULT_STAGE, read, write

CAPTION_LANGUAGES = ('da', 'de', 'is', 'ko', 'nb', 'sv')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def voice_matrix(renderer):
    """Read the renderer's literal inventory without importing TTS/GPU code."""
    tree = ast.parse(Path(renderer).read_text())
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


def check(stage, lesson_id, renderer):
    if Path(lesson_id).name != lesson_id or lesson_id in {'.', '..'}:
        raise ValueError('Expected one lesson identity')
    stage = Path(stage).resolve()
    folder = stage / 'production' / lesson_id
    english = read(folder / 'lesson.en.json')
    canonical = hashlib.sha256(json.dumps(english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    inventory = voice_matrix(renderer)
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
            if translated != english:
                raise ValueError('Staged English differs from the rendered lesson')
        else:
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
    visual = read(folder / 'visual.json')
    if len(visual['scenes']) != len(english['scenes']):
        raise ValueError('Visual scene count differs')
    for scene in visual['scenes']:
        path = (folder / scene['image']).resolve()
        if not path.is_relative_to(stage) or digest(path) != scene['capture_sha256']:
            raise ValueError('The captured image changed or leaves staging')
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
    return {'lesson': lesson_id, 'passed': True, 'canonical_english_sha256': canonical,
            'scene_count': len(english['scenes']), 'unique_final_tracks': len(tracks),
            'tracks': [{'language': key[0], 'voice': key[1], 'sha256': value}
                       for key, value in sorted(tracks.items())],
            'browser_reports': browser, 'master_sha256': digest(master), 'master_probe': probe,
            'poster_sha256': digest(folder / 'poster.jpg'),
            'scope': 'Final artifact reconciliation, not a new decode, editorial or listening review',
            'native_speaker_signoff': False, 'human_listening_review': False, 'published': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lesson', required=True)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    args = parser.parse_args()
    result = check(args.stage, args.lesson, DEFAULT_STAGE.parent / 'tools/render_all_voices.py')
    write(args.stage / 'production' / args.lesson / 'final-artifact-checks.json', result)
    print(f"{args.lesson}: {result['unique_final_tracks']} final tracks, "
          f"{len(result['browser_reports'])} browser checks, matching sources; not published.")
