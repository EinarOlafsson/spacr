"""Refuse a staged lesson whose narration audio starts a sentence late or cut.

For every voice track and every sentence: the 100 ms before ``audible_start``
must be silent and the first 150 ms after it must carry speech. A sentence
shifted against its recorded start or cut at a join fails one of the tests.
The browser twin (verify_staged_lesson --native-boundary-clocks) refuses any
narration pause or seek at a scene boundary.
usage: check_narration_onsets.py --stage STAGE --lesson LESSON
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np

RATE = 24000


def decode(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-ac', '1', '-ar', str(RATE),
                          '-f', 'f32le', '-'], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def check_track(audio_path, meta, *, min_onset=0.003, silence=0.01):
    x = decode(audio_path)
    problems = []
    for scene in meta['scenes']:
        for sentence in scene.get('sentences') or [scene]:
            start = float(sentence.get('audible_start', sentence['speech_start']))
            i = int(round(start * RATE))
            before = rms(x[max(0, i - int(0.1 * RATE)):i])
            onset = rms(x[i:i + int(0.15 * RATE)])
            body = rms(x[i + int(0.15 * RATE):i + int(0.65 * RATE)])
            # Fricative onsets ("Speed", "Start") are quiet, so the onset test is
            # absolute: speech must be present at once, and the sentence must
            # not have begun before its recorded start (a shifted or cut join).
            if body > 0 and (onset < min_onset or before > max(silence, 0.5 * onset)):
                problems.append({'scene': scene['scene'], 'sentence': sentence.get('sentence'),
                                 'audible_start': start, 'before': round(before, 4),
                                 'onset': round(onset, 4), 'body': round(body, 4)})
    return problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--lesson', required=True)
    a = p.parse_args()
    from published_terminal_prefix import RECORD as TERMINAL_RECORD, checked_terminal_narration
    terminal = a.stage / 'production' / a.lesson / TERMINAL_RECORD
    if terminal.exists() or terminal.is_symlink():
        checked_terminal_narration(a.stage, a.lesson, decode=False)
    root = a.stage / 'production' / a.lesson / 'audio'
    failures, tracks = {}, 0
    for meta_path in sorted(root.glob('*/*.json')):
        audio = meta_path.with_suffix('.m4a')
        if not audio.exists():
            continue
        tracks += 1
        problems = check_track(audio, json.loads(meta_path.read_text()))
        if problems:
            failures[f'{meta_path.parent.name}/{meta_path.stem}'] = problems
    print(json.dumps({'lesson': a.lesson, 'tracks': tracks, 'failed_tracks': len(failures),
                      'failures': failures}, indent=1)[:4000])
    if not tracks or failures:
        raise SystemExit(f'{a.lesson}: narration onset check failed on {len(failures)} of {tracks} tracks')
    print(f'{a.lesson}: {tracks} tracks, every sentence onset intact')


if __name__ == '__main__':
    main()
