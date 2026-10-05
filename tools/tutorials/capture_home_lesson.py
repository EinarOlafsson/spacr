#!/usr/bin/env python3
"""Prepare a private complete Home lesson review, without publishing narration.

Run under the agreed cgroup/CPU limits. Uses actual offscreen Qt controls and
an isolated filesystem namespace; existing masters and user settings stay read-only.
The silent video and captions have provisional reading times, not speech alignment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def stamp(seconds):
    milliseconds = round(seconds * 1000)
    return f'{milliseconds // 3600000:02}:{milliseconds // 60000 % 60:02}:{milliseconds // 1000 % 60:02}.{milliseconds % 1000:03}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    parser.add_argument('--capture-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--reuse-capture', action='store_true',
                        help='Stage an existing completed capture; do not record again')
    args = parser.parse_args()
    stage = args.stage.resolve()
    if not args.capture_child and not args.reuse_capture:
        stage.mkdir(parents=True, exist_ok=False)
    config = stage / 'config/home'
    for folder in ('config/home', 'state/home', 'cache', 'example_data', 'runs', 'app-state', 'logs', 'mpl', 'tmp'):
        (stage / folder).mkdir(parents=True, exist_ok=True)
    if not args.capture_child and not args.reuse_capture:
        # Disposable old folders make the genuine prune confirmation reachable.
        fixtures = []
        for number in (1, 2):
            folder = stage / 'app-state/runs' / f'tutorial-prune-example-{number}'
            folder.mkdir(parents=True)
            note = folder / 'README.txt'
            note.write_text('Disposable Storage prune demonstration; no analysis results.\n')
            old = time.time() - 30 * 86400
            os.utime(note, (old, old))
            os.utime(folder, (old, old))
            fixtures.append({'path': str(folder), 'sha256': digest(note),
                             'modified': old, 'analysis_results': False})
        write(stage / 'storage-demo-fixtures.json', fixtures)
    os.environ.update(XDG_CONFIG_HOME=str(config), XDG_STATE_HOME=str(stage / 'state/home'),
                      XDG_CACHE_HOME=str(stage / 'cache'), SPACR_LOG_DIR=str(stage / 'logs'),
                      SPACR_HOME=str(stage / 'app-state'),
                      CELLPOSE_LOCAL_MODELS_PATH=str(stage / 'cache/cellpose'),
                      HF_HOME=str(stage / 'cache/huggingface'),
                      TORCH_HOME=str(stage / 'cache/torch'),
                      SPACR_BACKENDS_DIR=str(stage / 'cache/backends'),
                      SPACR_NEWS_CACHE=str(stage / 'cache/news'),
                      MPLCONFIGDIR=str(stage / 'mpl'), TMPDIR=str(stage / 'tmp'), QT_QPA_PLATFORM='offscreen',
                      SPACR_LANGUAGE='en', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', SPACR_TUTORIAL_CACHE_ISOLATED='1')
    os.environ.pop('SPACR_CHAINING_PINS', None)
    if args.capture_child:
        sys.path.insert(0, str(REPO))
        from spacr.qt import preferences
        actual = Path(preferences._settings().fileName()).resolve()
        if not actual.is_relative_to(config):
            raise RuntimeError(f'Unsafe preferences path before any write: {actual}')
        write(stage / 'profile-guard.json', {'path': str(actual), 'inside_private_stage': True})
        import capture_refresh
        sys.argv = [str(REPO / 'tools/tutorials/capture_refresh.py'), '--module', 'home',
                    '--stage', str(stage), '--platform', 'offscreen']
        return capture_refresh.main()
    # Keep the ordinary HOME value; isolate only the application's actual paths.
    if not args.reuse_capture:
        subprocess.run(['bwrap', '--die-with-parent', '--unshare-net', '--ro-bind', '/', '/',
                        '--dev-bind', '/dev', '/dev', '--bind', str(stage), str(stage),
                        '--bind', str(stage / 'app-state'), str(Path.home() / '.spacr'),
                        '--bind', str(stage / 'example_data'), str(Path.home() / '.cache/spacr/example_data'),
                        '--', sys.executable, str(Path(__file__).resolve()), '--stage', str(stage),
                        '--capture-child'], check=True, timeout=180)
    from stage_lesson import stage_lesson
    lesson_path = REPO / 'tools/tutorials/lessons/05_home.json'
    lesson = json.loads(lesson_path.read_text())
    frames = json.loads((stage / 'captures/home/frames.json').read_text())
    focus = {scene['visual']: None for scene in lesson['scenes']}
    # The prior fragment used a larger standalone dialog. Bind this capture's
    # actual dialog bounds rather than cropping it with the old fixed rectangle.
    for name in {scene['visual'] for scene in lesson['scenes']
                 if scene['visual'].startswith('13')}:
        dialogs = [item for item in frames[name]['dialogs'] if item['rect']]
        if not dialogs:
            raise RuntimeError(f'Expected an actual dialog in {name}')
        # Prune has both Preferences and its confirmation; include their union.
        left = min(item['rect'][0] for item in dialogs)
        top = min(item['rect'][1] for item in dialogs)
        right = max(item['rect'][0] + item['rect'][2] for item in dialogs)
        bottom = max(item['rect'][1] + item['rect'][3] for item in dialogs)
        focus[name] = [left, top, right - left, bottom - top]
    focus_path = stage / 'home-focus.json'
    write(focus_path, {'english_sha256': hashlib.sha256(json.dumps(
        lesson, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), 'scenes': focus,
        'recapture': {'capture_module': 'home',
                      'frames': {name: frame['sha256'] for name, frame in frames.items()}}})
    stage_lesson(lesson_path, 'home', stage, check_only=True, focus_map=focus_path)
    stage_lesson(lesson_path, 'home', stage, focus_map=focus_path)
    visual = json.loads((stage / 'production/05_home/visual.json').read_text())
    from PIL import Image
    output = stage / 'review'
    output.mkdir()
    concat, captions, timeline = [], ['WEBVTT', ''], []
    elapsed = 0.0
    for index, (scene, geometry) in enumerate(zip(lesson['scenes'], visual['scenes'], strict=True), 1):
        source = (stage / 'production/05_home' / geometry['image']).resolve()
        with Image.open(source) as image:
            if 'focus' in geometry:
                x, y, width, height = geometry['focus']
                image = image.crop((x, y, x + width, y + height))
            image.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
            frame = Image.new('RGB', (1920, 1080), '#161719')
            frame.paste(image, ((1920 - image.width) // 2, (1080 - image.height) // 2))
            filename = f'scene_{index:02}.png'
            frame.save(output / filename)
        words = scene['narration'].split()
        duration = max(6, len(words) / 2.5 + scene.get('hold_after', 0))
        duration = round(duration, 1)
        concat += [f"file '{filename}'", f'duration {duration}']
        chunks = [' '.join(words[start:start + 18]) for start in range(0, len(words), 18)]
        for offset, chunk in enumerate(chunks):
            captions += [f'{stamp(elapsed + duration * offset / len(chunks))} --> {stamp(elapsed + duration * (offset + 1) / len(chunks))}', chunk, '']
        timeline.append({'scene': index, 'visual': scene['visual'], 'start': elapsed, 'duration': duration,
                         'narration': scene['narration'], 'source_sha256': digest(source),
                         'review_frame_sha256': digest(output / filename), 'focus': geometry.get('focus')})
        elapsed = round(elapsed + duration, 1)
    concat.append(f"file '{filename}'")
    (output / 'frames.concat').write_text('\n'.join(concat) + '\n')
    (output / 'home-provisional.en.vtt').write_text('\n'.join(captions))
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                    '-filter_threads', '1', '-f', 'concat', '-safe', '1', '-i', 'frames.concat',
                    '-vf', 'fps=10', '-t', str(elapsed), '-c:v', 'libx264', '-threads', '1',
                    '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    'home-silent.mp4'], cwd=output, check=True, timeout=240)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
                       '-show_format', '-of', 'json', str(output / 'home-silent.mp4')], text=True))
    if len(probe['streams']) != 1 or probe['streams'][0]['codec_type'] != 'video':
        raise RuntimeError('Private review must contain only its silent video stream')
    if abs(float(probe['format']['duration']) - elapsed) > .11:
        raise RuntimeError('Encoded duration differs from provisional caption timeline')
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-i', str(output / 'home-silent.mp4'),
                    '-f', 'null', '-'], check=True, timeout=120)
    (output / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Private Home review</title>'
        '<style>body{background:#161719;color:white;font:18px sans-serif}video{width:100%;max-height:85vh}</style>'
        '<p>Private silent review. Caption timing is provisional; no narration or publication acceptance.</p>'
        '<video controls><source src="home-silent.mp4"><track default kind="captions" srclang="en" '
        'label="Provisional English" src="home-provisional.en.vtt"></video>')
    write(output / 'receipt.json', {'accepted_stage': True, 'published': False, 'scene_count': len(timeline),
          'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
          'source_sha256': {p: digest(REPO / p) for p in ('tools/tutorials/lessons/05_home.json',
              'tools/tutorials/capture_home_lesson.py', 'tools/tutorials/capture_refresh.py', 'tools/tutorials/capture_home.py')},
          'platform': 'offscreen', 'user_files': 'read-only namespace; writable private application paths only',
          'duration': elapsed, 'timeline': timeline, 'video_sha256': digest(output / 'home-silent.mp4'),
          'captions_sha256': digest(output / 'home-provisional.en.vtt'), 'probe': probe,
          'remaining': ['Inspect all captured scenes and encoded review frames', 'Translate new narration',
                        'Render and review matching narration for supported voices',
                        'Align captions to actual speech and verify player/media matrix', 'Publish paired catalog and media']})
    print(f'Complete private Home stage: {len(timeline)} scenes, {elapsed}s silent review; not published.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
