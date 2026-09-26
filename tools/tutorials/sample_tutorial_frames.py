#!/usr/bin/env python3
"""Sampled-frame sweep of published tutorial media for local paths and theme (item 447).

For every lesson in the published player catalog this reads the poster and a
few evenly spaced frames of the copy the player actually streams (the hosted
web copy when the catalog names one, otherwise the Pages-tree video). Each
image is OCR-read after a brightness lift, so text inside the dimmed area
around a spotlight is still read, and the text is scanned for local or
maintainer paths. Each image's luminance is also measured, so a light-themed
frame shows up.

It reads media only and writes one JSON receipt. A hit or a light frame is a
finding to review; this is a screen, not a visual sign-off.

Run with tools/run_capped.sh; OCR uses rapidocr_onnxruntime on the CPU.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PUBLISHED = REPO / 'docs/source/_extra/tutorials'

# Matched against OCR text with whitespace removed and lower-cased. The second
# group also drops underscores, which OCR often loses.
PATH_PATTERNS = {
    'unix_home': re.compile(r'/home/'),
    'mounted_volume': re.compile(r'/mnt/|/nas_mnt'),
    'macos_users': re.compile(r'/users/'),
    'windows_drive': re.compile(r'(?<![a-z])[a-z]:\\'),
}
LOOSE_PATTERNS = {
    'maintainer_project': 'toxoplasmaprojects',
    'refresh_stage': 'refresh2026',
    'maintainer_disk': 'firecuda',
    'maintainer_account': 'olafsson',
}

# Dark theme: most pixels are dark. A light theme (or a light external window
# filling the frame) raises both numbers well past these limits.
DARK_MEAN_LIMIT = 0.35
BRIGHT_FRACTION_LIMIT = 0.20
BRIGHT_LEVEL = 0.75
SAMPLE_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
OCR_GAIN = 3.0
OCR_TILE = (1920, 1080)
OCR_OVERLAP = (120, 60)


# The project's public GitHub/Pages URLs carry the maintainer's handle; they
# are published addresses, not local paths.
PUBLIC_URL = re.compile(r'olafsson\.g[il1]thub\.io|g[il1]thub\.com/einarolafsson|einarolafsson/spacr')
# A generic sandbox account (the installation lessons' clean shells) is a
# /home/ path but names nobody; it is reported apart from real offenders.
# The author's full name (PyPI author field, commit author) is public attribution;
# a bare account name (file-dialog sidebar, /home/<account>) is local.
LOCAL_ACCOUNT = re.compile(r'(?<!einar)(?<!birnir)olafsson')
GENERIC_HOME = re.compile(r'/home/user/')
NEUTRAL_KINDS = {'generic_home'}


def path_hits(lines):
    """Return [(kind, text)] for every OCR line that names a local path."""
    hits = []
    for text in lines:
        squeezed = re.sub(r'\s+', '', text).lower()
        loose = squeezed.replace('_', '').replace('-', '')
        for kind, pattern in PATH_PATTERNS.items():
            if pattern.search(squeezed):
                if kind == 'unix_home' and GENERIC_HOME.search(squeezed) and \
                        len(re.findall(r'/home/', squeezed)) == len(GENERIC_HOME.findall(squeezed)):
                    kind = 'generic_home'
                hits.append((kind, text))
        for kind, token in LOOSE_PATTERNS.items():
            if token in loose and not (kind == 'maintainer_account'
                                       and not LOCAL_ACCOUNT.search(PUBLIC_URL.sub('', squeezed))):
                hits.append((kind, text))
    return hits


def offending(hits):
    return [h for h in hits if h['kind'] not in NEUTRAL_KINDS]


def theme(image):
    """Luminance summary of a PIL image; dark means both limits hold."""
    import numpy as np
    luma = np.asarray(image.convert('L'), dtype=np.float32) / 255.0
    mean = float(luma.mean())
    bright = float((luma > BRIGHT_LEVEL).mean())
    return {'mean_luma': round(mean, 4), 'bright_fraction': round(bright, 4),
            'dark': mean < DARK_MEAN_LIMIT and bright < BRIGHT_FRACTION_LIMIT}


def lifted(image):
    """Brightness-lifted RGB array so spotlight-dimmed text is readable."""
    import numpy as np
    luma = np.asarray(image.convert('L'), dtype=np.float32)
    lift = np.clip(luma * OCR_GAIN, 0, 255).astype(np.uint8)
    return np.stack([lift] * 3, axis=-1)


def catalog_items(published=PUBLISHED):
    """(lesson, poster, streamed video) for every playable published lesson."""
    source = (published / 'lesson_catalog.js').read_text(encoding='utf-8')
    catalog = json.loads(source[source.index('{'):source.rindex('}') + 1])
    index = (published / 'index.html').read_text(encoding='utf-8')
    root = re.search(r'data-web-root="([^"]*)"', index)
    web_root = root.group(1) if root else ''
    items = []
    for lesson in catalog['lessons']:
        if not lesson.get('poster') or not lesson.get('silent'):
            continue
        if web_root and lesson.get('web'):
            video, source_kind = f"{web_root}/{lesson['web']}", 'hosted_web_copy'
        else:
            video, source_kind = str(published / 'production' / lesson['silent']), 'pages_video'
        items.append({'lesson': lesson['id'], 'poster': str(published / 'production' / lesson['poster']),
                      'video': video, 'video_source': source_kind})
    return items


def duration(video):
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                   '-of', 'csv=p=0', video], timeout=120)
    return float(out.decode().strip())


def frame_at(video, seconds):
    from PIL import Image
    data = subprocess.check_output(
        ['ffmpeg', '-nostdin', '-v', 'error', '-threads', '2', '-ss', f'{seconds:.3f}', '-i', video,
         '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'], timeout=300)
    return Image.open(io.BytesIO(data)).convert('RGB')


_ENGINE = None


def ocr_lines(image):
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
    import numpy as np
    pixels = lifted(image)
    height, width = pixels.shape[:2]
    lines = []
    # The detector downsizes a whole 4K frame until small dimmed text is lost;
    # overlapping 1080p tiles keep it at native size.
    for top in range(0, max(1, height - OCR_OVERLAP[1]), OCR_TILE[1] - OCR_OVERLAP[1]):
        for left in range(0, max(1, width - OCR_OVERLAP[0]), OCR_TILE[0] - OCR_OVERLAP[0]):
            tile = np.ascontiguousarray(pixels[top:top + OCR_TILE[1], left:left + OCR_TILE[0]])
            result, _ = _ENGINE(tile)
            lines.extend(row[1] for row in (result or []))
    return lines


def inspect(image, label, ocr):
    record = {'image': label, 'size': list(image.size), **theme(image)}
    if ocr is not None:
        lines = ocr(image)
        record['ocr_lines'] = len(lines)
        record['path_hits'] = [{'kind': k, 'text': t} for k, t in sorted(set(path_hits(lines)))]
    return record


def sweep_item(item, frames=len(SAMPLE_FRACTIONS), ocr=ocr_lines):
    from PIL import Image
    records, errors = [], []
    try:
        with Image.open(item['poster']) as poster:
            records.append(inspect(poster.convert('RGB'), 'poster', ocr))
    except Exception as exc:  # report, keep sweeping
        errors.append(f'poster: {exc}')
    try:
        length = duration(item['video'])
        fractions = SAMPLE_FRACTIONS if frames == len(SAMPLE_FRACTIONS) else \
            tuple((i + 0.5) / frames for i in range(frames))
        for fraction in fractions:
            seconds = length * fraction
            records.append(inspect(frame_at(item['video'], seconds), f'video@{seconds:.2f}s', ocr))
    except Exception as exc:
        errors.append(f'video: {exc}')
    return finish({**item, 'images': records, 'errors': errors})


def finish(result):
    hits = [h for r in result['images'] for h in r.get('path_hits', [])]
    result['path_offender'] = bool(offending(hits))
    result['generic_home_only'] = bool(hits) and not result['path_offender']
    result['light_images'] = [r['image'] for r in result['images'] if not r['dark']]
    return result


def rescore(receipt):
    """Re-apply the current rules to a receipt's stored hit lines (rules only narrow or relabel)."""
    for lesson in receipt['lessons']:
        for image in lesson['images']:
            if 'path_hits' in image:
                texts = sorted({h['text'] for h in image['path_hits']})
                image['path_hits'] = [{'kind': k, 'text': t} for k, t in sorted(set(path_hits(texts)))]
        finish(lesson)
    return summarize(receipt['lessons'], ocr_used=receipt.get('ocr') is not None)


def summarize(results, *, ocr_used):
    return {
        'item': '447', 'date': dt.date.today().isoformat(),
        'scope': ('Poster plus evenly spaced frames of the streamed copy for every published lesson; '
                  'OCR of overlapping 1080p tiles after a brightness lift, scanned for local/maintainer paths; luminance theme screen. '
                  'A screen for review, not a visual sign-off.'),
        'ocr': 'rapidocr_onnxruntime (CPU)' if ocr_used else None,
        'patterns': sorted(PATH_PATTERNS) + sorted(LOOSE_PATTERNS),
        'theme_limits': {'dark_mean_below': DARK_MEAN_LIMIT, 'bright_fraction_below': BRIGHT_FRACTION_LIMIT,
                         'bright_level': BRIGHT_LEVEL},
        'lessons_checked': len(results),
        'images_checked': sum(len(r['images']) for r in results),
        'path_offenders': sorted(r['lesson'] for r in results if r['path_offender']),
        'generic_home_only_lessons': sorted(r['lesson'] for r in results if r.get('generic_home_only')),
        'light_frame_lessons': sorted(r['lesson'] for r in results if r['light_images']),
        'errors': {r['lesson']: r['errors'] for r in results if r['errors']},
        'lessons': results,
    }


def _worker(args):
    item, frames, use_ocr = args
    return sweep_item(item, frames, ocr_lines if use_ocr else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--published', type=Path, default=PUBLISHED)
    parser.add_argument('--lesson', action='append', help='Only these published lesson ids')
    parser.add_argument('--item', nargs=3, action='append', metavar=('ID', 'POSTER', 'VIDEO'),
                        help='Check an explicit (e.g. staged) lesson instead of the catalog')
    parser.add_argument('--frames', type=int, default=len(SAMPLE_FRACTIONS))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--no-ocr', action='store_true', help='Theme screen only')
    parser.add_argument('--rescore', type=Path, help='Re-apply the current rules to an earlier receipt; no OCR')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.rescore:
        receipt = rescore(json.loads(args.rescore.read_text(encoding='utf-8')))
        receipt['rescored_from'] = str(args.rescore)
        args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print('path offenders:', receipt['path_offenders'] or 'none')
        return 1 if receipt['path_offenders'] else 0
    if args.item:
        items = [{'lesson': i, 'poster': p, 'video': v, 'video_source': 'explicit'} for i, p, v in args.item]
    else:
        items = catalog_items(args.published)
        if args.lesson:
            items = [i for i in items if i['lesson'] in set(args.lesson)]
    jobs = [(item, args.frames, not args.no_ocr) for item in items]
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for result in pool.map(_worker, jobs):
            flag = ('PATH ' if result['path_offender'] else '') + ('LIGHT ' if result['light_images'] else '')
            print(result['lesson'], flag or 'ok', *result['errors'], flush=True)
            results.append(result)
    receipt = summarize(results, ocr_used=not args.no_ocr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('path offenders:', receipt['path_offenders'] or 'none')
    print('light-frame lessons:', receipt['light_frame_lessons'] or 'none')
    return 1 if receipt['path_offenders'] or receipt['errors'] else 0


if __name__ == '__main__':
    sys.exit(main())
