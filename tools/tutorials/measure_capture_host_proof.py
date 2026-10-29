"""Measure a capture-host recording: frame distinctness, cursor motion, window shape.

    python tools/tutorials/measure_capture_host_proof.py --video OUT/preferences_proof.mp4 \
        --receipt OUT/proof_receipt.json --trace OUT/pointer_trace.json \
        --old OLD_XVFB_CLIP.mp4 --out OUT/measurements

Reports (``measurements.json``):

* distinct frames: exact (decoded MD5 differs from the previous frame) and
  ``ffmpeg mpdecimate`` with its default thresholds, for the whole clip and
  for the opening Home segment where only the field animates;
* effective fps = distinct frames / duration;
* the cursor, found in every frame by masked template matching of the real
  cursor sprite (taken from the recording itself), its per-frame
  displacement and the largest jump as a percentage of the screen width;
* Preferences window shape: the card's corner pixels and the temporal
  variation inside the card while the field moves behind it (translucency),
  compared with the same measurements on the old Xvfb clip.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np


def frames(video: Path, scale: int = 1):
    """Yield (index, HxWx3 uint8) decoded frames."""
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                                '-show_entries', 'stream=width,height', '-of', 'json', str(video)]))
    w, h = probe['streams'][0]['width'] // scale, probe['streams'][0]['height'] // scale
    command = ['ffmpeg', '-v', 'error', '-i', str(video)]
    if scale != 1:
        command += ['-vf', f'scale={w}:{h}:flags=area']
    command += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    size = w * h * 3
    i = 0
    while True:
        data = proc.stdout.read(size)
        if len(data) < size:
            break
        yield i, np.frombuffer(data, np.uint8).reshape(h, w, 3)
        i += 1
    proc.wait()


def exact_distinct(video: Path):
    lines = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(video), '-an', '-f', 'framemd5', '-'],
                                    text=True).splitlines()
    md5 = [line.split(',')[-1].strip() for line in lines if not line.startswith('#')]
    changed = [i == 0 or md5[i] != md5[i - 1] for i in range(len(md5))]
    return md5, changed


def content_changes(video: Path, level: int = 24, pixels: int = 12):
    """Per frame: did the picture really change? (>= ``pixels`` pixels moved by > ``level``).

    Exact MD5 comparison counts encoder noise as change (the old Xvfb clip of
    a static dialog is "99.6 % distinct" by MD5); this ignores it.
    """
    import cv2
    previous, changed, counts = None, [], []
    for _, frame in frames(video):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if previous is None:
            changed.append(True)
            counts.append(None)
        else:
            n = int(np.count_nonzero(cv2.absdiff(gray, previous) > level))
            changed.append(n >= pixels)
            counts.append(n)
        previous = gray
    return changed, counts


def mpdecimate(video: Path, start: float | None = None, end: float | None = None):
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'debug']
    if start is not None:
        command += ['-ss', f'{start}']
    if end is not None:
        command += ['-t', f'{end - (start or 0)}']
    command += ['-i', str(video), '-vf', 'mpdecimate', '-an', '-f', 'null', '-']
    log = subprocess.run(command, capture_output=True, text=True).stderr
    keep = len(re.findall(r'mpdecimate.* keep ', log))
    drop = len(re.findall(r'mpdecimate.* drop ', log))
    return {'kept': keep, 'dropped': drop, 'ratio': keep / max(1, keep + drop)}


def size(video: Path):
    w, h = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                    'stream=width,height', '-of', 'csv=p=0', str(video)], text=True).strip().split(',')
    return int(w), int(h)


def duration(video: Path) -> float:
    return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                          '-of', 'csv=p=0', str(video)], text=True))


def xcursor_image(path: Path, size: int):
    """(RGBA uint8 image, hotspot) of the nominal ``size`` image in an Xcursor file."""
    import struct
    data = path.read_bytes()
    if data[:4] != b'Xcur':
        raise ValueError(f'not an Xcursor file: {path}')
    _, _, ntoc = struct.unpack_from('<III', data, 4)
    entries = [struct.unpack_from('<III', data, 16 + 12 * i) for i in range(ntoc)]
    images = [e for e in entries if e[0] == 0xfffd0002]
    kind, nominal, position = min(images, key=lambda e: (abs(e[1] - size), e[1]))
    _, _, _, _, width, height, xhot, yhot, _ = struct.unpack_from('<9I', data, position)
    argb = np.frombuffer(data, np.uint32, width * height, position + 36).reshape(height, width)
    a = (argb >> 24) & 255
    rgb = np.stack([(argb >> 16) & 255, (argb >> 8) & 255, argb & 255], axis=-1).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        rgb = np.where(a[..., None] > 0, rgb * 255.0 / np.maximum(a[..., None], 1), 0)
    return np.dstack([rgb.clip(0, 255), a]).astype(np.uint8), (xhot, yhot)


CURSOR_SHAPES = ('left_ptr', 'xterm', 'hand2', 'sb_h_double_arrow', 'sb_v_double_arrow',
                 'dnd-move', 'dnd-no-drop', 'dnd-copy')


def cursor_templates(theme: str, size: int):
    """Masked matching templates for the pointer shapes an app commonly shows."""
    templates = []
    for name, nominal in ((n, z) for z in dict.fromkeys((size, 24, 32, 48)) for n in CURSOR_SHAPES):
        path = Path('/usr/share/icons') / theme / 'cursors' / name
        if not path.exists():
            continue
        rgba, hotspot = xcursor_image(path.resolve(), nominal)
        name = f'{name}@{rgba.shape[0]}'
        if any(t['name'] == name for t in templates):
            continue
        mask = (rgba[..., 3] == 255).astype(np.float32)
        if mask.sum() < 20:
            continue
        gray = (rgba[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))
        templates.append({'name': name, 'gray': gray.astype(np.float32), 'mask': mask, 'hotspot': hotspot})
    return templates


def track_cursor(video: Path, templates, window=200, threshold=0.004):
    """Find the real cursor sprite in every frame by masked template matching.

    Only fully opaque sprite pixels are compared (they reach the frame
    unblended), so the moving field behind the pointer does not matter.
    Score = mean squared error / 255^2 over those pixels (0 = exact).
    """
    import cv2
    out, last = [], None
    for i, frame in frames(video):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        best = None
        for scope in (('near', 'full') if last is not None else ('full',)):
            if scope == 'near':
                ox, oy = max(0, last[0] - window), max(0, last[1] - window)
                region = gray[oy:last[1] + window, ox:last[0] + window]
            else:
                ox, oy, region = 0, 0, gray
            for t in templates:
                th, tw = t['gray'].shape
                if region.shape[0] < th or region.shape[1] < tw:
                    continue
                result = cv2.matchTemplate(region, t['gray'], cv2.TM_SQDIFF, mask=t['mask'])
                value, _, loc, _ = cv2.minMaxLoc(result)
                score = value / (t['mask'].sum() * 255.0 ** 2)
                if best is None or score < best[0]:
                    best = (score, loc[0] + ox + t['hotspot'][0], loc[1] + oy + t['hotspot'][1], t['name'])
            if best is not None and best[0] <= threshold:
                break
        found = best is not None and best[0] <= threshold
        out.append({'frame': i, 'found': bool(found), 'score': round(float(best[0]), 6) if best else None,
                    'shape': best[3] if found else None,
                    'x': int(best[1]) if found else None, 'y': int(best[2]) if found else None})
        if found:
            last = (int(best[1]), int(best[2]))
    return out


def card_measurements(video: Path, rect, times, label):
    """Corner and interior statistics of a window rectangle over several frames."""
    x, y, w, h = rect
    vw, vh = size(video)
    stack = []
    for t in times:
        data = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', f'{t}', '-i', str(video), '-frames:v', '1',
                                        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
        frame = np.frombuffer(data, np.uint8).reshape(vh, vw, 3)
        stack.append(frame[y:y + h, x:x + w].astype(np.int16))
    stack = np.stack(stack)
    # Interior band (inside the card, away from text): bottom-left help area.
    interior = stack[:, int(h * 0.45):int(h * 0.75), int(w * 0.02):int(w * 0.08)]
    corner = stack[:, 0:6, 0:6]
    edge_mid = stack[:, 0:6, w // 2 - 3:w // 2 + 3]
    return {'label': label, 'rect': rect, 'times': times,
            'interior_temporal_std': round(float(interior.std(axis=0).mean()), 3),
            'interior_spatial_std': round(float(interior.std(axis=(1, 2)).mean()), 3),
            'corner_mean_rgb': corner.mean(axis=(0, 1, 2)).round(1).tolist(),
            'top_edge_mid_mean_rgb': edge_mid.mean(axis=(0, 1, 2)).round(1).tolist()}


def card_body(video: Path, receipt: dict, out: Path, label: str) -> dict:
    """Pixel statistics of the Preferences card BODY while the pointer rests.

    Region: the empty part of the General page (40-60 % of the card height,
    4-96 % of its width). An opaque body is flat (spatial and temporal std ~0);
    a see-through or animated body varies in space and over time.
    """
    import cv2
    steps = {s['step']: s['t'] for s in receipt['steps']}
    start = steps['card still (pointer at rest)'] + 0.3
    still = [t for t in receipt['popup_backdrop']][-1]['widgets']
    stop = max(t['t'] for t in still) - 0.2 if still else start + 2.0
    x, y, w, h = receipt['dialog_on_open']['dialog_rect']
    body = (x + int(w * 0.04), y + int(h * 0.40), int(w * 0.92), int(h * 0.20))
    ref = (x + w + 40, y + int(h * 0.40), 300, int(h * 0.20))      # field outside the card
    stack, ref_stack = [], []
    for i, frame in frames(video):
        t = i / 30.0
        if start <= t <= stop:
            g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
            stack.append(g[body[1]:body[1] + body[3], body[0]:body[0] + body[2]])
            ref_stack.append(g[ref[1]:ref[1] + ref[3], ref[0]:ref[0] + ref[2]])
            if abs(t - (start + stop) / 2) < 1 / 60:
                cv2.imwrite(str(out / f'card_{label}_still.png'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                crop = frame[body[1]:body[1] + 160, body[0]:body[0] + 320]
                cv2.imwrite(str(out / f'card_{label}_body_4x.png'),
                            cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), None, fx=4, fy=4,
                                       interpolation=cv2.INTER_NEAREST))
    def stats(a):
        a = np.stack(a)
        rng = a.max(axis=0) - a.min(axis=0)
        changed = [float(np.mean(np.abs(a[k + 1] - a[k]) > 4)) for k in range(len(a) - 1)]
        return {'frames': len(a), 'mean_grey': round(float(a.mean()), 2),
                'spatial_std': round(float(a.std(axis=(1, 2)).mean()), 3),
                'temporal_std_mean': round(float(a.std(axis=0).mean()), 3),
                'pixels_range_over_8': round(float(np.mean(rng > 8)), 4),
                'frames_with_change': round(float(np.mean([c > 0.0005 for c in changed])), 3)}
    return {'label': label, 'window_s': [round(start, 2), round(stop, 2)], 'body_rect': body,
            'body': stats(stack), 'field_outside_card': stats(ref_stack),
            'popup_backdrop': receipt.get('popup_backdrop'), 'extra_settings': receipt.get('extra_settings')}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--old', type=Path)
    parser.add_argument('--old-dialog-rect', default=None, help='x,y,w,h of the dialog in the old clip')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cursor-theme', default='Yaru')
    parser.add_argument('--cursor-size', type=int, default=32)
    parser.add_argument('--card-body', metavar='LABEL',
                        help='Only measure the Preferences card body (needs a --card-still recording)')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(args.receipt.read_text())
    if args.card_body:
        result = card_body(args.video, receipt, args.out, args.card_body)
        (args.out / f'card_{args.card_body}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        return 0
    trace = json.loads(args.trace.read_text())
    steps = {s['step']: s['t'] for s in receipt['steps']}
    fps = 30.0
    result = {'video': str(args.video), 'size': size(args.video), 'duration_s': duration(args.video)}

    md5, changed = exact_distinct(args.video)
    n = len(md5)
    first_move = next(s['t'] for s in receipt['steps'] if s['step'].startswith('glide'))
    home = slice(15, int(first_move * fps) - 15)      # field animating, nothing else should move
    result['frames'] = n
    result['exact_distinct'] = {'whole': round(sum(changed) / n, 4),
                                'home_field_only': round(float(np.mean(changed[home])), 4),
                                'home_segment_s': [home.start / fps, home.stop / fps]}
    result['effective_fps_exact'] = round(sum(changed) / result['duration_s'], 2)
    real, _ = content_changes(args.video)
    result['content_changed'] = {'whole': round(sum(real) / n, 4),
                                 'home_field_only': round(float(np.mean(real[home])), 4),
                                 'rule': '>= 12 pixels changed by > 24 grey levels from the previous frame'}
    result['effective_fps_content'] = round(sum(real) / result['duration_s'], 2)
    result['mpdecimate'] = {'whole': mpdecimate(args.video),
                            'home_field_only': mpdecimate(args.video, home.start / fps, home.stop / fps)}
    result['effective_fps_mpdecimate'] = round(result['mpdecimate']['whole']['kept'] / result['duration_s'], 2)
    samples = receipt.get('field_samples') or []
    if len(samples) >= 3:
        def rate(a, b):
            return (b['frames_painted'] - a['frames_painted']) / (b['t'] - a['t'])
        home_fps = rate(samples[0], samples[1])
        home_distinct_fps = float(np.sum(real[home])) / ((home.stop - home.start) / fps)
        result['field'] = {
            'fps_cap': samples[0]['fps_cap'], 'gpu_renderer_refusal': samples[-1]['gpu_refusal'],
            'app_painted_fps_home': round(home_fps, 2),
            'app_painted_fps_whole': round(rate(samples[0], samples[-1]), 2),
            'video_distinct_fps_home': round(home_distinct_fps, 2),
            # 1.0 = every field frame the app painted on the still Home screen
            # reached the video as its own frame.
            'capture_coverage_home': round(min(1.0, home_distinct_fps / home_fps), 3) if home_fps else None,
            'max_distinct_ratio_possible_at_30fps': round(min(1.0, home_fps / fps), 3),
        }

    track = track_cursor(args.video, cursor_templates(args.cursor_theme, args.cursor_size))
    (args.out / 'cursor_track.json').write_text(json.dumps(track) + '\n')
    found = [t for t in track if t['found']]
    steps_px = [float(np.hypot(b['x'] - a['x'], b['y'] - a['y'])) for a, b in zip(track, track[1:])
                if a['found'] and b['found']]
    width = size(args.video)[0]
    result['cursor'] = {
        'found_fraction': round(len(found) / len(track), 4),
        'shapes_seen': sorted({t['shape'] for t in found}),
        'per_frame_displacement_px': {'median_moving': round(float(np.median([s for s in steps_px if s > 0.5] or [0])), 2),
                                      'p95': round(float(np.percentile(steps_px, 95)), 2),
                                      'p99': round(float(np.percentile(steps_px, 99)), 2),
                                      'max': round(max(steps_px), 2)},
        'max_jump_percent_of_width': round(100 * max(steps_px) / width, 3),
        'frames_with_jump_over_2pct_width': int(sum(s > 0.02 * width for s in steps_px)),
    }
    # Trace-side check: commanded per-video-frame displacement.
    moves = np.array([(e['t'], e['x'], e['y']) for e in trace if e['event'] == 'move'])
    # Per video frame, inside each glide (a pause between gestures is not a step).
    breaks = np.nonzero(np.diff(moves[:, 0]) > 0.1)[0] + 1
    commanded = np.zeros(1)
    for seg in np.split(moves, breaks):
        grid = np.arange(seg[0, 0], seg[-1, 0], 1 / fps)
        if len(grid) > 1:
            commanded = np.concatenate([commanded, np.hypot(np.diff(np.interp(grid, seg[:, 0], seg[:, 1])),
                                                            np.diff(np.interp(grid, seg[:, 0], seg[:, 2])))])
    gaps = np.diff(moves[:, 0])
    in_motion = gaps[gaps < 0.2]
    call_ms = [e['call_ms'] for e in trace if e.get('call_ms') is not None]
    result['pointer_trace'] = {
        'commanded_max_per_frame_px': round(float(commanded.max()), 2),
        'step_interval_ms': {'median': round(float(np.median(in_motion) * 1000), 2),
                             'p95': round(float(np.percentile(in_motion, 95) * 1000), 2),
                             'max': round(float(in_motion.max() * 1000), 2)},
        'dbus_call_ms': ({'median': round(float(np.median(call_ms)), 2),
                          'p95': round(float(np.percentile(call_ms, 95)), 2),
                          'max': round(float(np.max(call_ms)), 2)} if call_ms else None),
    }

    # Window shape: the Preferences card, while open on its first tab.
    dialog = receipt['dialog_on_open']
    t_open = steps[next(k for k in steps if k.startswith('Preferences open'))]
    t_hover = steps[next(k for k in steps if k.startswith('hover'))]
    rect = dialog['dialog_rect']
    result['preferences_window'] = {
        'qt': {k: dialog[k] for k in ('frameless', 'translucent', 'alpha_buffer', 'has_mask', 'window_opacity')},
        'new': card_measurements(args.video, rect, [round(t_open + 0.6 + 0.1 * k, 2) for k in range(8)], 'gnome host'),
    }
    for name, t in (('preferences_open', t_open + 0.8), ('preferences_category_scrolled',
                    steps['scroll the category'] + 1.6), ('preferences_hover', t_hover + 1.0)):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{t}', '-i', str(args.video), '-frames:v', '1',
                        str(args.out / f'{name}.png')], check=True)
    x, y, w, h = rect
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{t_open + 0.8}', '-i', str(args.video), '-frames:v', '1',
                    '-vf', f'crop=220:220:{x - 40}:{y - 40},scale=880:880:flags=neighbor',
                    str(args.out / 'preferences_corner_zoom_new.png')], check=True)
    if args.old:
        old_md5, old_changed = exact_distinct(args.old)
        old_real, _ = content_changes(args.old)
        result['old_xvfb'] = {'video': str(args.old), 'duration_s': duration(args.old),
                              'exact_distinct': round(sum(old_changed) / len(old_changed), 4),
                              'content_changed': round(sum(old_real) / len(old_real), 4),
                              'mpdecimate': mpdecimate(args.old)}
        result['old_xvfb']['effective_fps_mpdecimate'] = round(
            result['old_xvfb']['mpdecimate']['kept'] / result['old_xvfb']['duration_s'], 2)
        if args.old_dialog_rect:
            orect = [int(v) for v in args.old_dialog_rect.split(',')]
            result['preferences_window']['old'] = card_measurements(
                args.old, orect, [1.0 + 0.1 * k for k in range(8)], 'old xvfb')
            ox, oy = orect[:2]
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '1.0', '-i', str(args.old), '-frames:v', '1',
                            '-vf', f'crop=220:220:{ox - 40}:{oy - 40},scale=880:880:flags=neighbor',
                            str(args.out / 'preferences_corner_zoom_old.png')], check=True)
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '1.0', '-i', str(args.old), '-frames:v', '1',
                            str(args.out / 'old_xvfb_preferences.png')], check=True)
    (args.out / 'measurements.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
