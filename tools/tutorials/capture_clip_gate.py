"""Verification gate for every capture-host clip: smooth field, continuous pointer, no top bar.

A clip FAILS when any of these holds (thresholds are recorded in the result):

* **choppy**: fewer than ``min_fps`` (default 0.93 x the clip frame rate: 28 at
  30 fps, 56 at 60 fps) frames per second show new
  picture content while the field animates. A frame counts as new when at
  least ``change_pixels`` pixels differ from the previous frame by more than
  ``change_level`` grey levels OUTSIDE a box around the pointer (so the
  pointer moving over a frozen picture does not count as animation);
* **pointer jump**: between two consecutive video frames the real cursor
  (found by masked template matching of the installed cursor theme, as in
  :mod:`measure_capture_host_proof`) moves further than the minimum-jerk
  bound: ``jump_margin x Human.peak_per_frame x width`` (1.5 x 2 % = 3 % of
  the width, 77 px at 2560, per 30 fps frame; half that per 60 fps frame).
  A human glide made by :class:`human_input.Human` peaks at ~2 % of the
  width per 30 fps frame by construction;
* **pointer lost**: the cursor is found in fewer than ``min_cursor_found`` of
  the frames, so its continuity cannot be certified;
* **top bar**: the GNOME top bar (a black band across the top ~32 px with
  white text in it, the one in the 2026-10-11 proof) is visible in any
  sampled frame.
  When the receipt carries the app paint counter (``app_paint_samples``),
  low-picture-rate seconds are labelled **app freeze**, **app slow**,
  **capture loss**, or **unlabelled** when the app paint rate is unknown.
  Delayed GUI samples are separate stall evidence; resumed counters never
  establish an interpolated paint rate inside the delivery gap.
* **adapter bypass** (when a receipt is given): a scene action that did not
  go through the human pointer happened inside the clip.

    python tools/tutorials/capture_clip_gate.py --video CLIP.native.mp4 [--receipt CLIP.capture.json]

Exit status 0 = passed, 1 = failed; the full result is printed as JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
                str(Path(__file__).resolve().parent)]
from measure_capture_host_proof import (cursor_templates, duration, exact_distinct,  # noqa: E402
                                        frames, size, track_cursor)

#: Distinct frames required, as a fraction of the clip frame rate.
MIN_DISTINCT_FRACTION = 0.93

DEFAULTS = {
    'min_fps': None,               # None: MIN_DISTINCT_FRACTION x the clip frame rate
    'change_level': 10,
    'change_pixels': 4,
    'cursor_box': 72,
    'peak_per_frame': 0.02,       # human_input.Human.peak_per_frame (per 30 fps frame)
    'jump_margin': 1.5,
    'min_cursor_found': 0.9,
    'top_bar_height': 32,
    'top_bar_every': 10,
}


def top_bar_visible(frame: np.ndarray, height: int = 32) -> dict:
    """Does ``frame`` show GNOME's top bar? (black band + white text, distinct from below).

    GNOME 42 at scale 1: rows 0..31 are pure black except the white
    "Activities" label and the clock; the application starts at row 32.
    """
    band = frame[:height].astype(np.int16)
    below = frame[height:height + 8].astype(np.int16)
    black = float(np.mean(band.max(axis=2) < 12))
    edge_black = float(np.mean(band[:4].max(axis=2) < 6))
    text = int(np.count_nonzero(band.min(axis=2) > 180))
    below_black = float(np.mean(below.max(axis=2) < 12))
    visible = bool(black > 0.85 and edge_black > 0.99 and text >= 40 and below_black < 0.5)
    return {'visible': visible, 'black_fraction': round(black, 4), 'top_rows_black': round(edge_black, 4),
            'text_pixels': text, 'below_black_fraction': round(below_black, 4)}


def _box(mask, point, half):
    if point is None:
        return
    x, y = point
    mask[max(0, y - half):max(0, y + half), max(0, x - half):max(0, x + half)] = False


def picture_changes(video: Path, track, *, level: int, pixels: int, cursor_box: int,
                    top_bar_height: int, top_bar_every: int):
    """Per frame: new picture outside the pointer?  Plus sampled top-bar checks."""
    import cv2
    changed, counts, bars = [], [], []
    previous = None
    for i, frame in frames(video):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if i % top_bar_every == 0:
            bars.append({'frame': i, **top_bar_visible(frame, top_bar_height)})
        if previous is None:
            changed.append(True)
            counts.append(None)
        else:
            mask = np.ones(gray.shape, bool)
            for k in (i - 1, i):
                if k < len(track) and track[k]['found']:
                    _box(mask, (track[k]['x'], track[k]['y']), cursor_box)
            n = int(np.count_nonzero((cv2.absdiff(gray, previous) > level) & mask))
            changed.append(n >= pixels)
            counts.append(n)
        previous = gray
    return changed, counts, bars


def app_painted_per_second(samples, seconds: int, *, gaps=()):
    """Counter deltas per second; delivery gaps leave rates unknown.

    Interpolation is only an estimate between normally delivered samples.
    A delayed GUI sampler cannot establish when paints inside its gap took
    place, even when the counter has advanced when the GUI returns.
    """
    if not samples or len(samples) < 2:
        return None
    ts = np.array([t for t, _ in samples], float)
    ns = np.array([n for _, n in samples], float)
    if (not np.isfinite(ts).all() or not np.isfinite(ns).all()
            or np.any(np.diff(ts) <= 0) or np.any(np.diff(ns) < 0)):
        return None
    edges = np.arange(seconds + 1, dtype=float)
    counts = np.interp(edges, ts, ns)
    return [None if any(a < second + 1 and b > second for a, b in gaps)
            else round(float(end - start), 1)
            for second, (start, end) in enumerate(zip(counts, counts[1:]))]


def gui_sampler_gap_evidence(receipt):
    """Delayed GUI delivery, independent of paint count increases.

    Old receipts did not declare their timer cadence. Infer it only from
    sufficiently frequent samples; sparse old samples are not evidence of
    GUI stalls. Their original counters remain estimates, never relabelled
    as the new per-widget metric.
    """
    samples = receipt.get('app_paint_samples') or []
    intervals = [float(b[0]) - float(a[0]) for a, b in zip(samples, samples[1:])]
    cadence = receipt.get('app_gui_sampler_interval_s')
    mode = 'recorded_gui_sampler'
    if cadence is None:
        positive = [value for value in intervals if np.isfinite(value) and value > 0]
        if len(positive) < 3 or float(np.median(positive)) > 0.2:
            return [], 'legacy_cadence_unknown'
        cadence = float(np.median(positive))
        mode = 'inferred_legacy_gui_sampler'
    threshold = max(0.35, 3.5 * float(cadence))
    observed = [(float(a[0]), float(b[0])) for a, b in zip(samples, samples[1:])
                if float(b[0]) - float(a[0]) >= threshold]
    if 'app_gui_sampler_gaps_s' in receipt:
        recorded = [tuple(map(float, window)) for window in receipt['app_gui_sampler_gaps_s']]
        if (any(not np.isfinite([a, b]).all() or b - a < threshold - 0.002 for a, b in recorded)
                or any(not any(abs(a - x) <= 0.002 and abs(b - y) <= 0.002
                               for x, y in recorded)
                       for a, b in observed if b - a > threshold + 0.002)
                or any(not any(abs(a - float(x[0])) <= 0.002
                               and abs(b - float(y[0])) <= 0.002
                               for x, y in zip(samples, samples[1:]))
                       for a, b in recorded)):
            raise ValueError('Recorded GUI sampler gaps disagree with the observed samples')
        observed = recorded
    return observed, mode


def stable_field_painted_per_second(samples, seconds, *, gaps=()):
    """Lowest rate among fields visible throughout the sampled clip.

    A sum of N widget counters is not one field's fps. If navigation leaves
    no stable field identity, paint rates cannot diagnose the new scene;
    GUI delivery gaps remain independent evidence.
    """
    if not samples or len(samples) < 2:
        return None
    if any(row['visible'] and row.get('counter_reset') for _time, rows in samples for row in rows):
        return None
    visible = [{row['id']: row['observed_frames_painted']
                for row in rows if row['visible'] and not row.get('counter_reset')}
               for _time, rows in samples]
    common = set(visible[0]).intersection(*(set(rows) for rows in visible[1:]))
    if not common:
        return None
    rates = [app_painted_per_second(
        [(time, rows[identity]) for (time, _widgets), rows in zip(samples, visible)],
        seconds, gaps=gaps) for identity in common]
    if any(rate is None for rate in rates):
        return None
    return [None if any(rate[second] is None for rate in rates)
            else min(rate[second] for rate in rates) for second in range(seconds)]


#: A second counts as an app freeze when the app painted under this
#: fraction of its own typical (median) rate in the clip.
FREEZE_DROP = 0.8


def sampled_hold_pacing_per_second(samples, seconds, min_fps, *, cadence=None, gaps=()):
    """Diagnostic caps only for fully bracketed, uniformly sampled held fields.

    Before/after snapshots cannot establish pacing within a second. Neither
    can mixed rates, missing/hidden identities, resets or delayed sampling.
    These caps never alter the capture's required distinct-picture rate.
    """
    if (not samples or len(samples) < 2 or type(cadence) not in (int, float)
            or not math.isfinite(cadence) or not 0 < cadence <= 0.2):
        return [None] * seconds
    times = [row[0] for row in samples]
    if (any(type(t) not in (int, float) or not math.isfinite(t) for t in times)
            or any(b <= a for a, b in zip(times, times[1:]))):
        return [None] * seconds
    visible = [{row['id']: row for row in rows if row.get('visible') is True}
               for _time, rows in samples]
    common = set(visible[0]).intersection(*(set(rows) for rows in visible[1:]))
    if not common:
        return [None] * seconds
    result = []
    for second in range(seconds):
        left = [i for i, t in enumerate(times) if t <= second]
        right = [i for i, t in enumerate(times) if t >= second + 1]
        if (not left or not right or any(a < second + 1 and b > second for a, b in gaps)):
            result.append(None)
            continue
        lo, hi = left[-1], right[0]
        if any(times[i + 1] - times[i] >= max(0.35, 3.5 * cadence) for i in range(lo, hi)):
            result.append(None)
            continue
        caps = []
        for identity in common:
            rows = [visible[i][identity] for i in range(lo, hi + 1)]
            states = [(r.get('configured_fps'), r.get('effective_rate'), r.get('_hold_paced')) for r in rows]
            configured, effective, held = states[0]
            if (any(r.get('counter_reset') for r in rows) or any(state != states[0] for state in states)
                    or held is not True or any(type(v) not in (int, float) or not math.isfinite(v)
                                               for v in (configured, effective))
                    or not 0 < effective < min_fps <= configured):
                caps = []
                break
            caps.append(effective)
        result.append(min(caps) if caps else None)
    return result


def label_seconds(distinct, painted, min_fps, *, sampler_gaps=(), sample_offset=0.0,
                  include_unknown=False, held_pacing=None):
    """Low-picture-rate seconds: app freeze, capture loss, app slow, or unknown.

    App freeze: in that second the app itself painted fewer frames than the
    threshold AND clearly fewer than it usually does in this clip (below
    :data:`FREEZE_DROP` x its median), i.e. its GUI thread stalled. Otherwise
    the app painted as usual and the recording lost the frames. A delayed
    GUI sampler independently establishes a delivery stall; missing paint
    evidence is unknown and cannot support a capture-loss-only approval.
    ``include_unknown`` adds the unknown-second list to the legacy triple.
    """
    freeze, loss, slow, unknown = [], [], [], []
    known = [value for value in (painted or []) if value is not None]
    typical = float(np.median(known)) if known else None
    for second, value in enumerate(distinct):
        if value >= min_fps:
            continue
        if any(a + sample_offset < second + 1 and b + sample_offset > second
               for a, b in sampler_gaps):
            freeze.append(second)
        elif (painted is not None and second < len(painted)
              and painted[second] is not None and painted[second] < min_fps):
            # The app itself painted too few frames: a stall (a drop below its
            # usual pace) or a steadily slow screen.
            reference = typical
            if held_pacing is not None and second < len(held_pacing) and held_pacing[second] is not None:
                reference = min(reference, held_pacing[second])
            (freeze if painted[second] < FREEZE_DROP * reference else slow).append(second)
        elif painted is not None and second < len(painted) and painted[second] is not None:
            loss.append(second)
        else:
            unknown.append(second)
    return (freeze, loss, slow, unknown) if include_unknown else (freeze, loss, slow)


def app_stall_windows(samples, min_gap=0.06):
    """Sample intervals in which the app painted nothing (its GUI thread was busy)."""
    if not samples:
        return []
    return [(a[0], b[0]) for a, b in zip(samples, samples[1:]) if b[1] == a[1] and b[0] - a[0] >= min_gap]


def lost_in_stalls(changed, fps, windows, slack=0.12, offset=0.0):
    """Unchanged video frames, and how many fall in (or just after) an app stall.

    ``offset``: seconds the video starts before the paint samples' zero (the
    recorder is already running when the scene clock starts).
    """
    lost = [i for i, c in enumerate(changed) if i and not c]
    inside = [i for i in lost if any(a <= i / fps - offset <= b + slack for a, b in windows)]
    return lost, inside


def best_stall_offset(changed, fps, windows, limit=1.0):
    """The video/sample clock offset that explains most unchanged frames by app stalls."""
    best = (0, 0.0)
    for k in range(int(limit * fps) + 1):
        offset = k / fps
        hits = len(lost_in_stalls(changed, fps, windows, offset=offset)[1])
        if hits > best[0]:
            best = (hits, offset)
    return best[1]


def cursor_steps(track):
    """Displacements between consecutive frames where the cursor was found in both."""
    return [(b['frame'], float(np.hypot(b['x'] - a['x'], b['y'] - a['y'])))
            for a, b in zip(track, track[1:]) if a['found'] and b['found']]


def frame_rate(video: Path) -> float:
    import subprocess
    rate = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                    'stream=r_frame_rate', '-of', 'csv=p=0', str(video)], text=True).strip()
    num, _, den = rate.partition('/')
    return float(num) / float(den or 1)


def evaluate(video: Path, *, receipt: dict | None = None, fps: float | None = None,
             field_window: tuple[float, float] | None = None, cursor_theme: str = 'Yaru',
             cursor_size: int = 32, **overrides) -> dict:
    """Measure ``video`` and decide pass/fail. Returns the full result."""
    limits = {**DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
    video = Path(video)
    fps = fps or frame_rate(video)
    if limits['min_fps'] is None:
        limits['min_fps'] = round(MIN_DISTINCT_FRACTION * fps, 2)
    width, height = size(video)
    seconds = duration(video)
    track = track_cursor(video, cursor_templates(cursor_theme, cursor_size))
    changed, counts, bars = picture_changes(
        video, track, level=limits['change_level'], pixels=limits['change_pixels'],
        cursor_box=limits['cursor_box'], top_bar_height=limits['top_bar_height'],
        top_bar_every=limits['top_bar_every'])
    n = len(changed)
    start, stop = 1, n
    if field_window is not None:
        start = max(1, int(round(field_window[0] * fps)))
        stop = min(n, int(round(field_window[1] * fps)))
    window_frames = max(1, stop - start)
    distinct_fps = float(np.sum(changed[start:stop])) * fps / window_frames
    md5, exact = exact_distinct(video)
    steps = cursor_steps(track)
    bound = limits['jump_margin'] * limits['peak_per_frame'] * width * 30.0 / fps
    jumps = [{'frame': f, 'step_px': round(s, 1)} for f, s in steps if s > bound]
    found = float(np.mean([t['found'] for t in track])) if track else 0.0
    bar_frames = [b['frame'] for b in bars if b['visible']]
    whole = int(n // fps)
    per_second = [int(np.sum(changed[max(1, int(k * fps)):int((k + 1) * fps)])) for k in range(whole)]
    facts = receipt or {}
    sampler_gaps, sampler_evidence = gui_sampler_gap_evidence(facts)
    painted = app_painted_per_second(facts.get('app_paint_samples'), whole, gaps=sampler_gaps)
    counter_semantics = facts.get('app_paint_counter_semantics', 'legacy_visible_widget_sum')
    diagnostic_painted = painted
    diagnostic_scope = 'legacy_aggregate_estimate'
    if counter_semantics == 'observed_per_widget_deltas':
        diagnostic_painted = stable_field_painted_per_second(
            facts.get('app_field_widget_samples'), whole, gaps=sampler_gaps)
        diagnostic_scope = 'minimum_rate_of_stably_visible_fields'
    stalls = sorted(set(app_stall_windows(facts.get('app_paint_samples')) + sampler_gaps))
    stall_offset = best_stall_offset(changed, fps, stalls) if stalls else 0.0
    lost, lost_in_app_stall = lost_in_stalls(changed, fps, stalls, offset=stall_offset)
    held_pacing = (sampled_hold_pacing_per_second(
        facts.get('app_field_widget_samples'), whole, limits['min_fps'],
        cadence=facts.get('app_gui_sampler_interval_s'), gaps=sampler_gaps)
        if counter_semantics == 'observed_per_widget_deltas' else [None] * whole)
    admission_freeze, _, _, _ = label_seconds(
        per_second, diagnostic_painted, limits['min_fps'], sampler_gaps=sampler_gaps,
        sample_offset=stall_offset, include_unknown=True)
    freeze, loss, slow, unknown = label_seconds(
        per_second, diagnostic_painted, limits['min_fps'], sampler_gaps=sampler_gaps,
        sample_offset=stall_offset, include_unknown=True, held_pacing=held_pacing)
    reasons = []
    if distinct_fps < limits['min_fps']:
        where = []
        if freeze:
            where.append(f'app freeze in second(s) {freeze}')
        if slow:
            where.append(f'app painting below the rate in second(s) {slow}')
        if loss:
            where.append(f'capture loss in second(s) {loss}')
        if unknown:
            where.append(f'unproven app paint rate in second(s) {unknown}')
        kinds = [k for k, v in (('app freeze', freeze), ('app slow', slow),
                                ('capture loss', loss), ('unlabelled', unknown)) if v]
        label = ' + '.join(kinds) or 'unlabelled'
        if stalls:
            where.append(f'{len(lost_in_app_stall)} of {len(lost)} unchanged frames fall in the app\'s '
                         f'{len(stalls)} paint stalls')
        reasons.append(f'choppy ({label}): {distinct_fps:.1f} distinct fps < {limits["min_fps"]}'
                       + (f' [{"; ".join(where)}]' if where else ''))
    elif admission_freeze:
        reasons.append(f'app freeze: second(s) {admission_freeze} (paint scope: {diagnostic_scope})')
    if jumps:
        reasons.append(f'pointer jump: {len(jumps)} frame step(s) over {bound:.0f} px '
                       f'(max {max(j["step_px"] for j in jumps):.0f} px)')
    if found < limits['min_cursor_found']:
        reasons.append(f'pointer lost: found in {100 * found:.0f} % of frames')
    if bar_frames:
        reasons.append(f'top bar visible in sampled frames {bar_frames[:5]}')
    bypasses = []
    if receipt is not None:
        bypasses = receipt.get('adapter_bypasses') or []
        if bypasses:
            reasons.append(f'{len(bypasses)} scene action(s) bypassed the human pointer')
    longest_static = 0
    run = 0
    for c in changed[start:stop]:
        run = 0 if c else run + 1
        longest_static = max(longest_static, run)
    return {
        'video': str(video), 'size': [width, height], 'fps': round(fps, 3),
        'duration_s': round(seconds, 3), 'frames': n,
        'passed': not reasons, 'reasons': reasons, 'limits': limits,
        'field_window_s': list(field_window) if field_window else None,
        'distinct_fps': round(distinct_fps, 2),
        'distinct_fraction': round(float(np.mean(changed[start:stop])), 4) if stop > start else None,
        'longest_unchanged_run_frames': longest_static,
        'distinct_per_second': per_second, 'app_painted_per_second': painted,
        'app_paint_counter_semantics': counter_semantics,
        'app_paint_counter_scope': facts.get('app_paint_counter_scope', 'legacy_visible_widget_sum'),
        'app_diagnostic_painted_per_second': diagnostic_painted,
        'app_diagnostic_paint_scope': diagnostic_scope,
        'app_diagnostic_sampled_hold_pacing_per_second': held_pacing,
        'app_gui_sampler_gaps_s': [[round(a, 3), round(b, 3)] for a, b in sampler_gaps],
        'app_gui_sampler_gap_evidence': sampler_evidence,
        'app_admission_freeze_seconds': admission_freeze,
        'app_freeze_seconds': freeze, 'app_slow_seconds': slow, 'capture_loss_seconds': loss,
        'app_unknown_seconds': unknown,
        'app_stall_windows_s': [[round(a, 3), round(b, 3)] for a, b in stalls],
        'app_stall_video_offset_s': round(stall_offset, 3),
        'unchanged_frames': len(lost), 'unchanged_frames_in_app_stalls': len(lost_in_app_stall),
        'exact_md5_distinct_fraction': round(sum(exact) / max(1, len(exact)), 4),
        'cursor': {
            'found_fraction': round(found, 4),
            'shapes_seen': sorted({t['shape'] for t in track if t['found']}),
            'max_step_px': round(max((s for _, s in steps), default=0.0), 1),
            'p95_step_px': round(float(np.percentile([s for _, s in steps], 95)), 1) if steps else None,
            'bound_px': round(bound, 1), 'jumps': jumps[:50], 'jump_count': len(jumps),
        },
        'top_bar': {'hidden': not bar_frames, 'sampled_frames': len(bars),
                    'visible_in': bar_frames, 'first_sample': bars[0] if bars else None},
        'adapter_bypasses': bypasses,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, help='The clip receipt (adapter bypasses are read from it)')
    parser.add_argument('--out', type=Path, help='Write the result JSON here')
    parser.add_argument('--field-window', help='START,END seconds where the field must animate (default: whole clip)')
    parser.add_argument('--min-fps', type=float)
    parser.add_argument('--jump-margin', type=float)
    parser.add_argument('--min-cursor-found', type=float)
    parser.add_argument('--cursor-theme', default='Yaru')
    parser.add_argument('--cursor-size', type=int, default=32)
    args = parser.parse_args(argv)
    window = tuple(float(v) for v in args.field_window.split(',')) if args.field_window else None
    receipt = json.loads(args.receipt.read_text()) if args.receipt else None
    result = evaluate(args.video, receipt=receipt, field_window=window, cursor_theme=args.cursor_theme,
                      cursor_size=args.cursor_size, min_fps=args.min_fps, jump_margin=args.jump_margin,
                      min_cursor_found=args.min_cursor_found)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + '\n')
    print(text)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
