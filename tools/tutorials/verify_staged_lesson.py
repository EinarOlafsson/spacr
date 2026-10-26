#!/usr/bin/env python3
"""Check a real staged English lesson in Chromium without uploading its media.

The live player code is served unchanged. Only catalog/media locations are
redirected to the staging directory; untouched lessons use their existing files.
This verifies playback/synchronization, not human pronunciation or translation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import functools
import hashlib
import http.server
import json
import math
import re
import threading
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_ROOT), str(_ROOT / 'tools'), str(Path(__file__).resolve().parent)):
    if _path in sys.path:
        sys.path.remove(_path)
sys.path[:0] = [str(_ROOT), str(_ROOT / 'tools'), str(Path(__file__).resolve().parent)]

from playwright.sync_api import sync_playwright
from stage_lesson import DEFAULT_STAGE, REPO, read, write

WORKSPACE = DEFAULT_STAGE.parent


def staged_idle_binding(lesson, master, timing_path, policy, *, required=False):
    """Bind the selected policy to actual staged bytes, never drop stale fences."""
    from idle_tail_policy import bind_policy
    if (not isinstance(policy, dict) or set(policy) != {'schema', 'lessons'}
            or policy['schema'] != 1 or not isinstance(policy['lessons'], list)
            or any(not isinstance(entry, dict) for entry in policy['lessons'])):
        raise ValueError('Invalid staged idle-tail policy')
    selected = [entry for entry in policy['lessons'] if entry.get('lesson') == lesson['id']]
    timing = read(timing_path)
    records = [dict(path='media_host/' + lesson[key], sha256=hashlib.sha256(path.read_bytes()).hexdigest())
               for key, path in [('silent', master), ('visual_timings', timing_path)]]
    bound = bind_policy(dict(schema=1, lessons=selected), records, [lesson], lambda _path: timing)[0]
    expected = deepcopy(timing)
    for fence in bound.get('native_idle_tails', {}).get('scenes', []):
        matches = [scene for scene in expected['scenes'] if scene['visual'] == fence['visual']
                   and abs(scene['speech_start'] - fence['source_start']) <= 1e-7
                   and abs(scene['scene_end'] - fence['source_end']) <= 1e-7]
        # bind_policy already enforces exactly one immutable interval match.
        matches[0]['idle_tail_start'] = fence['idle_tail_start']
    if required and any('idle_tail_start' not in scene for scene in expected['scenes']):
        raise ValueError('One-second acceptance requires reviewed fences for every scene')
    return bound, expected


def require_loaded_idle_timings(actual, expected, spoken):
    """Check the real player's policy and complete selected-voice speech fit."""
    if actual != expected:
        raise ValueError('Loaded native clocks or idle fences differ from bound staged sources')
    if len(spoken['scenes']) != len(expected['scenes']):
        raise ValueError('Native idle acceptance requires every spoken scene')
    for scene, speech in zip(expected['scenes'], spoken['scenes']):
        fence = scene.get('idle_tail_start')
        if (type(fence) not in (int, float) or not math.isfinite(fence)
                or not 0 <= fence <= float(speech['speech_end']) - float(speech['speech_start'])):
            raise ValueError('Reviewed action fence exceeds complete selected-voice speech')


def check_related_links(actual, expected):
    """Require exactly the authored links, including a genuinely linkless lesson."""
    if sorted(set(actual)) != sorted(set(expected)):
        raise ValueError(f'Related lesson links differ: {actual!r} != {expected!r}')


def check_native_boundary_playback(page, visual, spoken):
    """Measure actual media clocks across a recorded scene boundary at normal speed."""
    if (visual.get('complete_native_clips') is True
            or visual.get('preserve_visual_intervals') is True):
        return check_complete_native_boundary_playback(page, visual, spoken)
    candidates = [i for i in range(len(visual['scenes']) - 1)
                  if visual['scenes'][i].get('native_live_video')
                  and visual['scenes'][i + 1].get('native_live_video')]
    if not candidates:
        raise ValueError('Boundary verification needs two consecutive recorded scenes')
    index = max(candidates, key=lambda i: visual['scenes'][i]['duration'] - spoken['scenes'][i]['duration'])
    target = float(spoken['scenes'][index]['scene_end']) - .7
    page.evaluate('(target) => seekTo(target)', target)
    page.wait_for_function('!videoClockCorrectionPending && !elements.video.seeking && !elements.audio.paused', timeout=15000)
    page.evaluate('''() => {
        window.__narrationInterruptions=[];
        if (!window.__narrationListeners) {
            window.__narrationListeners=true;
            for (const type of ['pause','seeking','waiting','stalled'])
                elements.audio.addEventListener(type, () => window.__narrationInterruptions.push(
                    {type, audio:elements.audio.currentTime, video:elements.video.currentTime}));
        }
        window.__actualNativeTrace=[];
        window.__actualNativeTimer=setInterval(()=>window.__actualNativeTrace.push({
            audio:elements.audio.currentTime, video:elements.video.currentTime,
            rate:elements.video.playbackRate, audioRate:elements.audio.playbackRate,
            seeking:elements.video.seeking, correcting:videoClockCorrectionPending,
            error:elements.video.error?.message || elements.audio.error?.message || null}),50);
    }''')
    page.wait_for_timeout(3500)
    trace=page.evaluate('window.__actualNativeTrace')
    interruptions=page.evaluate('window.__narrationInterruptions')
    if interruptions:
        # The narration is the master clock: a picture skip at a scene
        # boundary must never pause or seek it (a paused output stream drops
        # the first word of the next sentence on real audio hardware).
        raise AssertionError({'narration_interrupted_at_boundary': interruptions})
    page.evaluate('clearInterval(window.__actualNativeTimer)')
    settled=[]
    for row in trace:
        if row['seeking'] or row['correcting']:
            continue
        scene=next((i for i, source in enumerate(spoken['scenes'])
                    if row['audio'] < float(source['scene_end'])), len(spoken['scenes'])-1)
        expected=float(visual['scenes'][scene]['speech_start']) + row['audio'] - float(spoken['scenes'][scene]['speech_start'])
        if (row['rate'] != 1 or row['audioRate'] != 1 or row['error']
                or abs(row['video']-expected) >= .5):
            raise AssertionError({'scene':scene+1,'expected_from_staged_timing':expected,'actual':row})
        settled.append(dict(row,scene=scene+1,expected_from_staged_timing=expected))
    if len(settled)<20 or len({row['scene'] for row in settled})<2:
        raise AssertionError('Playback did not traverse two native scenes with settled clocks')
    pairs=[(a,b) for a,b in zip(settled,settled[1:])
           if a['scene']==b['scene'] and b['audio']-a['audio']>.02]
    if not pairs or any(abs((b['video']-a['video'])-(b['audio']-a['audio']))>=.2 for a,b in pairs):
        raise AssertionError('Recorded elapsed motion differs from narration elapsed time')
    return {'checked_boundary_after_scene':index+1,'trace':settled,'actual_video_speed':1,
            'narration_never_paused_at_boundary':True,
            'independent_staged_timing_comparison':True,'native_animation_stretched_or_looped':False}


def check_complete_native_trace(trace, events, visual, spoken, index, final=False, *, require_gap=False):
    """Check actual sound gaps and only the selected voice's planned idle seek."""
    scene, speech = visual['scenes'][index], spoken['scenes'][index]
    start, end = float(scene['speech_start']), float(scene['scene_end'])
    audio_start, audio_end = float(speech['speech_start']), float(speech['speech_end'])
    last_sound = float(speech['sentences'][-1]['audible_end'])
    following = None if final else spoken['scenes'][index + 1]
    next_sound = None if final else float(following['sentences'][0]['audible_start'])
    next_lead = 0 if final else next_sound - float(following['speech_start'])
    expected_gap = 1.0 if final else min(next_sound - last_sound, 1.0)
    cut_at = start + last_sound - audio_start + expected_gap - next_lead
    fence = scene.get('idle_tail_start')
    planned_cut = (type(fence) in (int, float) and math.isfinite(fence)
                   and 0 <= fence <= audio_end - audio_start <= cut_at - start and cut_at < end)
    if len(trace) < 20 or last_sound > audio_end or next_lead < 0:
        raise AssertionError('Insufficient or malformed sound-boundary trace')
    seeks = [event for event in events if event['type'] == 'seeking']
    if len(seeks) != int(planned_cut):
        raise AssertionError({'missing_or_unplanned_seek': events, 'expected_idle_cut': planned_cut})
    for event in seeks:
        if abs(event['video'] - end) > .15:
            raise AssertionError({'idle_seek_wrong_destination': event, 'expected': end})
    starts = [event for event in events if event['type'] == 'idle-cut']
    if len(starts) > int(planned_cut) or any(event['audio'] < audio_end for event in starts):
        raise AssertionError({'unfinished_or_duplicate_idle_transition': starts})
    # Decoder events are permitted only around the one planned idle transition.
    # The independently measured sound gap below still rejects a long stall.
    for event in events:
        if event['type'] in ('seeking', 'idle-cut'):
            continue
        if (event['type'] not in ('pause', 'waiting', 'stalled') or not seeks
                or 'wall' not in event or abs(event['wall'] - seeks[0].get('wall', -1e9)) > 500):
            raise AssertionError({'unplanned_video_interruption': event})
    for row in trace:
        quiet_only_rate = (not final and planned_cut and audio_end <= row['audio']
                           <= float(following['speech_start']) - .05
                           and .0625 <= row['audioRate'] <= 1)
        if row['rate'] != 1 or (row['audioRate'] != 1 and not quiet_only_rate) or row['error']:
            raise AssertionError({'invalid_native_clock': row})
        if row['videoPaused'] and not row['videoEnded'] and not (row['seeking'] or row['correcting']):
            quiet_hold = (planned_cut and seeks and abs(row['video'] - end) < .15
                          and 0 <= row['wall'] - seeks[0].get('wall', -1e9) <= 1150)
            if not quiet_hold:
                raise AssertionError({'unplanned_video_pause': row})
    settled = [row for row in trace if not row['seeking'] and not row['correcting']]
    for left, right in zip(settled, settled[1:]):
        crossed = left['video'] < end - .15 and right['video'] >= end - .15
        if crossed and planned_cut:
            if not seeks:
                raise AssertionError('Video jumped without a planned seek event')
            continue
        elapsed = (right['wall'] - left['wall']) / 1000
        expected = min(elapsed, max(0, end - left['video'])) if right['videoEnded'] else elapsed
        if abs(right['video'] - left['video'] - expected) >= .25:
            raise AssertionError({'unplanned_video_jump_or_stall': [left, right]})
    # A completed actual speech clip must be observed BEFORE any automatic seek.
    if planned_cut:
        before_seek = [row for row in settled if row['video'] < end - .15
                       and row['wall'] <= seeks[0].get('wall', -1e9)]
        if not any(row['audio'] >= audio_end for row in before_seek) and not starts:
            raise AssertionError('Idle seek occurred before actual complete speech was observed')
    def crossing(threshold, upper=None):
        for left, right in zip(trace, trace[1:]):
            if (left['seeking'] or right['seeking'] or left['correcting'] or right['correcting']
                    or (upper is not None and right['audio'] >= upper)):
                continue
            delta = right['audio'] - left['audio']
            wall = (right['wall'] - left['wall']) / 1000
            if left['audio'] <= threshold <= right['audio'] and 0 < delta <= wall + .1:
                return left['wall'] + (threshold - left['audio']) / delta * (right['wall'] - left['wall'])
        raise AssertionError('Actual audible boundary was not continuously observed')
    last_wall = crossing(last_sound, None if final else float(following['speech_start']))
    if final:
        ending = next((row for row in trace if row['videoEnded'] and abs(row['video'] - end) < .15), None)
        if ending is None:
            raise AssertionError('Real final video EOF was not observed')
        next_wall = ending['wall']
    else:
        next_wall = crossing(next_sound)
        resumed = [row for row in settled if end + .15 < row['video'] < end + 1.4
                   and not row['audioPaused']]
        if len(resumed) < 5:
            raise AssertionError('Next narration did not resume after its planned visual transition')
        for row in resumed:
            expected = float(following['speech_start']) + row['video'] - end
            if abs(row['audio'] - expected) >= .5:
                raise AssertionError({'next_narration_wrong_offset': row, 'expected': expected})
    gap = (next_wall - last_wall) / 1000
    # Shorten long gaps without padding an already brief source pause. Final
    # EOF has no following sound: retain its existing one-second end policy.
    if (planned_cut or require_gap) and not expected_gap - .15 <= gap <= expected_gap + .15:
        raise AssertionError({'actual_sound_gap_outside_policy': gap,
                              'expected_sound_gap_seconds': expected_gap,
                              'planned_idle_cut': planned_cut})
    return {'checked_boundary_after_scene': index + 1, 'trace': trace,
        'selected_visual_tail_observed': True, 'recorded_tail_observed': True,
        'entire_recorded_tail_played': not planned_cut,
        'narration_paused_for_recorded_tail': not planned_cut,
        'idle_padding_skipped': planned_cut, 'actual_sound_gap_seconds': gap,
        'actual_speech_end_observed_before_idle_seek': True,
        'next_narration_resumed': not final, 'final_visual_eof_observed': final,
        'final_native_eof_observed': final and scene.get('native_live_video') is True,
        'actual_video_speed': 1, 'native_animation_stretched_or_looped': False}


def check_complete_native_boundary_playback(page, visual, spoken, *, all_boundaries=False,
                                          require_gap=False, raw_path=None):
    """Observe complete speech, one-second idle cuts, and actual final EOF."""
    preserve = visual.get('preserve_visual_intervals') is True
    candidates = [i for i in range(len(visual['scenes']) - 1)
                  if visual['scenes'][i].get('native_live_video')
                  and (preserve or visual['scenes'][i + 1].get('native_live_video'))
                  and (all_boundaries or float(visual['scenes'][i]['duration'])
                       - (float(spoken['scenes'][i]['speech_end'])
                          - float(spoken['scenes'][i]['speech_start'])) > .8)]
    if not candidates or (not preserve and not visual['scenes'][-1].get('native_live_video')):
        raise ValueError('Complete native verification needs a boundary and final recorded tail')
    index = max(candidates, key=lambda i: float(visual['scenes'][i]['duration'])
                - (float(spoken['scenes'][i]['speech_end'])
                   - float(spoken['scenes'][i]['speech_start'])))
    results = []
    observed = []
    selected = candidates if all_boundaries else [index]
    for current, final in [*((i, False) for i in selected), (len(visual['scenes']) - 1, True)]:
        scene, speech = visual['scenes'][current], spoken['scenes'][current]
        speech_length = float(speech['speech_end']) - float(speech['speech_start'])
        target = float(scene['speech_start']) + max(0, speech_length - .7)
        # Setup seek precedes observation; only the independently planned
        # idle transition may seek during the measured interval.
        page.evaluate("""async target => {
            elements.video.pause();
            elements.video.currentTime = target;
            await new Promise(resolve => {
                if (!elements.video.seeking) resolve();
                else elements.video.addEventListener('seeked', resolve, {once:true});
            });
            syncCompleteNativeNarration(true);
            await elements.video.play();
        }""", target)
        page.wait_for_function('target => !videoClockCorrectionPending && !elements.video.seeking && '
                               '!elements.audio.paused && !elements.video.paused && '
                               'elements.video.readyState >= 3 && elements.video.currentTime > target + .15',
                               arg=target, timeout=15000)
        page.evaluate("""() => {
            window.__completeNativeTrace=[];
            window.__completeNativeEvents=[];
            window.__completeNativeObserve=true;
            if (!window.__completeNativeListeners) {
                window.__completeNativeListeners=true;
                const begin = beginNativeIdleTransition;
                beginNativeIdleTransition = (cut,state,clock,index) => {
                    if (window.__completeNativeObserve)
                        window.__completeNativeEvents.push({type:'idle-cut',wall:performance.now(),
                            video:elements.video.currentTime,audio:elements.audio.currentTime});
                    return begin(cut,state,clock,index);
                };
                for (const type of ['seeking','pause','waiting','stalled'])
                    elements.video.addEventListener(type, () => {
                        if (window.__completeNativeObserve && (type === 'seeking' || !elements.video.ended))
                            window.__completeNativeEvents.push({type,video:elements.video.currentTime,wall:performance.now(),audio:elements.audio.currentTime});
                    });
            }
            window.__completeNativeTimer=setInterval(() => window.__completeNativeTrace.push({
                wall:performance.now(), audio:elements.audio.currentTime,
                video:elements.video.currentTime, rate:elements.video.playbackRate,
                audioRate:elements.audio.playbackRate, audioPaused:elements.audio.paused,
                videoPaused:elements.video.paused, videoEnded:elements.video.ended,
                seeking:elements.video.seeking, correcting:videoClockCorrectionPending,
                error:elements.video.error?.message || elements.audio.error?.message || null
            }),50);
        }""")
        try:
            timeout = int((float(scene['scene_end']) - target + 6) * 1000)
            if final:
                page.wait_for_function('elements.video.ended', timeout=timeout)
            else:
                page.wait_for_function('end => elements.video.currentTime > end + 1.2',
                                       arg=float(scene['scene_end']), timeout=timeout)
            trace = page.evaluate('window.__completeNativeTrace')
            if final:
                # An ended event may occur between the final periodic sample
                # and this read. Capture the actual EOF state, never synthesize it.
                trace.append(page.evaluate("""() => ({wall:performance.now(),
                    audio:elements.audio.currentTime,video:elements.video.currentTime,
                    rate:elements.video.playbackRate,audioRate:elements.audio.playbackRate,
                    audioPaused:elements.audio.paused,videoPaused:elements.video.paused,
                    videoEnded:elements.video.ended,seeking:elements.video.seeking,
                    correcting:videoClockCorrectionPending,
                    error:elements.video.error?.message || elements.audio.error?.message || null})"""))
            events = page.evaluate('window.__completeNativeEvents')
        finally:
            page.evaluate('clearInterval(window.__completeNativeTimer); window.__completeNativeObserve=false')
        observed.append(dict(scene=current + 1, final=final, trace=trace, events=events))
        if raw_path is not None:
            write(raw_path, dict(visual=visual, spoken=spoken, observations=observed))
        results.append(check_complete_native_trace(trace, events, visual, spoken, current, final,
                                                   require_gap=require_gap))
    result = {'complete_native_clips': visual.get('complete_native_clips') is True,
            'preserve_visual_intervals': visual.get('preserve_visual_intervals') is True,
            'boundary': results[0], 'final_tail': results[-1],
            'actual_video_speed': 1, 'independent_wall_clock_comparison': True,
            'native_animation_stretched_or_looped': False}
    if all_boundaries:
        result['boundaries'] = results[:-1]
    return result


def find_host_lesson(lessons, navigation, host_key):
    """Allow an absent parent lesson only when navigation records that gap."""
    hosts = [item for item in lessons
             if item.get('app_key') == host_key and not item.get('host_app_key')]
    if len(hosts) == 1:
        return hosts[0]
    missing = {item['app_key'] for item in navigation['missing_tutorials']}
    if hosts or host_key not in missing:
        raise ValueError(f'Unexpected parent tutorial coverage for {host_key}')
    return None


class Handler(http.server.SimpleHTTPRequestHandler):
    web_lesson = None
    player_source = None

    def log_message(self, *args):
        pass

    def translate_path(self, path):
        candidate = Path(super().translate_path(path))
        player = WORKSPACE / 'web'
        if candidate.is_relative_to(player):
            if candidate == player / 'app_v2.js' and self.player_source is not None:
                return str(self.player_source)
            return str(REPO / 'docs/source/_extra/tutorials' / candidate.relative_to(player))
        staged_media = DEFAULT_STAGE / 'production'
        if self.web_lesson:
            relative = Path(self.web_lesson) / 'video' / f'{self.web_lesson}_silent.mp4'
            if candidate == staged_media / relative:
                return str(DEFAULT_STAGE / 'web-renditions' / relative)
        if candidate.is_relative_to(staged_media) and not candidate.is_file():
            return str(WORKSPACE / 'production' / candidate.relative_to(staged_media))
        return str(candidate)

    def send_head(self):
        # Match the production host's byte-range contract. Without it Chrome
        # can clamp a chapter seek back to zero even with metadata loaded.
        self._range_remaining = None
        requested = self.headers.get('Range')
        path = Path(self.translate_path(self.path))
        if not requested or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
        if not match or not any(match.groups()):
            self.send_error(416)
            return None
        left, right = match.groups()
        start = int(left) if left else max(0, size - int(right))
        end = min(size - 1, int(right)) if left and right else size - 1
        if start > end or start >= size:
            self.send_error(416)
            return None
        stream = path.open('rb')
        stream.seek(start)
        self._range_remaining = end - start + 1
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Content-Length', str(self._range_remaining))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        try:
            if self._range_remaining is None:
                return super().copyfile(source, outputfile)
            while self._range_remaining:
                chunk = source.read(min(65536, self._range_remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                self._range_remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Native video seeking cancels its preceding range request.
            pass


def main(argv=None):
    global DEFAULT_STAGE, WORKSPACE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE,
                        help='Isolated tutorial stage; does not modify the shared stage')
    parser.add_argument('--lesson', default='05_home')
    parser.add_argument('--language', default='en')
    parser.add_argument('--voice', default='af_heart')
    parser.add_argument('--caption-language', help='Independently test a staged caption language with this voice')
    parser.add_argument('--sentence-cues', action='store_true',
                        help='Check every native English caption at its recorded speech midpoint')
    parser.add_argument('--retained-media', action='store_true',
                        help='Require unchanged original catalogs and media, with no staged override')
    parser.add_argument('--retain-visual-from', type=Path,
                        help='Check a bounded visual refresh against this published baseline')
    parser.add_argument('--web-rendition', action='store_true',
                        help='Check the verified private 1440p copy, preserving original browser reports')
    parser.add_argument('--candidate-player', action='store_true',
                        help='Serve the exact private authoring player source, including native timing and lesson redirects; no publication')
    parser.add_argument('--native-boundary-clocks', action='store_true',
                        help='Measure actual normal-speed media clocks across a recorded scene boundary')
    parser.add_argument('--idle-tail-policy', type=Path,
                        help='Explicit reviewed source-bound policy; defaults to the canonical policy')
    parser.add_argument('--require-idle-cuts', action='store_true',
                        help='Require bound fences and measure every native boundary and final one-second gap')
    parser.add_argument('--output-suffix', default='',
                        help='Fresh report suffix to preserve earlier playback evidence')
    args = parser.parse_args(argv)
    if args.retain_visual_from is not None and args.retained_media:
        parser.error('A visual refresh cannot also retain unchanged whole media')
    if args.sentence_cues and (args.language != 'en' or args.caption_language):
        parser.error('--sentence-cues requires English narration and its native captions')
    if args.require_idle_cuts and not (args.native_boundary_clocks and args.candidate_player):
        parser.error('--require-idle-cuts requires --native-boundary-clocks and --candidate-player')
    if args.output_suffix and not re.fullmatch('[a-zA-Z0-9_-]+', args.output_suffix):
        parser.error('Use a safe output suffix')
    DEFAULT_STAGE = args.stage.resolve()
    WORKSPACE = DEFAULT_STAGE.parent
    retained = None
    if args.retained_media:
        from verify_retained_media import retained_media_sources
        from check_completed_matrix import voice_matrix
        retained = retained_media_sources(
            DEFAULT_STAGE, WORKSPACE, REPO / 'docs/source/_extra/tutorials/catalog',
            args.lesson, voice_matrix(WORKSPACE / 'tools/render_all_voices.py'))
    english = read(DEFAULT_STAGE / 'catalog/lessons_en.json')
    player_path = (REPO / 'tools/tutorials/authoring/web/app_v2.js' if args.candidate_player
                   else REPO / 'docs/source/_extra/tutorials/app_v2.js')
    player_bytes = player_path.read_bytes()
    lesson = next(item for item in english['lessons'] if item['id'] == args.lesson)
    if args.caption_language:
        from catalog_preflight import validate_caption_structure
        filename = f'captions_{args.caption_language}.json'
        caption_path = DEFAULT_STAGE / 'catalog' / filename
        if not caption_path.exists():
            caption_path = DEFAULT_STAGE / 'catalog' / f'lessons_{args.caption_language}.json'
        validate_caption_structure(english, read(caption_path), args.caption_language)
    from build_navigation import build
    navigation = build(english)
    expected_native = None
    for item in english['lessons']:
        item['poster'] = f"{item['id']}/poster.jpg"
        item['silent'] = f"{item['id']}/video/{item['id']}_silent.mp4"
        if item['id'] == args.lesson:
            if args.retain_visual_from is not None:
                from retained_visual_refresh import checked_retained_visual_timing
                native_timings = checked_retained_visual_timing(DEFAULT_STAGE, item['id'], args.retain_visual_from)
            else:
                from native_live_timing import checked_native_timing
                native_timings = checked_native_timing(DEFAULT_STAGE, item['id'])
            if native_timings is not None:
                if not args.candidate_player:
                    raise ValueError('Native footage acceptance requires the source-bound candidate player')
                item['visual_timings'] = f"{item['id']}/video/{native_timings.name}"
                from idle_tail_policy import load_policy
                bound, expected_native = staged_idle_binding(
                    item, DEFAULT_STAGE / 'production' / item['silent'], native_timings,
                    load_policy(REPO, args.idle_tail_policy), required=args.require_idle_cuts)
                item.update(bound)
    source = (REPO / 'docs/source/_extra/tutorials/index.html').read_text(encoding='utf-8')
    production = '/' + str(DEFAULT_STAGE.relative_to(WORKSPACE)) + '/production'
    for attribute in ('production-root', 'audio-root', 'video4k-root'):
        source = re.sub(rf'data-{attribute}="[^"]*"', f'data-{attribute}="{production}"', source)
    rendition = None
    if args.web_rendition:
        from check_completed_matrix import digest
        folder = DEFAULT_STAGE / 'web-renditions' / args.lesson
        rendition = read(folder / 'rendition-checks.json')
        if (rendition.get('accepted') is not True or rendition.get('lesson') != args.lesson
                or digest(folder / 'video' / f'{args.lesson}_silent.mp4') != rendition.get('rendition_sha256')):
            raise ValueError('Web rendition is missing or changed since verification')
        source = re.sub(r'data-video4k-root="[^"]*"', 'data-video4k-root=""', source)
    tag = f'{args.language}-{args.voice}'
    if args.caption_language:
        tag += f'-captions-{args.caption_language}'
    if args.sentence_cues:
        tag += '-sentence-cues'
    if args.output_suffix:
        tag += '-' + args.output_suffix
    output = DEFAULT_STAGE / ('browser-web' if args.web_rendition else 'browser') / args.lesson / tag
    if args.output_suffix and output.exists():
        raise FileExistsError('Preserve existing suffixed playback evidence')
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    handler = type('StagedPlayerHandler', (Handler,), {
        'web_lesson': args.lesson if args.web_rendition else None,
        'player_source': player_path})
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
                functools.partial(handler, directory=str(WORKSPACE)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    evidence = {'lesson': args.lesson, 'scope': f'{args.language}/{args.voice} playback and scene links only',
                'uploaded': False, 'translation_or_listening_review': False}
    try:
        with sync_playwright() as engine:
            browser = engine.chromium.launch(
                executable_path='/opt/google/chrome/chrome', headless=True,
                args=['--disable-gpu', '--disable-accelerated-video-decode'])
            context = browser.new_context(viewport={'width': 1440, 'height': 1100})
            base = f'http://127.0.0.1:{server.server_port}'
            context.route(base + '/web/', lambda route: route.fulfill(body=source, content_type='text/html'))
            catalog_js = 'window.SPACR_LESSON_CATALOG = ' + json.dumps(english, ensure_ascii=False) + ';'
            context.route('**/web/lesson_catalog.js*', lambda route: route.fulfill(
                body=catalog_js, content_type='application/javascript'))
            navigation_js = 'window.SPACR_TUTORIAL_NAVIGATION = ' + json.dumps(navigation, ensure_ascii=False) + ';'
            context.route('**/web/module_navigation.js*', lambda route: route.fulfill(
                body=navigation_js, content_type='application/javascript'))
            context.route('**/web/catalog/lessons_en.json*', lambda route: route.fulfill(
                json=english))
            if args.language != 'en':
                localized = read(DEFAULT_STAGE / 'catalog' / f'lessons_{args.language}.json')
                context.route(f'**/web/catalog/lessons_{args.language}.json*',
                              lambda route: route.fulfill(json=localized))
            caption_lesson = None
            if args.caption_language:
                prefix = 'captions' if args.caption_language in {'da', 'de', 'is', 'ko', 'nb', 'sv'} else 'lessons'
                filename = f'{prefix}_{args.caption_language}.json'
                captions = read(DEFAULT_STAGE / 'catalog' / filename)
                caption_lesson = next(item for item in captions['lessons'] if item['id'] == args.lesson)
                context.route(f'**/web/catalog/{filename}*', lambda route: route.fulfill(json=captions))
            page = context.new_page()
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(base + '/web/#lesson=' + args.lesson, wait_until='domcontentloaded')
            player_response = context.request.get(base + '/web/app_v2.js')
            assert player_response.ok
            assert player_response.body() == player_bytes
            evidence['repository_player_sha256'] = hashlib.sha256(player_bytes).hexdigest()
            evidence['repository_player_source'] = str(player_path)
            evidence['candidate_player'] = args.candidate_player
            page.wait_for_function('document.querySelectorAll(".chapter-button").length === ' + str(len(lesson['scenes'])), timeout=60000)
            evidence['navigation_contains_staged_lesson'] = args.lesson in navigation['preserved_lesson_ids']
            assert evidence['navigation_contains_staged_lesson']
            assert page.locator(f'#curriculum [data-lesson="{args.lesson}"]').count() == 1
            host_key = navigation['routes'].get(args.lesson, {}).get('host_app_key') or lesson.get('host_app_key')
            if host_key:
                assert page.locator('#lesson-content').get_attribute('data-app-key') == host_key
                assert page.locator(f'#curriculum [data-host="{host_key}"] [data-lesson="{args.lesson}"]').count() == 1
                host = find_host_lesson(english['lessons'], navigation, host_key)
                if host is not None:
                    assert host['title'] in page.locator('#lesson-route').inner_text()
                else:
                    expected_title = navigation['routes'][args.lesson]['host_title']
                    heading = page.locator(f'#curriculum [data-host="{host_key}"] .module-group-heading')
                    assert heading.inner_text() == expected_title
                    assert page.locator('#lesson-route').is_hidden()
                evidence['host_has_tutorial'] = host is not None
                evidence['host_app_key'] = host_key
            page.wait_for_function('elements.video.readyState >= 2 && elements.audio.readyState >= 2', timeout=60000)
            page.select_option('#language-select', args.language)
            page.wait_for_function('(voice) => [...elements.voice.options].some(o => o.value === voice)',
                                   arg=args.voice, timeout=30000)
            page.select_option('#voice-select', args.voice)
            example_files = list(dict.fromkeys(lesson.get('example_files', [])))
            example_links = page.locator('#prerequisite-copy a[download]')
            assert example_links.all_text_contents() == example_files
            evidence['example_files'] = []
            for name in example_files:
                assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:csv|zip)', name) and len(name) <= 128
                link = page.get_by_role('link', name=name, exact=True)
                assert link.get_attribute('download') == name
                response = context.request.get(base + '/web/' + link.get_attribute('href'))
                assert response.ok
                digest = hashlib.sha256(response.body()).hexdigest()
                expected = REPO / 'docs/source/_extra/tutorials/examples' / name
                assert digest == hashlib.sha256(expected.read_bytes()).hexdigest()
                evidence['example_files'].append({'name': name, 'sha256': digest})
            try:
                page.wait_for_function('(voice) => audioTimings?.voice === voice && narrationAudioAvailable && elements.audio.readyState >= 2',
                                       arg=args.voice, timeout=30000)
            except Exception:
                diagnostic = page.evaluate('({audioSrc: elements.audio.currentSrc, audioReady: elements.audio.readyState, error: elements.audio.error?.message, language: elements.language.value, voice: elements.voice.value, lesson: activeLesson?.id, status: elements.status.textContent, toast: elements.toast.textContent})')
                diagnostic['javascript_errors'] = errors
                write(output / 'language-load-failure.json', diagnostic)
                print(json.dumps(diagnostic, indent=2), flush=True)
                raise
            # The player intentionally prefetches narration into a blob URL.
            # Match its actual bytes, not an assumed URL shape or just the
            # selected label, which could still show a previous voice.
            audio_hash = page.evaluate('''async () => {
                const bytes = await (await fetch(elements.audio.currentSrc)).arrayBuffer();
                const digest = await crypto.subtle.digest('SHA-256', bytes);
                return [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, '0')).join('');
            }''')
            expected_hash = hashlib.sha256(((WORKSPACE if retained else DEFAULT_STAGE) / 'production' / args.lesson /
                            'audio' / args.language / f'{args.voice}.m4a').read_bytes()).hexdigest()
            assert audio_hash == expected_hash, (audio_hash, expected_hash)
            evidence['loaded_audio_sha256'] = audio_hash
            if args.language != 'en' and not args.caption_language:
                translated_lesson = next(item for item in localized['lessons'] if item['id'] == args.lesson)
                texts = page.evaluate('chapterData.map(chapter => chapter.text)')
                assert texts == [scene['narration'] for scene in translated_lesson['scenes']], texts
                evidence['narration_language_scenes_match_staging'] = True
            if caption_lesson is not None:
                page.locator('#caption-settings-button').click()
                page.select_option('#caption-language-select', args.caption_language)
                page.wait_for_function('(language) => effectiveCaptionLanguage() === language && elements.chapters.lang === language',
                                       arg=args.caption_language, timeout=30000)
                texts = page.evaluate('chapterData.map(chapter => chapter.text)')
                assert texts == [scene['narration'] for scene in caption_lesson['scenes']], texts
                assert page.evaluate('elements.language.value') == args.language
                assert page.evaluate('audioTimings.voice') == args.voice
                evidence['caption_language'] = args.caption_language
                evidence['caption_scenes_match_staging'] = True
                # Check the actual generated WebVTT, not just the selector.
                vtt = page.evaluate('async () => await (await fetch(elements.captionTrack.src)).text()')
                assert vtt.startswith('WEBVTT') and '-->' in vtt, vtt[:100]
                evidence['caption_webvtt_sha256'] = hashlib.sha256(vtt.encode()).hexdigest()
                page.locator('#caption-settings-close').click()
            evidence['scene_count'] = page.wait_for_function(
                '(expected) => { const actual = document.querySelectorAll(".chapter-button").length; '
                'return actual === expected && actual; }',
                arg=len(lesson['scenes']), timeout=30000).json_value()
            expected = sorted({identity for scene in lesson['scenes'] for identity in scene.get('related_lessons', [])})
            actual = page.locator('#chapter-list [data-related-lesson]').evaluate_all(
                '(nodes) => [...new Set(nodes.map(n => n.dataset.relatedLesson))].sort()')
            check_related_links(actual, expected)
            page.evaluate('elements.audio.muted = true; elements.video.muted = true')
            seek_index = min(5, len(lesson['scenes']) - 1)
            page.locator('.chapter-button').nth(seek_index).click()
            try:
                page.wait_for_function('(index) => !elements.audio.paused && elements.audio.currentTime > chapterData[index].start + 1', arg=seek_index, timeout=15000)
            except Exception:
                diagnostic = page.evaluate('(index) => ({audio: {time: elements.audio.currentTime, paused: elements.audio.paused, ready: elements.audio.readyState, src: elements.audio.currentSrc, error: elements.audio.error?.message}, video: {time: elements.video.currentTime, paused: elements.video.paused, ready: elements.video.readyState, src: elements.video.currentSrc, error: elements.video.error?.message}, voice: elements.voice.value, available: narrationAudioAvailable, status: elements.status.textContent, chapter: chapterData[index]})', seek_index)
                write(output / 'playback-failure.json', diagnostic)
                print(json.dumps(diagnostic, indent=2), flush=True)
                page.screenshot(path=str(output / 'playback-failure.png'))
                raise
            try:
                page.wait_for_function('''!videoClockCorrectionPending &&
                    !elements.video.seeking && !elements.audio.seeking &&
                    Math.abs(elements.video.currentTime -
                        videoTimeFromAudio(elements.audio.currentTime)) < .5''',
                    timeout=1000, polling=50)
            except Exception:
                diagnostic = page.evaluate('''({
                    correctionPending: videoClockCorrectionPending,
                    audio: {time: elements.audio.currentTime, seeking: elements.audio.seeking,
                        ready: elements.audio.readyState, paused: elements.audio.paused},
                    video: {time: elements.video.currentTime, seeking: elements.video.seeking,
                        ready: elements.video.readyState, paused: elements.video.paused},
                    expectedVideo: videoTimeFromAudio(elements.audio.currentTime),
                    mediaError: elements.video.error?.message || elements.audio.error?.message || null
                })''')
                write(output / 'clock-failure.json', diagnostic)
                print(json.dumps(diagnostic, indent=2), flush=True)
                raise
            clock = page.evaluate('({audio: elements.audio.currentTime, video: elements.video.currentTime, expectedVideo: videoTimeFromAudio(elements.audio.currentTime), audioDuration: elements.audio.duration, videoDuration: elements.video.duration, mediaError: elements.video.error?.message || elements.audio.error?.message || null})')
            assert not clock['mediaError'], clock
            assert abs(clock['video'] - clock['expectedVideo']) < 0.5, clock
            evidence['seek_playback_clocks'] = clock
            if args.native_boundary_clocks:
                native_path=DEFAULT_STAGE/'production'/args.lesson/'video/native-live-timings.json'
                if not args.candidate_player or not native_path.exists():
                    raise ValueError('Native boundary verification needs its accepted sidecar and candidate player')
                spoken=read(DEFAULT_STAGE/'production'/args.lesson/'audio'/args.language/(args.voice+'.json'))
                actual_native = page.evaluate('visualTimings')
                if expected_native is None or actual_native != expected_native:
                    raise ValueError('Loaded native clocks differ from the source-bound stage policy')
                if args.require_idle_cuts:
                    require_loaded_idle_timings(actual_native, expected_native, spoken)
                    evidence['native_boundary_clock_checks'] = check_complete_native_boundary_playback(
                        page, actual_native, spoken, all_boundaries=True, require_gap=True,
                        raw_path=output / 'native-boundary-raw.json')
                else:
                    evidence['native_boundary_clock_checks']=check_native_boundary_playback(page,actual_native,spoken)
            if rendition:
                video = page.evaluate('''async () => {
                    const bytes = await (await fetch(elements.video.currentSrc)).arrayBuffer();
                    const hash = await crypto.subtle.digest('SHA-256', bytes);
                    return {sha256: [...new Uint8Array(hash)].map(x => x.toString(16).padStart(2, '0')).join(''),
                            width: elements.video.videoWidth, height: elements.video.videoHeight};
                }''')
                assert video['sha256'] == rendition['rendition_sha256'], video
                from stage_web_renditions import web_dimensions
                assert [video['width'], video['height']] == list(web_dimensions(args.lesson)), video
                evidence['checked_web_rendition'] = video
            page.evaluate('elements.audio.pause(); elements.video.pause()')
            # Chapter navigation can leave the viewport halfway down the page.
            # Reset it before capturing fixed navigation in a full-page image.
            page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
            page.screenshot(path=str(output / 'desktop.png'), full_page=True)
            page.locator('#transcript-tab').click()
            transcript_links = page.locator('#transcript-list [data-related-lesson]').evaluate_all(
                '(nodes) => nodes.map(n => n.dataset.relatedLesson)')
            check_related_links(transcript_links, expected)
            evidence['chapter_and_transcript_links'] = expected
            if args.sentence_cues:
                from verify_release_candidate import check_sentence_cues

                evidence['sentence_cue_checks'] = check_sentence_cues(page)
            page.set_viewport_size({'width': 390, 'height': 844})
            # The mobile sidebar slides offscreen on resize. Measure its final
            # position, rather than accepting a frame halfway through that slide.
            page.wait_for_function('''() => {
                const sidebar = document.querySelector('#sidebar');
                return sidebar.inert && sidebar.getBoundingClientRect().right <= 1;
            }''')
            page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
            mobile_geometry = page.evaluate('''() => ({
                viewport: window.innerWidth, width: document.documentElement.scrollWidth,
                overflowing: [...document.querySelectorAll('body *')].filter(node => {
                    const r = node.getBoundingClientRect();
                    return r.width > 0 && (r.right > window.innerWidth + 1 || r.left < -1);
                }).slice(0, 30).map(node => ({tag: node.tagName, id: node.id,
                    className: String(node.className), text: node.textContent.slice(0, 160),
                    width: node.getBoundingClientRect().width}))
            })''')
            if mobile_geometry['width'] > mobile_geometry['viewport']:
                write(output / 'mobile-overflow.json', mobile_geometry)
                page.screenshot(path=str(output / 'mobile-overflow.png'), full_page=True)
                raise AssertionError(f'Mobile overflow: {mobile_geometry}')
            page.screenshot(path=str(output / 'mobile.png'), full_page=True)
            assert not errors, errors
            evidence['passed'] = True
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    if retained:
        after = retained_media_sources(
            DEFAULT_STAGE, WORKSPACE, REPO / 'docs/source/_extra/tutorials/catalog',
            args.lesson, voice_matrix(WORKSPACE / 'tools/render_all_voices.py'))
        if after != retained:
            raise ValueError('Retained source media changed during playback verification')
        evidence['whole_media_retained'] = True
        evidence['original_sources_unchanged'] = True
    write(output / 'playback-checks.json', evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
