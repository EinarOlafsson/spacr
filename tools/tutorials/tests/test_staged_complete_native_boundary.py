"""Delivery verifier rejects skipped native tails and checks the real browser clocks."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools'), str(ROOT / 'tools/tutorials')]
from verify_staged_lesson import check_complete_native_trace, check_native_boundary_playback
from test_player_complete_native_clips import complete_fixture, open_player
from test_player_holds_narration import served


def timings():
    visual = {'complete_native_clips': True, 'scenes': [dict(
        speech_start=i*4, speech_end=(i+1)*4, scene_end=(i+1)*4,
        duration=4, native_live_video=True) for i in range(3)]}
    spoken = {'scenes': [dict(speech_start=i, speech_end=i+.65,
                             scene_end=i+1, duration=1) for i in range(3)]}
    return visual, spoken


def trace(index=0, final=False):
    rows = []
    for tick in range(107 if not final else 81):
        elapsed = tick * .05
        v = index*4 + min(elapsed, 4) if final else elapsed
        scene = min(int(v // 4), 2)
        spoken = scene + min(v-scene*4, .65)
        rows.append(dict(wall=elapsed*1000, video=v, audio=spoken, rate=1,
            audioRate=1, audioPaused=v-scene*4 >= .65,
            videoPaused=final and elapsed>=4, videoEnded=final and elapsed>=4,
            seeking=False, correcting=False, error=None))
    return rows


def test_recorded_tail_and_next_narration_clocks_pass():
    visual, spoken = timings()
    assert check_complete_native_trace(trace(), [], visual, spoken, 0)['next_narration_resumed']
    assert check_complete_native_trace(trace(2, True), [], visual, spoken, 2, True)['final_native_eof_observed']


@pytest.mark.parametrize('fault', ['skip', 'seek', 'early_pause', 'rate', 'audio_runs', 'wrong_resume', 'slow', 'no_eof'])
def test_actual_clock_failures_are_rejected(fault):
    visual, spoken = timings()
    final = fault == 'no_eof'
    rows = trace(2, True) if final else trace()
    events = []
    if fault == 'skip':
        rows = [row for row in rows if not 1 < row['video'] < 4]
    elif fault == 'seek':
        events = [{'type': 'seeking', 'video': 4}]
    elif fault == 'early_pause':
        rows[30]['videoPaused'] = True
    elif fault == 'rate':
        rows[30]['rate'] = 2
    elif fault == 'audio_runs':
        rows[30]['audioPaused'] = False
    elif fault == 'slow':
        for row in rows:
            row['wall'] *= 2
    elif fault == 'wrong_resume':
        for row in rows:
            if row['video'] > 4:
                row['audio'] += 1
    else:
        rows = rows[:-1]
    with pytest.raises(AssertionError):
        check_complete_native_trace(rows, events, visual, spoken, 2 if final else 0, final)


@pytest.mark.parametrize('old_skip', [False, True])
def test_delivery_verifier_observes_complete_native_playback(served, tmp_path, old_skip):
    from playwright.sync_api import sync_playwright
    visual, spoken = timings()
    url = complete_fixture(served, tmp_path)
    with sync_playwright() as engine:
        browser, page = open_player(engine, url)
        try:
            if old_skip:
                page.evaluate('''() => {
                    const actual = syncCompleteNativeNarration;
                    syncCompleteNativeNarration = function(seek=false) {
                        actual(seek);
                        if (!seek && elements.video.currentTime > .7 && elements.video.currentTime < 4)
                            elements.video.currentTime = 4;
                    };
                }''')
                with pytest.raises(AssertionError):
                    check_native_boundary_playback(page, visual, spoken)
                return
            result = check_native_boundary_playback(page, visual, spoken)
            assert result['complete_native_clips']
            assert result['boundary']['recorded_tail_observed']
            assert result['final_tail']['final_native_eof_observed']
        finally:
            browser.close()
