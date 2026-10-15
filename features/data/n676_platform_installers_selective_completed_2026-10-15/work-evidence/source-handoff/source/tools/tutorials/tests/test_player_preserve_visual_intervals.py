"""Retained visual intervals stay intact without asserting full raw-clip coverage."""
import json

from test_player_complete_native_clips import complete_fixture, open_player
from test_player_holds_narration import served
from verify_staged_lesson import check_native_boundary_playback


def test_preserved_intervals_hold_audio_and_finish_selected_video_at_one_speed(served, tmp_path, monkeypatch):
    from playwright.sync_api import sync_playwright
    import test_player_holds_narration
    monkeypatch.setattr(test_player_holds_narration, 'VIDEO_DELAY_SECONDS', 0)
    url = complete_fixture(served, tmp_path)
    visual_path = tmp_path / 'complete-native-visual.json'
    visual = json.loads(visual_path.read_text())
    del visual['complete_native_clips']
    visual['preserve_visual_intervals'] = True
    visual['scenes'][0]['native_live_video'] = False
    visual['scenes'][-1]['native_live_video'] = False
    visual_path.write_text(json.dumps(visual))
    spoken = json.loads((tmp_path / 'complete-native-audio.json').read_text())
    with sync_playwright() as engine:
        browser, page = open_player(engine, url)
        try:
            page.wait_for_function('elements.video.buffered.length && elements.video.buffered.end(elements.video.buffered.length-1)>=11.99', timeout=30000)
            assert page.evaluate('visualTimings.complete_native_clips === undefined')
            result = check_native_boundary_playback(page, visual, spoken)
            assert result['preserve_visual_intervals'] is True
            assert result['complete_native_clips'] is False
            assert result['actual_video_speed'] == 1
            assert result['boundary']['recorded_tail_observed']
            assert result['boundary']['next_narration_resumed']
            assert result['boundary']['checked_boundary_after_scene'] == 2
            assert result['final_tail']['final_visual_eof_observed']
            assert result['final_tail']['final_native_eof_observed'] is False
            assert page.evaluate('elements.video.currentTime') > 11.9
            assert page.evaluate('elements.video.playbackRate === 1 && elements.audio.playbackRate === 1')
            page.evaluate('visualTimings.native_live_video=false')
            assert page.evaluate('completeNativePlayback()') is False
        finally:
            browser.close()


def test_absent_preservation_flag_keeps_legacy_narration_owned_boundary_skip(served, tmp_path):
    from test_player_native_live_video import test_native_video_runs_at_one_speed_and_seeks_only_scene_boundaries
    test_native_video_runs_at_one_speed_and_seeks_only_scene_boundaries(served, tmp_path)
