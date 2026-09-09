from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "web" / "app_v2.js"
INDEX = ROOT / "web" / "index.html"
VOICE_CATALOG = ROOT / "web" / "voice_catalog.js"
PUBLISHER = ROOT / "tools" / "publish_tutorials.py"


def test_publisher_preserves_the_canonical_thinned_spacr_logo():
    source = PUBLISHER.read_text(encoding="utf-8")

    assert 'if filename == "logo_spacr.png"' in source
    assert 'REPO / "spacr" / "resources" / "icons" / "logo_spacr.png"' in source
    assert hashlib.sha256((ROOT / "web" / "logo_spacr.png").read_bytes()).hexdigest() == (
        "269fe8f2d7040ad1241c6f5110e5f1fa5cd1426141837edb7ad1d3ffde390761"
    )


def extract_simple_function(source: str, name: str) -> str:
    start = source.index(f"function {name}(")
    end = source.index("\n}\n", start) + len("\n}\n")
    return source[start:end]


def test_video_pause_decision_keeps_narration_running_only_at_visual_eof():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the browser-player contract test")
    source = APP.read_text(encoding="utf-8")
    decision = extract_simple_function(
        source, "shouldPauseNarrationForVideoPause"
    )
    assertions = """
const decide = shouldPauseNarrationForVideoPause;
if (decide(true, true) !== false) process.exit(1);
if (decide(true, false) !== true) process.exit(2);
if (decide(false, true) !== true) process.exit(3);
"""
    subprocess.run([node, "-e", decision + assertions], check=True)


def test_video_play_decision_distinguishes_internal_park_resume_and_replay():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the browser-player contract test")
    source = APP.read_text(encoding="utf-8")
    decision = extract_simple_function(source, "narratedVideoPlayAction")
    assertions = """
const decide = narratedVideoPlayAction;
if (decide(true, false, false) !== "programmatic") process.exit(1);
if (decide(false, true, true) !== "replay") process.exit(2);
if (decide(false, false, true) !== "resume") process.exit(3);
if (decide(false, false, false) !== "sync") process.exit(4);
"""
    subprocess.run([node, "-e", decision + assertions], check=True)


def test_audio_ended_is_authoritative_for_narrated_playback():
    source = APP.read_text(encoding="utf-8")
    assert 'elements.audio.addEventListener("ended", finishNarratedPlayback);' in source
    assert re.search(
        r'elements\.video\.addEventListener\("ended", \(\) => \{\s*'
        r'if \(!parkVideoForNarration\(true\)\) markCompleteAtEnd\(\);',
        source,
    )
    assert "if (!parkVideoForNarration(false)) syncVideoToNarration(false);" in source

    park = extract_simple_function(source, "parkVideoForNarration")
    assert "VIDEO_END_PARK_RATE" in park
    assert "VIDEO_END_PARK_SECONDS" in park
    assert "playVideoWithoutNarrationSync();" in park
    assert "elements.video.pause()" not in park

    finish = extract_simple_function(source, "finishNarratedPlayback")
    assert "setVideoTimeWithoutNarrationSeek(elements.video.duration);" in finish
    assert "elements.video.pause();" in finish

    replay = extract_simple_function(source, "restartNarratedPlayback")
    assert "elements.audio.currentTime = 0;" in replay
    assert "setVideoTimeWithoutNarrationSeek(0);" in replay
    assert "elements.video.playbackRate = 1;" in replay
    assert "seekNarrationToVideo" not in replay


def test_programmatic_video_seek_and_rate_changes_do_not_seek_narration():
    source = APP.read_text(encoding="utf-8")
    setter = extract_simple_function(source, "setVideoTimeWithoutNarrationSeek")
    assert setter.index("videoClockCorrectionPending = true;") < setter.index(
        "elements.video.currentTime = bounded;"
    )
    correction = extract_simple_function(source, "setVideoTimeFromNarration")
    assert correction.index("videoClockCorrectionPending = true;") < correction.index(
        "elements.video.currentTime = bounded;"
    )
    assert source.count("elements.video.currentTime =") == 2
    assert re.search(
        r'elements\.video\.addEventListener\("seeking", \(\) => \{\s*'
        r'if \(!videoClockCorrectionPending\) \{[^}]*seekNarrationToVideo\(\);',
        source,
    )

    park = extract_simple_function(source, "parkVideoForNarration")
    assert "setVideoTimeWithoutNarrationSeek(" in park
    assert "seekNarrationToVideo" not in park
    assert park.index("programmedVideoRate = VIDEO_END_PARK_RATE;") < park.index(
        "elements.video.playbackRate = VIDEO_END_PARK_RATE;"
    )

    rate_handler = source[
        source.index('elements.video.addEventListener("ratechange"'):
        source.index('elements.video.addEventListener("ended"')
    ]
    assert "seekNarrationToVideo" not in rate_handler


def test_phone_emulated_eof_pause_and_replay_contract(tmp_path):
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        pytest.skip("Chrome is unavailable for the mobile player contract")
    source = APP.read_text(encoding="utf-8")
    pause_decision = extract_simple_function(
        source, "shouldPauseNarrationForVideoPause"
    )
    play_decision = extract_simple_function(source, "narratedVideoPlayAction")
    harness = tmp_path / "mobile_player_contract.html"
    harness_html = (
        '<!doctype html><meta name="viewport" '
        'content="width=device-width,initial-scale=1">'
        '<body data-result="pending"><script>\n'
        + pause_decision
        + play_decision
        + """
const naturalEofKeepsAudio =
  shouldPauseNarrationForVideoPause(true, true) === false;
const nativePauseDuringParkPausesAudio =
  shouldPauseNarrationForVideoPause(true, false) === true;
const completedNativePlayRestarts =
  narratedVideoPlayAction(false, true, false) === "replay";
const parkedNativePlayResumes =
  narratedVideoPlayAction(false, false, true) === "resume";
const mobileViewport = matchMedia("(max-width: 780px)").matches;
const mobileAgent = /iPhone/.test(navigator.userAgent);
document.body.dataset.result = [
  naturalEofKeepsAudio,
  nativePauseDuringParkPausesAudio,
  completedNativePlayRestarts,
  parkedNativePlayResumes,
  mobileViewport,
  mobileAgent
].every(Boolean) ? "pass" : "fail";
</script>"""
    )
    harness.write_text(harness_html, encoding="utf-8")
    result = subprocess.run(
        [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            f"--user-data-dir={tmp_path / 'chrome-profile'}",
            "--window-size=390,844",
            "--user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X)",
            "--dump-dom",
            harness.as_uri(),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert 'data-result="pass"' in result.stdout


def test_caption_cues_prefer_exact_sentence_sidecar_times():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the caption timing contract")
    source = APP.read_text(encoding="utf-8")
    helper = extract_simple_function(source, "captionCueIntervals")
    assertions = r"""
const chapter = { start: 0, end: 10 };
const timing = { sentences: [
  { speech_start: 0, speech_end: 2.5 },
  { speech_start: 3.0, speech_end: 10 }
] };
const exact = captionCueIntervals(chapter, timing, ["Short.", "Longer sentence."]);
if (exact.length !== 2 || exact[0].end !== 2.5 || exact[1].start !== 3.0) process.exit(1);
const fallback = captionCueIntervals(chapter, {}, ["A.", "Much longer sentence."]);
if (!(fallback[0].end < 5 && fallback[1].end === 10)) process.exit(2);
"""
    subprocess.run([node, "-e", helper + assertions], check=True)


def test_caption_splitter_preserves_real_filenames_and_model_labels():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the caption splitter contract")
    source = APP.read_text(encoding="utf-8")
    splitter = extract_simple_function(source, "splitCaptionText")
    assertions = r"""
const linux = "On Linux, make the downloaded .run file executable, then start it. The default uses automatic hardware acceleration.";
const log = "If setup fails, install.log remains beside the installer. Open it before retrying.";
const models = "Configure Model A and Model B. Keep shared thresholds matched.";
if (splitCaptionText(linux).length !== 2) process.exit(1);
if (splitCaptionText(log).length !== 2) process.exit(2);
if (splitCaptionText(models).length !== 2) process.exit(3);
if (!splitCaptionText(linux)[0].includes(".run file")) process.exit(4);
if (!splitCaptionText(log)[0].includes("install.log")) process.exit(5);
"""
    subprocess.run([node, "-e", splitter + assertions], check=True)


def test_render_captions_passes_the_matching_audio_scene_to_cue_builder():
    source = APP.read_text(encoding="utf-8")
    render = extract_simple_function(source, "renderCaptions")
    assert "audioTimings?.scenes?.[chapter.index - 1]" in render
    assert "captionCueIntervals(chapter, timing, sentences)" in render


def test_player_release_cache_key_and_voice_counts_are_current():
    index = INDEX.read_text(encoding="utf-8")
    assert "English · 24 voices" in index
    assert "8 languages · 50 voices" in index
    assert 'voice_catalog.js?v=20260811-50-voices' in index
    assert 'lesson_catalog.js?v=20260827-conda-live' in index
    assert 'app_v2.js?v=20260825-folded-routes' in index
    assert "20260810-mobile-smooth" not in index


def test_folded_lesson_route_uses_the_current_host_module() -> None:
    source = APP.read_text(encoding="utf-8")
    index = INDEX.read_text(encoding="utf-8")

    assert 'id="lesson-route"' in index
    assert "lesson?.host_app_key || lesson?.app_key" in source
    assert "item.app_key === lesson.host_app_key && !item.host_app_key" in source
    assert "elements.content.dataset.appKey = route.appKey" in source
    assert "localizedLesson(route.host.id)" in source


def test_player_catalog_excludes_pause_heavy_retired_voices():
    catalog = VOICE_CATALOG.read_text(encoding="utf-8")
    for voice in ("af_nicole", "af_alloy", "af_kore", "af_nova"):
        assert f'id: "{voice}"' not in catalog
