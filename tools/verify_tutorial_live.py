#!/usr/bin/env python3
"""Verify the published tutorial inventory and mobile playback contract.

Static checks use only the standard library. Pass ``--browser`` to exercise
the live media clocks in headless Chrome; that mode requires Selenium.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "docs" / "source" / "_extra" / "tutorials"
DEFAULT_URL = "https://einarolafsson.github.io/spacr/tutorials/"
EXPECTED_CACHE_KEY = "20260815-73-lessons"
EXPECTED_VOICE_KEY = "20260811-50-voices"
EXPECTED_APP_KEY = "20260811-audio-end-park-captions"
RETIRED_VOICES = {"af_alloy", "af_kore", "af_nicole", "af_nova"}


def _fetch(url: str, timeout: int) -> bytes:
    separator = "&" if "?" in url else "?"
    request = urllib.request.Request(
        f"{url}{separator}audit={time.time_ns()}",
        headers={"User-Agent": "spaCR tutorial release audit"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{url}: HTTP {response.status}")
        return response.read()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _catalog_payload(source: str) -> dict:
    prefix = "window.SPACR_LESSON_CATALOG = Object.freeze("
    start = source.index(prefix) + len(prefix)
    end = source.rindex(");")
    return json.loads(source[start:end])


def _voice_inventory(source: str) -> tuple[list[str], list[str]]:
    languages = re.findall(r'^\s{4}id: "([^"]+)",$', source, re.MULTILINE)
    voices = re.findall(r'^\s{6}\{ id: "([^"]+)"', source, re.MULTILINE)
    return languages, voices


def static_audit(url: str, *, timeout: int, compare_local: bool = True) -> dict:
    base = url.rstrip("/") + "/"
    names = ("index.html", "app_v2.js", "voice_catalog.js", "lesson_catalog.js")
    remote = {name: _fetch(urllib.parse.urljoin(base, name), timeout) for name in names}
    index = remote["index.html"].decode("utf-8")
    voices_source = remote["voice_catalog.js"].decode("utf-8")
    catalog = _catalog_payload(remote["lesson_catalog.js"].decode("utf-8"))
    languages, voices = _voice_inventory(voices_source)
    lessons = catalog["lessons"]
    scene_count = sum(len(lesson["scenes"]) for lesson in lessons)
    def versioned(script: str) -> str:
        match = re.search(rf'{re.escape(script)}\?v=([^"\s]+)', index)
        if not match:
            raise AssertionError(f"index.html has no versioned {script} reference")
        return match.group(1)

    result = {
        "url": base,
        "lessons": len(lessons),
        "scenes": scene_count,
        "languages": len(languages),
        "voices": len(voices),
        "cache_key": versioned("lesson_catalog.js"),
        "voice_cache_key": versioned("voice_catalog.js"),
        "app_cache_key": versioned("app_v2.js"),
        "retired_voices_present": sorted(RETIRED_VOICES & set(voices)),
        "sha256": {name: _sha256(payload) for name, payload in remote.items()},
    }
    if compare_local:
        local_hashes = {name: _sha256((LOCAL / name).read_bytes()) for name in names}
        result["local_sha256"] = local_hashes
        result["hashes_match_local"] = result["sha256"] == local_hashes
    assert result["lessons"] == 73, result
    assert result["scenes"] == 507, result
    assert result["languages"] == 8, result
    assert result["voices"] == 50, result
    assert result["cache_key"] == EXPECTED_CACHE_KEY, result
    assert result["voice_cache_key"] == EXPECTED_VOICE_KEY, result
    assert result["app_cache_key"] == EXPECTED_APP_KEY, result
    assert not result["retired_voices_present"], result
    if compare_local:
        assert result["hashes_match_local"], result
    return result


def mobile_browser_audit(url: str, *, timeout: int, screenshot: Path) -> dict:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("--browser requires `python -m pip install selenium`") from exc

    options = Options()
    for flag in (
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--autoplay-policy=no-user-gesture-required",
        "--window-size=390,844",
        "--user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
    ):
        options.add_argument(flag)

    driver = webdriver.Chrome(options=options)
    wait = WebDriverWait(driver, timeout)
    try:
        driver.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {
            "width": 390,
            "height": 844,
            "deviceScaleFactor": 1,
            "mobile": True,
        })
        separator = "&" if "?" in url else "?"
        driver.get(f"{url}{separator}audit={time.time_ns()}")
        wait.until(lambda d: d.execute_script(
            "return Array.isArray(window.SPACR_VOICE_CATALOG) && "
            "window.SPACR_VOICE_CATALOG.reduce((n,g)=>n+g.voices.length,0)===50"
        ))
        wait.until(lambda d: d.execute_script(
            "return document.querySelector('#tutorial-video').readyState>=1"
        ))
        wait.until(lambda d: d.execute_script(
            "return document.querySelector('#narration-audio').src.startsWith('blob:')"
        ))
        wait.until(lambda d: d.execute_script(
            "return document.querySelector('#caption-track').src.startsWith('blob:')"
        ))
        initial = driver.execute_script("""
          const v=document.querySelector('#tutorial-video');
          const a=document.querySelector('#narration-audio');
          return {lessonPosition:document.querySelector('#lesson-position').textContent.trim(),
            voices:SPACR_VOICE_CATALOG.reduce((n,g)=>n+g.voices.length,0),
            languages:document.querySelectorAll('#language-select option').length,
            captionsEnabled:document.querySelector('#caption-enabled').checked,
            captionSrc:document.querySelector('#caption-track').src,
            audioSrc:a.src,videoSrc:v.currentSrc||v.src,
            viewport:[innerWidth,innerHeight],mobileAgent:/iPhone/.test(navigator.userAgent)};
        """)
        driver.execute_script(
            "void document.querySelector('#tutorial-video').play().catch(()=>{})"
        )
        wait.until(lambda d: d.execute_script(
            "const a=document.querySelector('#narration-audio');"
            "return !a.paused&&a.currentTime>0.05"
        ))
        playing = driver.execute_script("""
          const v=document.querySelector('#tutorial-video'),a=document.querySelector('#narration-audio');
          return {videoPaused:v.paused,audioPaused:a.paused,audioTime:a.currentTime,
            videoTime:v.currentTime,toast:document.querySelector('#toast').textContent};
        """)
        driver.execute_script("document.querySelector('#tutorial-video').pause()")
        wait.until(lambda d: d.execute_script(
            "return document.querySelector('#narration-audio').paused"
        ))
        paused = driver.execute_script("""
          const v=document.querySelector('#tutorial-video'),a=document.querySelector('#narration-audio');
          return {videoPaused:v.paused,audioPaused:a.paused,audioTime:a.currentTime};
        """)
        wait.until(lambda d: d.execute_script("""
          const v=document.querySelector('#tutorial-video'),a=document.querySelector('#narration-audio');
          if (a.paused && !a.ended) void v.play().catch(()=>{});
          return !a.paused && a.currentTime>arguments[0]+0.05;
        """, paused["audioTime"]))
        wait.until(lambda d: d.execute_script(
            "return document.querySelector('#narration-audio').ended"
        ))
        ended = driver.execute_script("""
          const v=document.querySelector('#tutorial-video'),a=document.querySelector('#narration-audio');
          return {videoPaused:v.paused,audioEnded:a.ended,audioTime:a.currentTime,
            audioDuration:a.duration};
        """)
        driver.execute_script(
            "void document.querySelector('#tutorial-video').play().catch(()=>{})"
        )
        wait.until(lambda d: d.execute_script(
            "const a=document.querySelector('#narration-audio');return !a.paused&&a.currentTime<3"
        ))
        replay = driver.execute_script("""
          const v=document.querySelector('#tutorial-video'),a=document.querySelector('#narration-audio');
          return {videoPaused:v.paused,audioPaused:a.paused,audioTime:a.currentTime,
            toast:document.querySelector('#toast').textContent};
        """)
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        driver.save_screenshot(str(screenshot))
    finally:
        driver.quit()

    result = {"initial": initial, "playing": playing, "paused": paused,
              "ended": ended, "replay": replay, "screenshot": str(screenshot)}
    assert initial["lessonPosition"] == "Lesson 1 of 73", result
    assert initial["voices"] == 50 and initial["languages"] == 8, result
    assert initial["captionsEnabled"] and initial["captionSrc"].startswith("blob:"), result
    assert initial["audioSrc"].startswith("blob:"), result
    assert initial["mobileAgent"] and initial["viewport"][0] <= 390, result
    assert not playing["audioPaused"] and playing["audioTime"] > 0, result
    assert paused["audioPaused"] and paused["videoPaused"], result
    assert ended["audioEnded"], result
    assert not replay["audioPaused"] and replay["audioTime"] < 3, result
    assert "Narration is unavailable" not in playing["toast"], result
    assert "Narration is unavailable" not in replay["toast"], result
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--no-local-compare", action="store_true")
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--screenshot", type=Path,
                        default=Path("/tmp/spacr-tutorial-live-mobile.png"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = {"static": static_audit(
        args.url, timeout=args.timeout, compare_local=not args.no_local_compare
    )}
    if args.browser:
        report["mobile"] = mobile_browser_audit(
            args.url, timeout=args.timeout, screenshot=args.screenshot
        )
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
