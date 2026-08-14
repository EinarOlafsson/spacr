from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


live = _load("verify_tutorial_live", "tools/verify_tutorial_live.py")
frames = _load("sample_tutorial_frames", "tools/sample_tutorial_frames.py")


def test_release_audit_parsers_pin_the_current_inventory():
    tutorial_root = ROOT / "docs" / "source" / "_extra" / "tutorials"
    catalog = frames.load_catalog(tutorial_root / "lesson_catalog.js")
    languages, voices = live._voice_inventory(
        (tutorial_root / "voice_catalog.js").read_text(encoding="utf-8")
    )
    assert len(catalog["lessons"]) == 69
    assert sum(len(lesson["scenes"]) for lesson in catalog["lessons"]) == 487
    assert len(languages) == 8
    assert len(voices) == 50
    assert not (live.RETIRED_VOICES & set(voices))
    assert live.EXPECTED_CACHE_KEY in (
        tutorial_root / "index.html"
    ).read_text(encoding="utf-8")


def test_scene_midpoint_stays_inside_speech():
    scene = {"speech_start": 14.105, "speech_end": 20.26}
    midpoint = frames.scene_midpoint(scene)
    assert scene["speech_start"] < midpoint < scene["speech_end"]
    assert midpoint == pytest.approx(17.1825)


def test_frame_sampler_recreates_a_reviewable_still(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is required for visual audit sampling")
    video = tmp_path / "source.mp4"
    image = tmp_path / "scene.jpg"
    subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=blue:s=320x180:d=1", "-pix_fmt",
         "yuv420p", str(video)],
        check=True,
    )
    frames.extract_frame(video, 0.5, image, 160)
    assert image.is_file()
    assert image.stat().st_size > 0


def test_tutorial_release_skill_has_no_template_placeholders():
    skill_path = (
        ROOT / ".claude" / "skills" / "tutorial-release-audit" / "SKILL.md"
    )
    skill = skill_path.read_text(encoding="utf-8")
    contract = (skill_path.parent / "references" / "release-contract.md").read_text(
        encoding="utf-8"
    )
    assert "TODO" not in skill
    assert "tools/verify_tutorial_live.py" in skill
    assert "tools/sample_tutorial_frames.py" in skill
    assert "verify_audio_release.py" in contract
