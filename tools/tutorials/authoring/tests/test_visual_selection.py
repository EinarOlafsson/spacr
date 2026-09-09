from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tutorial_build_visual_specs",
    ROOT / "tools" / "build_visual_specs.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


LESSONS = [
    {"id": "10_classify_cv", "number": 10},
    {"id": "11_classify_ml", "number": 11},
    {"id": "17_timelapse", "number": 17},
]


def test_visual_selection_accepts_numbers_ranges_and_exact_ids():
    assert module.selected_lesson_ids(
        "10-11,17_timelapse", LESSONS
    ) == {"10_classify_cv", "11_classify_ml", "17_timelapse"}


@pytest.mark.parametrize("value", ["12", "missing", "11-10", ",,"])
def test_visual_selection_rejects_unknown_or_empty_requests(value):
    with pytest.raises(ValueError):
        module.selected_lesson_ids(value, LESSONS)


def test_hosted_visual_keeps_scene_count_and_uses_the_live_host_capture(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(module, "PRODUCTION", tmp_path)
    keyframes = tmp_path / "10_classify_cv" / "keyframes"
    keyframes.mkdir(parents=True)
    (keyframes / "01_module.png").write_bytes(b"current host capture")
    (keyframes / "geometry.json").write_text(json.dumps({
        "app_key": "classify",
        "host_app_key": "classify_merged",
        "frame_size": [3840, 2160],
        "overview": [100, 100, 3000, 1800],
        "input": [200, 200, 900, 700],
        "settings": [200, 200, 900, 1200],
        "run": [200, 1700, 200, 80],
        "output": [1200, 200, 1800, 1200],
    }))
    lesson = {
        "id": "10_classify_cv",
        "app_key": "classify",
        "host_app_key": "classify_merged",
        "scenes": [
            {"visual": "overview", "narration": "Current overview."},
            {"visual": "inputs", "focus_key": "input",
             "narration": "Current inputs."},
            {"visual": "run", "focus_key": "actions",
             "narration": "Current action."},
            {"visual": "result", "focus_key": "console",
             "narration": "Current result."},
        ],
    }
    module.build_hosted_app_lesson(lesson)
    spec = json.loads((tmp_path / "10_classify_cv" / "scenes.json").read_text())
    assert spec["host_app_key"] == "classify_merged"
    assert len(spec["scenes"]) == len(lesson["scenes"])
    assert {scene["image"] for scene in spec["scenes"]} == {
        "keyframes/01_module.png"
    }
    assert spec["scenes"][-1]["focus"] == [1200, 200, 1800, 1200]


def test_hosted_visual_rejects_a_retired_standalone_capture(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(module, "PRODUCTION", tmp_path)
    keyframes = tmp_path / "48_hit_list" / "keyframes"
    keyframes.mkdir(parents=True)
    (keyframes / "01_module.png").write_bytes(b"retired standalone capture")
    (keyframes / "geometry.json").write_text(json.dumps({
        "frame_size": [3840, 2160],
        "frames": {"overview": {"file": "01_overview.png"}},
    }))
    lesson = {
        "id": "48_hit_list",
        "app_key": "hit_list",
        "host_app_key": "regression",
        "scenes": [{"visual": "overview", "narration": "Current overview."}],
    }

    with pytest.raises(RuntimeError, match="standalone multi-frame capture"):
        module.build_hosted_app_lesson(lesson)


def test_hosted_visual_rejects_a_capture_from_the_wrong_host(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(module, "PRODUCTION", tmp_path)
    keyframes = tmp_path / "46_napari_bridge" / "keyframes"
    keyframes.mkdir(parents=True)
    (keyframes / "01_module.png").write_bytes(b"wrong host capture")
    (keyframes / "geometry.json").write_text(json.dumps({
        "app_key": "napari_bridge",
        "host_app_key": "napari_bridge",
        "frame_size": [3840, 2160],
    }))
    lesson = {
        "id": "46_napari_bridge",
        "app_key": "napari_bridge",
        "host_app_key": "make_masks",
        "scenes": [{"visual": "overview", "narration": "Current overview."}],
    }

    with pytest.raises(RuntimeError, match="capture route"):
        module.build_hosted_app_lesson(lesson)
