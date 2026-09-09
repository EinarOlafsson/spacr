from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from render_visual_master import frame_aligned_durations  # noqa: E402


def test_scene_boundaries_are_quantized_cumulatively() -> None:
    source = [{"duration": 1.02}, {"duration": 1.02}, {"duration": 1.02}]

    durations = frame_aligned_durations(source, fps=30)

    assert sum(durations) == pytest.approx(round(3.06 * 30) / 30)
    cumulative = 0.0
    rendered_cumulative = 0.0
    for scene, duration in zip(source, durations):
        cumulative += scene["duration"]
        rendered_cumulative += duration
        assert abs(rendered_cumulative - cumulative) <= 1 / 60


@pytest.mark.parametrize(
    ("scenes", "fps"),
    [([{"duration": 1.0}], 0), ([{"duration": 0.0}], 30)],
)
def test_invalid_frame_alignment_inputs_fail(scenes: list[dict], fps: int) -> None:
    with pytest.raises(ValueError):
        frame_aligned_durations(scenes, fps)
