"""A long glide stays under the per-frame pointer limit (lesson 08, 2026-10-17)."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import human_input  # noqa: E402


class _Host:
    width, height = 2560, 1440


def _human():
    human = human_input.Human.__new__(human_input.Human)
    human.host = _Host()
    human.rng = random.Random(0)
    return human


def _peak_px_per_frame(human, distance, duration, steps=2000):
    a = human.cruise_share(distance, duration)
    path = [((1 - a) * human_input.min_jerk(i / steps) + a * i / steps) * distance
            for i in range(steps + 1)]
    per_second = max(b - c for b, c in zip(path[1:], path)) * steps / duration
    return per_second / 30.0


def test_a_short_move_keeps_minimum_jerk():
    human = _human()
    assert human.cruise_share(400, human.duration_for(400)) == 0.0


def test_a_screen_long_glide_stays_under_the_limit():
    human = _human()
    limit = human.peak_per_frame * _Host.width
    for distance in (1200, 2080, 2900):
        duration = human.duration_for(distance) / 1.08
        assert _peak_px_per_frame(human, distance, duration) <= limit * 1.001, distance
