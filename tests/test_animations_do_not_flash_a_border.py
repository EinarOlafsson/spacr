"""No setting animation may flash a bright ring around itself.

THE FAULT, found by comparing the shipped GIFs against what the generator
renders and then confirmed through Qt's ``QMovie``: 26 of the 94 packaged
animations draw a 9-pixel near-white border on every frame after the first.
Frame 0 is clean, the rest are ringed, so in the tooltip the ring snaps on
and off once per loop around a picture it has nothing to do with.

IT IS AN ENCODER FAULT, NOT A DRAWING ONE. ``_write_gif`` saved with
``disposal=2`` -- "restore to background colour before the next frame" --
while ``optimize=True`` shrinks each later frame to the sub-rectangle that
actually changed. Everything outside that rectangle is then background. The
26 affected files are exactly the ones regenerated after 2026-08-09, whose
changes happen to sit away from the edges; the untouched 68 were encoded
with frames that still covered the whole canvas and hid it.

IT ALSO CORRUPTED THE AUDIT. A ring around a 360x360 frame is 12,636 px, or
9.75% of it, and ``measure_visible_change`` counted every one of them. Nine
animations were recorded as clearing the 1% visible-change bar by 10x when
their real content change is between 1.3% and 3.1% -- `eps` measured 10.9%
and is 1.33%. The instruction's "27 -> 0 under threshold" was true; the
margins behind it were not.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from spacr.setting_animations import (
    MAX_BORDER_ARTIFACT, measure_border_artifact, measure_visible_change,
    setting_animations, validate_animations_have_no_border_artifact,
)


def _gif(path, frames, **kwargs):
    images = [Image.fromarray(f.astype(np.uint8), "RGB") for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], loop=0, **kwargs)
    return str(path)


def _ringed(size=60, band=6, inner=255):
    """Frame 0 black; frame 1 identical inside, background outside."""
    first = np.zeros((size, size, 3))
    second = np.full((size, size, 3), 237.0)
    second[band:-band, band:-band] = 0.0
    second[size // 2, size // 2] = inner
    return [first, second]


class TestTheMeasurement:

    def test_a_clean_animation_scores_zero(self, tmp_path):
        a = np.zeros((60, 60, 3))
        b = a.copy()
        b[20:40, 20:40] = 255                   # a change well inside
        assert measure_border_artifact(_gif(tmp_path / "clean.gif", [a, b])) == 0.0

    def test_a_ringed_animation_scores_one(self, tmp_path):
        assert measure_border_artifact(
            _gif(tmp_path / "ring.gif", _ringed())) == pytest.approx(1.0)

    def test_a_scene_touching_one_edge_is_not_a_ring(self, tmp_path):
        """The border-object family draws at the edge on purpose.

        A check that fired on any edge change would flag the four animations
        whose whole subject is objects on the border, so it must require the
        WHOLE perimeter -- which no drawn scene produces.
        """
        a = np.zeros((60, 60, 3))
        b = a.copy()
        b[:6, :12] = 255                        # one object at one corner
        got = measure_border_artifact(_gif(tmp_path / "edge.gif", [a, b]))
        assert 0.0 < got < MAX_BORDER_ARTIFACT

    def test_an_unreadable_file_measures_zero_rather_than_raising(self, tmp_path):
        broken = tmp_path / "broken.gif"
        broken.write_bytes(b"GIF89a not really")
        assert measure_border_artifact(broken) == 0.0

    def test_a_single_frame_file_measures_zero(self, tmp_path):
        path = tmp_path / "one.png"
        Image.fromarray(np.zeros((8, 8, 3), np.uint8), "RGB").save(path)
        assert measure_border_artifact(path) == 0.0


class TestTheEncodingThatCausedIt:
    """The bug reproduces from the save arguments alone."""

    def _frames(self):
        frames = []
        for step in range(6):
            frame = np.zeros((60, 60, 3))
            frame[24:36, 20 + step:32 + step] = 200
            frames.append(frame)
        return frames

    def test_disposal_2_with_optimize_leaves_a_ring(self, tmp_path):
        path = _gif(tmp_path / "d2.gif", self._frames(), disposal=2, optimize=True)
        assert measure_border_artifact(path) > MAX_BORDER_ARTIFACT

    def test_disposal_1_does_not(self, tmp_path):
        path = _gif(tmp_path / "d1.gif", self._frames(), disposal=1, optimize=True)
        assert measure_border_artifact(path) == 0.0

    def test_the_ring_inflates_the_visible_change_measurement(self, tmp_path):
        """Which is why this was never noticed: it made the numbers better."""
        frames = self._frames()
        clean = measure_visible_change(
            _gif(tmp_path / "c.gif", frames, disposal=1, optimize=True))
        ringed = measure_visible_change(
            _gif(tmp_path / "r.gif", frames, disposal=2, optimize=True))
        assert ringed > clean * 2


class TestTheGenerator:

    def test_it_saves_with_a_disposal_that_keeps_the_canvas(self):
        """Pinned in the generator, because regenerating one GIF with the
        wrong disposal reintroduces this for that GIF alone."""
        gen = pytest.importorskip("generate_setting_animations")
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(gen._write_gif).lstrip())
        saves = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "save"
        ]
        assert saves, "the generator no longer saves a GIF here"
        disposals = [
            keyword.value.value for save in saves for keyword in save.keywords
            if keyword.arg == "disposal"
        ]
        assert disposals == [1], f"saved with disposal={disposals}"


class TestEveryShippedAnimation:

    def test_none_of_the_94_flash_a_border(self):
        failures = validate_animations_have_no_border_artifact()
        assert failures == {}, (
            "these animations flash a ring around themselves; regenerate them "
            "with tools/generate_setting_animations.py --only SLUG: "
            + ", ".join(sorted(failures))
        )

    def test_the_report_is_a_dict_of_every_offender_not_the_first(self):
        failures = validate_animations_have_no_border_artifact(maximum=-1.0)
        assert len(failures) == len(setting_animations())
