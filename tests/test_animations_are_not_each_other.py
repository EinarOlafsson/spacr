"""Two settings must not be illustrated by the same picture.

`*_perimeter_fraction` and `*_intensity_merge` are different criteria --
the first merges on how much boundary two objects share, the second on
whether there is a real membrane between them -- and both were drawn by one
scene in which a single pair dissolved. Measured over the drawn area of the
GIFs, 98% of the ink was identical; the whole difference was a pulsing line.

That is the failure this audit exists to catch, and it is invisible to every
other check: both files are intact, both change far more than 1% of the
frame, and both illustrate *a* merge. What neither illustrated was its own
criterion, so a user comparing them learns the two settings do the same
thing.

The fix is a second pair that FAILS the criterion and survives, because a
threshold that keeps something is the honest picture of a threshold.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spacr.setting_animations import setting_animations

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

KINDS = ("cell", "nucleus", "pathogen", "organelle")

#: The bug measured 0.9-2.0%. Anything this close is one picture with a
#: decoration on it, not two illustrations.
MIN_DISTINCT_INK = 0.15


def _frames(path):
    frames = []
    with Image.open(path) as image:
        try:
            while True:
                frames.append(np.asarray(image.convert("RGB"), dtype=np.int16))
                image.seek(image.tell() + 1)
        except EOFError:
            pass
    return frames


def _ink_difference(first, second):
    """Differing pixels as a fraction of the area either animation draws in.

    Normalising by the FRAME instead of by the ink is what let this hide: a
    360x360 canvas is mostly black, so two line drawings that share every
    stroke still differ in "only 0.1% of the frame" and look fine.
    """
    a, b = _frames(first), _frames(second)
    assert len(a) == len(b), "different frame counts cannot be compared this way"
    drawn = np.logical_or(
        np.logical_or.reduce([f.sum(2) > 45 for f in a]),
        np.logical_or.reduce([f.sum(2) > 45 for f in b]),
    )
    differs = np.logical_or.reduce(
        [np.abs(x - y).sum(2) > 30 for x, y in zip(a, b)])
    return float((differs & drawn).sum()) / float(drawn.sum())


@pytest.fixture(scope="module")
def paths():
    return {animation.slug: animation.path for animation in setting_animations()}


@pytest.mark.parametrize("kind", KINDS)
def test_perimeter_and_intensity_merge_are_different_pictures(kind, paths):
    got = _ink_difference(
        paths[f"{kind}_perimeter_fraction"], paths[f"{kind}_intensity_merge"])
    assert got > MIN_DISTINCT_INK, (
        f"{kind}_perimeter_fraction and {kind}_intensity_merge differ in only "
        f"{got:.1%} of their drawn area; they illustrate different criteria"
    )


class TestTheSceneDrawsItsCriterion:
    """Both variants must keep a pair that the criterion rejects."""

    def _record(self, kind, intensity, action):
        gen = pytest.importorskip("generate_setting_animations")
        outlines, lines = [], []
        spec = next(
            s for s in gen._specs()
            if s.slug == (f"{kind}_intensity_merge" if intensity
                          else f"{kind}_perimeter_fraction")
        )

        class Recorder:
            def rectangle(self, *a, **k):
                pass

            def line(self, points, color=None, width=0.5, **k):
                lines.append((list(points), color, width))

            def __getattr__(self, name):
                return lambda *a, **k: None

        painter = Recorder()
        real = gen._object_outline

        def spy(_painter, kind_, center, size, amount=1.0, phase=0.0, **k):
            outlines.append((kind_, center, size, amount))

        gen._object_outline = spy
        gen_well = gen._well
        gen._well = lambda *a, **k: None
        try:
            gen._generic_merge(painter, spec, action)
        finally:
            gen._object_outline = real
            gen._well = gen_well
        return outlines, lines

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("intensity", [False, True])
    def test_a_pair_survives_the_merge(self, kind, intensity):
        outlines, _ = self._record(kind, intensity, 1.0)
        solid = [o for o in outlines if o[3] >= 0.99]
        # one merged object plus the pair that failed the criterion
        assert len(solid) == 3, [(o[1], o[3]) for o in outlines]
        survivors = sorted(solid, key=lambda o: -o[1][1])[:2]
        assert survivors[0][1][1] == survivors[1][1][1], "the pair is not level"
        assert survivors[0][1][0] != survivors[1][1][0], "the pair is one object"

    @pytest.mark.parametrize("kind", KINDS)
    def test_only_the_intensity_variant_draws_a_membrane(self, kind):
        _, without = self._record(kind, False, 1.0)
        _, with_line = self._record(kind, True, 1.0)
        assert without == []
        assert len(with_line) == 1
        points, _color, width = with_line[0]
        assert points[0][0] == points[1][0], "the membrane is not vertical"
        assert width > 0.5, "a membrane a viewer cannot see is not evidence"

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_perimeter_variant_separates_its_surviving_pair(self, kind):
        """Its criterion is shared boundary length, so the pair that fails it
        must share less boundary than the pair that passes -- which is
        distance, and is checkable."""
        outlines, _ = self._record(kind, False, 0.0)
        levels = {}
        for _kind, center, _size, amount in outlines:
            if amount >= 0.99:
                levels.setdefault(center[1], []).append(center[0])
        pairs = {y: xs for y, xs in levels.items() if len(xs) == 2}
        assert len(pairs) == 2, f"expected two pairs before the merge: {levels}"
        merging_y = min(pairs)
        surviving_y = max(pairs)
        gap = {y: abs(xs[0] - xs[1]) for y, xs in pairs.items()}
        assert gap[surviving_y] > gap[merging_y] * 1.5, (
            "the pair that survives must overlap visibly less than the pair "
            f"that merges: {gap}"
        )
