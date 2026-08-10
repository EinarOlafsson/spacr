"""The "remove border objects" animations must remove border objects.

The maintainer reported that these removed one edge object AND one object
that was not on the edge. Two separate faults produced that, and NEITHER is
visible to a pixel diff that trusts the scene's own ``touches`` flag as
ground truth -- such a diff validates the generator against itself and
passes:

  * one object was flagged as touching the well edge while sitting wholly
    inside the well, so the animation removed an interior object;
  * a KEPT object lay outside the close camera, so it slid out of frame
    during the zoom in a way no viewer can distinguish from removal.

Both are geometry, so these tests check geometry -- against the drawn well
rectangle, not against the flag.
"""

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

gen = pytest.importorskip("generate_setting_animations")

EDGE_LOW = 12.0
EDGE_HIGH = float(gen.CANVAS) - 12.0
BORDER_SLUGS = (
    "cell_remove_border_objects",
    "nucleus_remove_border_objects",
    "organelle_remove_border_objects",
    "pathogen_remove_border_objects",
)


def _straddles(center, size):
    lo_x, hi_x = center[0] - size[0] / 2.0, center[0] + size[0] / 2.0
    lo_y, hi_y = center[1] - size[1] / 2.0, center[1] + size[1] / 2.0
    return (
        lo_x < EDGE_LOW < hi_x or lo_x < EDGE_HIGH < hi_x
        or lo_y < EDGE_LOW < hi_y or lo_y < EDGE_HIGH < hi_y
    )


@pytest.fixture(scope="module")
def specs():
    return {spec.slug: spec for spec in gen._specs()}


class TestTheGateRefusesABadLayout:
    """The check has to fire, or it is decoration."""

    def test_it_rejects_the_layout_that_shipped_the_bug(self):
        offset = gen.Y_OFFSET
        shipped = [
            ((278, 18 + offset), (30, 23), True),
            ((346, 80 + offset), (27, 22), True),
            ((274, 91 + offset), (24, 19), False),
            ((315, 130 + offset), (18, 15), False),
        ]
        with pytest.raises(ValueError, match="does not cross"):
            gen._assert_border_layout(shipped, (205.0, 0.0, 360.0, 155.0))

    def test_it_rejects_a_kept_object_that_leaves_the_frame(self):
        offset = gen.Y_OFFSET
        objects = [
            ((348, 40 + offset), (30, 23), True),
            ((350, 130 + offset), (27, 22), True),
            ((255, 45 + offset), (24, 19), False),
            ((262, 300 + offset), (18, 15), False),   # below the camera
        ]
        with pytest.raises(ValueError, match="leave the frame"):
            gen._assert_border_layout(objects, (228.0, 70.5, 378.0, 220.5))

    def test_it_rejects_a_border_object_that_is_kept(self):
        offset = gen.Y_OFFSET
        objects = [
            ((348, 40 + offset), (30, 23), True),
            ((350, 130 + offset), (27, 22), True),
            ((255, 45 + offset), (24, 19), False),
            ((348, 100 + offset), (18, 15), False),   # on the edge, kept
        ]
        with pytest.raises(ValueError, match="border object surviving"):
            gen._assert_border_layout(objects, (228.0, 70.5, 378.0, 220.5))


class TestTheShippedScene:

    @pytest.mark.parametrize("slug", BORDER_SLUGS)
    def test_every_frame_renders(self, specs, slug):
        """Rendering runs the gate, so this asserts the layout is coherent."""
        for index in range(gen.FRAMES):
            assert gen.render_frame(specs[slug], index) is not None

    @pytest.mark.parametrize("slug", BORDER_SLUGS)
    def test_removed_objects_cross_the_edge_and_kept_ones_do_not(
        self, specs, slug,
    ):
        import numpy as np

        spec = specs[slug]
        before = np.asarray(
            gen.render_frame(spec, 13).convert("RGB"), dtype=np.int16)
        after = np.asarray(
            gen.render_frame(spec, 21).convert("RGB"), dtype=np.int16)
        changed = np.abs(after - before).sum(axis=2) > 30

        kind = spec.params["kind"]
        sizes = (
            [(30, 23), (27, 22), (24, 19), (18, 15)] if kind == "cell"
            else [(16, 13), (15, 12), (14, 11), (11, 9)]
        )
        centers = [(348, 40), (350, 130), (255, 45), (262, 135)]
        left, top = 228.0, 70.5
        scale = gen.CANVAS / 150.0

        for (cx, cy), size in zip(centers, sizes):
            y = cy + gen.Y_OFFSET
            box = (
                int(max(0, (cx - size[0] / 2 - left) * scale)),
                int(max(0, (y - size[1] / 2 - top) * scale)),
                int(min(gen.CANVAS, (cx + size[0] / 2 - left) * scale)),
                int(min(gen.CANVAS, (y + size[1] / 2 - top) * scale)),
            )
            region = changed[box[1]:box[3], box[0]:box[2]]
            assert region.size, f"object at {(cx, cy)} is outside the frame"
            removed = bool(region.mean() > 0.05)
            assert removed is _straddles((cx, y), size), (
                f"{slug}: object at {(cx, y)} "
                f"{'was removed' if removed else 'survived'} but it "
                f"{'crosses' if _straddles((cx, y), size) else 'does not cross'}"
                " the well edge"
            )

    @pytest.mark.parametrize("slug", BORDER_SLUGS)
    def test_two_objects_are_removed_and_two_survive(self, specs, slug):
        kind = specs[slug].params["kind"]
        sizes = (
            [(30, 23), (27, 22), (24, 19), (18, 15)] if kind == "cell"
            else [(16, 13), (15, 12), (14, 11), (11, 9)]
        )
        centers = [(348, 40), (350, 130), (255, 45), (262, 135)]
        crossing = [
            _straddles((cx, cy + gen.Y_OFFSET), size)
            for (cx, cy), size in zip(centers, sizes)
        ]
        assert crossing.count(True) == 2
        assert crossing.count(False) == 2
