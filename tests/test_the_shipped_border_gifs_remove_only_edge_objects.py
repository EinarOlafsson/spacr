"""The reported bug, checked against the PIXELS that ship.

The maintainer reported that "remove edge objects" removes one edge object
AND one object that is not on the edge. That was true, and it was fixed in
the generator -- ``tests/test_border_animation_geometry.py`` pins the fix
there, in the spec, before anything is drawn.

Nothing checked the GIF. A spec can be right while the asset beside it is
the one rendered before the fix, from a different build, or by hand; the
manifest validator passes on any file of the right size and digest, and
``validate_animations_show_something`` passes because 13-21% of the frame
changes. Both said the broken animations were fine for nine days.

So this file opens the four shipped GIFs and asks what they actually remove.
The three ways of measuring that got this wrong before are each pinned as a
test, because they are the mistakes the next person makes:

  * comparing frame 0 with the most-different frame measures the ZOOM;
  * classifying objects by the generator's ``touches`` flag validates the
    generator against itself;
  * counting raw connected components splits one object into many, because
    the well's bright rule is drawn over it.
"""

import numpy as np
import pytest
from PIL import Image

from spacr.setting_animations import (
    SettingAnimationError,
    measure_border_object_removal,
    measure_visible_change,
    setting_animations,
    validate_border_animations_remove_only_edge_objects,
)

SIZE = 120
EDGE = 80          # the drawn well edge
STRADDLING = (68, 94)   # spans EDGE
INTERIOR = (18, 44)     # wholly left of EDGE
ROWS = (40, 70)


def _frame(objects):
    """A frame with a full-height well rule and the named object columns.

    The rule is drawn LAST, over the objects, exactly as the real scene
    draws it -- which is what splits one straddling object into a left and
    a right fragment in any diff taken across it.
    """
    frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for low, high in objects:
        frame[ROWS[0]:ROWS[1], low:high, :] = 120
    frame[:, EDGE:EDGE + 2, :] = 200
    return frame


def _gif(path, frames):
    images = [Image.fromarray(f, "RGB") for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], loop=0)
    return str(path)


@pytest.fixture
def only_the_edge_object(tmp_path):
    """Correct: the object crossing the edge goes, the interior one stays."""
    return _gif(tmp_path / "good.gif", [
        _frame([STRADDLING, INTERIOR]),
        _frame([INTERIOR]),
    ])


@pytest.fixture
def the_reported_bug(tmp_path):
    """The bug as reported: an edge object AND an interior object vanish."""
    return _gif(tmp_path / "bad.gif", [
        _frame([STRADDLING, INTERIOR]),
        _frame([]),
    ])


class TestTheMeasurementCatchesTheReportedBug:

    def test_removing_only_the_edge_object_reports_no_interior_removal(
        self, only_the_edge_object
    ):
        got = measure_border_object_removal(only_the_edge_object)
        assert got["crossing"] == 1
        assert got["interior"] == 0

    def test_removing_an_interior_object_too_is_reported(self, the_reported_bug):
        """The whole point of the file: this is the maintainer's report."""
        got = measure_border_object_removal(the_reported_bug)
        assert got["interior"] == 1, "the interior removal was not seen"
        assert got["crossing"] == 1

    def test_the_edge_column_is_read_off_the_drawn_rule(
        self, only_the_edge_object
    ):
        """Not off a flag the generator also wrote. A check that trusts the
        generator's own ``touches`` boolean passed on the broken assets."""
        assert measure_border_object_removal(only_the_edge_object)["edge_x"] == EDGE

    def test_one_object_split_by_the_rule_counts_once(self, only_the_edge_object):
        """The rule is drawn over the object, so the raw changed mask has a
        left fragment and a right fragment. Counted separately, the right
        fragment lies wholly past the edge and is reported as an INTERIOR
        removal -- a false alarm on a correct animation."""
        assert measure_border_object_removal(only_the_edge_object)["crossing"] == 1

    def test_a_minimum_change_validator_cannot_see_this(self, the_reported_bug):
        """Why this file exists beside the visible-change check: the broken
        animation changes a great deal of the frame. Size, digest and
        percent-changed all pass while the picture says the wrong thing."""
        assert measure_visible_change(the_reported_bug) > 0.02
        assert measure_border_object_removal(the_reported_bug)["interior"] == 1


class TestItRefusesToGuess:

    def test_an_unreadable_file_raises_rather_than_reporting_zero(self, tmp_path):
        """``measure_visible_change`` returns 0.0 for an unreadable file so a
        report can list it beside the others. Here 0.0 would read as "no
        interior object removed", which is a PASS -- so this raises."""
        broken = tmp_path / "broken.gif"
        broken.write_bytes(b"not a gif")
        with pytest.raises(SettingAnimationError):
            measure_border_object_removal(str(broken))

    def test_a_frame_with_no_well_edge_raises(self, tmp_path):
        """This is what the BROKEN assets do, and it is why the check raises
        instead of returning zero. Their close camera was
        ``(205, 0, 360, 155)``, which does not contain the well edge, so no
        frame draws a full-height rule. Run against the four GIFs as they
        shipped before ``ec12448a``, every one of them lands here -- the
        family test above fails loudly rather than reporting "0 interior
        objects removed", which is what a check that guessed would say."""
        blank = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        lit = blank.copy()
        lit[10:20, 10:20, :] = 90
        path = _gif(tmp_path / "nowell.gif", [blank, lit])
        with pytest.raises(SettingAnimationError, match="well edge"):
            measure_border_object_removal(path)

    def test_a_zoom_that_never_rests_raises(self, tmp_path):
        """Two frames at different zooms are not comparable: the diff between
        them is the camera. Better to say so than to report the camera."""
        frames = []
        for column in (60, 70):
            frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            frame[ROWS[0]:ROWS[1], 18:44, :] = 120
            frame[:, column:column + 2, :] = 200
            frames.append(frame)
        path = _gif(tmp_path / "zooming.gif", frames)
        with pytest.raises(SettingAnimationError, match="never rests"):
            measure_border_object_removal(path)


class TestTheShippedAssets:

    @pytest.fixture(scope="class")
    def border_animations(self):
        found = [a for a in setting_animations() if a.scene == "border"]
        assert len(found) == 4, "the border family is four animations"
        return found

    def test_each_removes_only_objects_crossing_the_well_edge(
        self, border_animations
    ):
        for animation in border_animations:
            got = measure_border_object_removal(animation.path)
            assert got["interior"] == 0, (
                f"{animation.slug} removes {got['interior']} object(s) that do "
                "not cross the well edge -- the reported bug is back")

    def test_each_removes_the_two_objects_that_do_cross_it(
        self, border_animations
    ):
        """Not one, and not three. Before the fix a single object crossed the
        edge and two more left the frame during the zoom, which a viewer
        cannot tell from removal."""
        for animation in border_animations:
            assert measure_border_object_removal(animation.path)["crossing"] == 2

    def test_the_pair_compared_is_not_the_zoomed_out_first_frame(
        self, border_animations
    ):
        """Frame 0 is fully zoomed OUT. Comparing against it is where
        "interior: 3 / 4 / 18 / 6" came from -- that was the camera."""
        for animation in border_animations:
            got = measure_border_object_removal(animation.path)
            assert got["before"] > 0
            assert got["after"] > got["before"]

    def test_the_family_validator_is_clean(self):
        assert validate_border_animations_remove_only_edge_objects() == {}
