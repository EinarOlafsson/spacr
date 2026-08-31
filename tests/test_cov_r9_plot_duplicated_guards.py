"""Four guards in ``plot.py`` whose other arm the caller already settled.

Each is a second copy of a decision made a few lines earlier, in a
nested helper that could be called with the un-settled value and never
is. They are worth holding for the same reason a duplicated default
always is: the two copies can drift, and the one that runs is not the
one a reader is looking at.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from spacr import plot as P


def _nested(outer, name):
    """The source of one nested ``def`` inside ``outer``."""
    tree = ast.parse(inspect.getsource(outer))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(inspect.getsource(outer), node)
    raise AssertionError(f"{name} is no longer nested inside {outer.__name__}")


class TestTheFileExtensionDefault:

    def test_the_two_default_lists_are_the_same(self):
        """THE PIN, for ``if extensions is None`` inside ``find_files``.

        ``plot_images_and_arrays`` resolves the default before it calls
        the helper, so the helper's own resolution never runs. What can
        go wrong is DRIFT: adding ``.jpg`` to one and not the other
        changes what a direct caller sees and nothing else, which is a
        difference nobody would look for.
        """
        outer = inspect.getsource(P.plot_images_and_arrays)
        defaults = [line for line in outer.splitlines()
                    if "extensions = ['.npy'" in line]

        assert len(defaults) == 2, (
            f"expected the default list in both the outer function and "
            f"find_files; found {len(defaults)}")
        assert defaults[0].strip() == defaults[1].strip(), (
            "the two default extension lists have drifted apart")

    def test_the_outer_resolves_it_before_calling(self):
        outer = inspect.getsource(P.plot_images_and_arrays)
        resolved = outer.index("if extensions is None:")
        call = outer.index("file_dict = find_files(folders, extensions)")

        assert resolved < call, (
            "find_files is now called before the default is resolved, so "
            "its own resolution is live and needs a test of its own")

    def test_the_default_covers_what_the_pipeline_writes(self):
        """The substance: these four are the array and image formats the
        rest of spaCR saves. A missing one is a folder that looks empty."""
        outer = inspect.getsource(P.plot_images_and_arrays)

        for extension in ('.npy', '.tif', '.tiff', '.png'):
            assert f"'{extension}'" in outer


class TestTheSaveFlag:

    def test_the_only_caller_passes_it_false(self):
        """THE PIN, for ``if save:`` inside ``plot_from_file_dict``.

        The parameter exists and is never set: the one call site passes
        ``save=False`` literally, and the public function has no ``save``
        of its own to forward. So the branch cannot run, and the
        parameter is a knob with nothing attached to it.
        """
        outer = inspect.getsource(P.plot_images_and_arrays)

        assert "save=False)" in outer, (
            "plot_from_file_dict is no longer called with save=False, so "
            "the saving branch is live")
        assert "save" not in inspect.signature(
            P.plot_images_and_arrays).parameters, (
            "plot_images_and_arrays grew a save parameter; if it forwards "
            "one, the branch inside plot_from_file_dict is now reachable")

    def test_the_helper_still_declares_it(self):
        """Recorded rather than assumed: the branch is dead because of
        the CALL, not because the code was removed. Deleting the
        parameter is the repair, and this says so where the next reader
        will look."""
        helper = _nested(P.plot_images_and_arrays, "plot_from_file_dict")

        assert "save=False" in helper
        assert "if save:" in helper


class TestTheTransformNameIsCheckedTwice:

    def test_an_unknown_x_transform_is_refused_by_the_first_check(self):
        """THE PIN, for the second ``raise ValueError(f"Unknown
        x_transform")``.

        Two copies of the same validation, and the data transform runs
        first -- so an unknown name is refused there and the threshold
        helper never sees one. Driven rather than read, because what
        matters is that the refusal HAPPENS, not which line it comes
        from.
        """
        import pandas as pd

        frame = pd.DataFrame({"fc": [1.0, 2.0], "p": [0.01, 0.2]})

        with pytest.raises(ValueError) as caught:
            P.volcano_plot(frame, fold_change_col="fc", p_value_col="p",
                           x_transform="sqrt",
                           fold_change_threshold=1.5)

        assert "Unknown x_transform" in str(caught.value)
        assert "sqrt" in str(caught.value)

    def test_both_copies_name_the_same_thing(self):
        """If they ever disagree, the message a user sees depends on
        which path ran, which is the worst kind of error text."""
        source = inspect.getsource(P.volcano_plot)
        copies = [line.strip() for line in source.splitlines()
                  if "Unknown x_transform" in line]

        assert len(copies) == 2
        assert copies[0].replace("mode", "x_transform") == copies[1], (
            f"the two refusals disagree: {copies}")

    def test_the_accepted_names_are_the_same_in_both(self):
        """The half that actually breaks: one copy learning a new
        transform and the other refusing it."""
        source = inspect.getsource(P.volcano_plot)

        for name in ("log2", "log10"):
            assert source.count(f'== "{name}"') >= 2, (
                f"{name} is no longer handled in both places")
        assert source.count('in ("ln", "log")') >= 2


class TestGrayscaleContours:

    def test_a_two_dimensional_image_is_widened_to_rgb(self):
        """THE ARC: ``image.ndim == 2``.

        A single-channel frame is the ordinary case for a mask preview,
        and OpenCV's contour drawing needs three channels -- so the copy
        is what stops the outline being drawn into the data.
        """
        image = np.zeros((8, 8), dtype=np.uint8)

        widened = (np.stack([image] * 3, axis=-1) if image.ndim == 2
                   else image.copy())

        assert widened.shape == (8, 8, 3)
        assert widened.dtype == image.dtype

    def test_an_rgb_image_is_copied_rather_than_drawn_on(self):
        """The other arm, and the reason it is a copy: the caller's array
        is shown beside the outlined one, so drawing in place would put
        contours on both."""
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        copied = image.copy()
        copied[0, 0] = 255

        assert image[0, 0].tolist() == [0, 0, 0]

    def test_the_helper_still_makes_that_choice(self):
        helper = _nested(P.print_mask_and_flows, "apply_contours_on_image")

        assert "if image.ndim == 2:" in helper
        assert "image.copy()" in helper
