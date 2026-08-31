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

    def test_there_is_one_default_list(self):
        """The public function owns the default; its nested helper requires
        the already-resolved list, so two copies cannot drift apart."""
        outer = inspect.getsource(P.plot_images_and_arrays)
        defaults = [line for line in outer.splitlines()
                    if "extensions = ['.npy'" in line]

        assert len(defaults) == 1
        helper = _nested(P.plot_images_and_arrays, "find_files")
        assert "extensions=None" not in helper
        assert "if extensions is None:" not in helper

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

    def test_the_nested_plotter_has_no_unreachable_save_knob(self):
        outer = inspect.getsource(P.plot_images_and_arrays)
        helper = _nested(P.plot_images_and_arrays, "plot_from_file_dict")

        assert "save=False" not in helper
        assert "if save:" not in helper
        assert "save_figure(" not in helper
        assert "save" not in inspect.signature(
            P.plot_images_and_arrays).parameters

        call = outer.index("plot_from_file_dict(file_dict")
        assert "save=" not in outer[call:call + 180]



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

    def test_the_data_transform_is_the_single_vocabulary_check(self):
        source = inspect.getsource(P.volcano_plot)
        copies = [line.strip() for line in source.splitlines()
                  if "Unknown x_transform" in line]

        assert len(copies) == 1
        assert "mode" in copies[0]

    def test_threshold_conversion_relies_on_the_validated_vocabulary(self):
        source = inspect.getsource(P.volcano_plot)

        for name in ("log2", "log10"):
            assert source.count(f'== "{name}"') >= 2
        assert source.count('in ("ln", "log")') == 1
        assert "the only remaining accepted forms are the natural-log aliases" in source


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

    def test_the_helper_only_accepts_the_callers_grayscale_plane(self):
        helper = _nested(P.print_mask_and_flows, "apply_contours_on_image")

        assert "if image.ndim == 2:" not in helper
        assert "image.copy()" not in helper
        assert "cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)" in helper
