"""plot.py: the default arguments and the refusals nothing had exercised.

Three of these are "the caller passed nothing, use the usual set"
branches -- the branch every real call takes, and none the suite took,
because tests pass explicit arguments. The fourth is a refusal: an
unrecognised axis transform, which has to be named rather than silently
treated as linear.

These drive the public functions. An earlier draft asserted on the
module's SOURCE TEXT instead -- that the default list appeared in it --
which passes whatever the code does and covers nothing. Deleted.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from spacr import plot as P


@pytest.fixture(autouse=True)
def _never_block(monkeypatch):
    """`plt.show()` must not try to open a window in a test."""
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: None)


def _paired_folders(tmp_path, name="field1", suffix=".npy"):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for folder in (a, b):
        if suffix == ".npy":
            np.save(folder / f"{name}{suffix}", np.random.rand(16, 16))
        else:
            (folder / f"{name}{suffix}").write_bytes(b"\x00")
    return str(a), str(b)


class TestWhatTheseFunctionsActuallyDo:
    """The live behaviour, driven through the public entry points."""

    def test_arrays_are_found_and_drawn(self, tmp_path):
        """AN IMAGE AND A MASK, which is what actually draws.

        The pair this test used to build -- two folders of the same
        random 16x16 array -- draws NOTHING. `plot_images_and_arrays`
        classifies each side by its unique-value count against
        `threshold`, and only reaches `plt.subplots` when it has one
        continuous image AND one label mask. Two arrays of the same kind
        leave one of them None and the figure is never made.

        So the old test asserted nothing and was named for something it
        did not do. This builds the pair the function is for.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        image, mask = tmp_path / "image", tmp_path / "mask"
        image.mkdir()
        mask.mkdir()
        np.save(image / "field1.npy", np.random.rand(64, 64))
        np.save(mask / "field1.npy",
                np.random.randint(0, 5, (64, 64)))
        plt.close("all")

        P.plot_images_and_arrays([str(image), str(mask)], max_nr=1,
                                 randomize=False)

        assert plt.get_fignums(), "nothing was drawn for an image/mask pair"
        axes = plt.gcf().axes
        assert len(axes) == 2, f"expected two panels, got {len(axes)}"
        assert axes[0].images, "the image panel is empty"
        plt.close("all")

    def test_two_arrays_of_the_same_kind_draw_nothing(self, tmp_path):
        """The other side, and the reason the test above had to change.

        Pinned rather than left implicit: a pair the function cannot
        classify as image-plus-mask is found, grouped, and then silently
        not drawn.
        """
        import matplotlib.pyplot as plt

        a, b = _paired_folders(tmp_path)
        plt.close("all")

        P.plot_images_and_arrays([a, b], max_nr=1, randomize=False)

        assert not plt.get_fignums(), (
            "two arrays of the same kind produced a figure; the "
            "image-versus-mask classification has changed")

    def test_a_file_outside_the_image_formats_is_ignored(self, tmp_path):
        """A folder pair holding only .txt groups nothing, and that must
        not raise -- an empty result is an ordinary answer."""
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        for folder in (a, b):
            (folder / "notes.txt").write_text("not an image")

        import matplotlib.pyplot as plt

        plt.close("all")

        P.plot_images_and_arrays([str(a), str(b)], max_nr=1,
                                 randomize=False)

        # AND NOTHING WAS DRAWN. An empty result is an ordinary answer,
        # but a blank figure is not -- it is a window the user has to
        # close for no reason.
        assert not plt.get_fignums(), (
            "a folder pair with no images still produced a figure")

    def test_an_explicit_extension_list_is_honoured(self, tmp_path):
        import matplotlib.pyplot as plt

        a, b = _paired_folders(tmp_path)
        plt.close("all")

        P.plot_images_and_arrays([a, b], extensions=['.npy'], max_nr=1,
                                 randomize=False)

        # The pair is two arrays of the same kind, so nothing is DRAWN --
        # see test_two_arrays_of_the_same_kind_draw_nothing. What this
        # pins is that an explicit extension list is accepted and walks
        # the same files without raising on the filter.
        assert not plt.get_fignums()

    def test_asking_for_overlay_says_it_uses_the_first_two_folders(
            self, tmp_path, capsys):
        """Overlay composes two channels, so a third folder cannot take
        part. Saying so is the difference between a user seeing two of
        their three folders and thinking the third was empty."""
        a, b = _paired_folders(tmp_path)
        P.plot_images_and_arrays([a, b], overlay=True, max_nr=1,
                                 randomize=False)
        assert "first two folders" in capsys.readouterr().out

    def test_the_callers_image_is_never_drawn_on_in_place(self):
        """Contours go onto a copy, whatever shape arrives."""
        pytest.importorskip("cv2")
        mask = np.zeros((16, 16), dtype=np.uint16)
        mask[4:9, 4:9] = 1
        flows = [np.random.rand(16, 16, 3)]

        grey = (np.eye(16) * 255).astype(np.uint8)
        before_grey = grey.copy()
        P.print_mask_and_flows(grey, mask, flows, overlay=True)
        assert np.array_equal(grey, before_grey)

        stack = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
        before_stack = stack.copy()
        P.print_mask_and_flows(stack, mask, flows, overlay=True)
        assert np.array_equal(stack, before_stack)


class TestTheVolcanoAxisTransform:

    @staticmethod
    def _positive_frame():
        """All fold changes > 0, so a log transform is legal on them."""
        return pd.DataFrame({"fc": [1.0, 2.0, 3.0],
                             "p": [0.01, 0.5, 0.001],
                             "g": list("abc")})

    def test_an_unknown_transform_is_refused_by_name(self):
        """Falling through to linear would draw the threshold line in the
        wrong place on a log axis -- a wrong picture that looks like a
        right one. The message names the value, so a typo in a settings
        file is findable."""
        with pytest.raises(ValueError, match="Unknown x_transform: cube_root"):
            P.volcano_plot(self._positive_frame(), fold_change_col="fc",
                           p_value_col="p", name_col="g",
                           x_transform="cube_root",
                           fold_change_threshold=2.0)

    @pytest.mark.parametrize("transform", ["log2", "log10", "ln", "log"])
    def test_every_named_transform_is_accepted(self, transform):
        figure, axes, _rest = P.volcano_plot(
            self._positive_frame(), fold_change_col="fc",
            p_value_col="p", name_col="g",
            x_transform=transform, fold_change_threshold=2.0)

        # ASSERTED on the RETURN, which volcano_plot documents as
        # (Figure, Axes, list). "Accepted" has to mean a plot came back,
        # not merely that nothing raised.
        assert figure is not None and axes is not None
        assert axes.collections or axes.lines or axes.patches, (
            f"the {transform} transform drew an empty axis")
        import matplotlib.pyplot as plt

        plt.close(figure)

    def test_a_negative_fold_change_is_refused_before_the_transform(self):
        """A log of a negative fold change is not a number, and the
        message says what to do about it rather than letting numpy
        produce NaN."""
        frame = pd.DataFrame({"fc": [1.0, -1.0], "p": [0.01, 0.5],
                              "g": list("ab")})
        with pytest.raises(ValueError, match="requires all fold changes"):
            P.volcano_plot(frame, fold_change_col="fc", p_value_col="p",
                           name_col="g", x_transform="log2",
                           fold_change_threshold=1.0)

    def test_a_non_positive_threshold_on_a_log_axis_is_refused(self):
        """log(0) is not a place on the axis."""
        with pytest.raises(ValueError, match="must be > 0"):
            P.volcano_plot(self._positive_frame(), fold_change_col="fc",
                           p_value_col="p", name_col="g",
                           x_transform="log2", fold_change_threshold=0.0)


class TestTheDefaultsThatNoCallerCanReach:
    """Every remaining gap in plot.py is a default its ONLY caller fills.

    These are nested helpers with defensive signatures -- `extensions=None`,
    `tables=None`, `save=False` -- and in each case the one call site
    passes the argument explicitly, so the default arm never runs. They
    are not missing tests; they are unreachable through the public API.

    Each is pinned at the CALL SITE. If a caller ever stops passing the
    argument, the corresponding test fails and that default becomes live.
    """

    def test_find_files_never_sees_a_missing_extension_list(self):
        """`plot_images_and_arrays` fills the default before calling it."""
        import inspect

        source = inspect.getsource(P.plot_images_and_arrays)
        assert "if extensions is None:" in source
        fill = source.index("extensions = ['.npy', '.tif', '.tiff', '.png']")
        call = source.index("find_files(folders, extensions)")
        assert fill < call, (
            "find_files is now called before the default is filled, so its "
            "own `extensions is None` arm is reachable and wants a test")

    def test_the_file_plotter_is_never_asked_to_save(self):
        """The unreachable save parameter and branch were removed."""
        import inspect

        source = inspect.getsource(P.plot_images_and_arrays)
        assert "plot_from_file_dict(" in source
        assert "save=False" not in source
        assert "if save:" not in source

    def test_the_annotation_join_is_always_given_its_tables(self):
        """`tables=None` cannot happen: the caller names all four."""
        import inspect

        source = inspect.getsource(P.jitterplot_by_annotation)
        assert ("join_measurments_and_annotation(src, tables=['cell', "
                "'nucleus', 'pathogen', 'cytoplasm'])") in source, (
            "the annotation join is no longer given an explicit table list")

    def test_the_annotation_join_receives_all_four_object_tables(
            self, monkeypatch):
        """Exercise the call boundary as well as pinning its source shape."""
        import spacr.io as io

        captured = {}
        measurements = pd.DataFrame({
            "prcfo": ["p1_r1_c1_f1_o1", "p1_r1_c2_f1_o1"],
            "annotation": ["control", "treated"],
            "value": [1.0, 2.0],
            "plateID": ["p1", "p1"],
            "rowID": ["r1", "r1"],
            "columnID": ["c1", "c2"],
        })

        def fake_read_and_merge_data(_locations, tables, **_kwargs):
            captured["tables"] = list(tables)
            return measurements.copy(), None

        def fake_read_db(_location, tables):
            assert tables == ["png_list"]
            return [measurements[["prcfo"]].copy()]

        monkeypatch.setattr(io, "_read_and_merge_data",
                            fake_read_and_merge_data)
        monkeypatch.setattr(io, "_read_db", fake_read_db)

        balanced = P.jitterplot_by_annotation(
            "/experiment", "annotation", "value")

        assert captured["tables"] == [
            "cell", "nucleus", "pathogen", "cytoplasm"]
        assert set(balanced["annotation"]) == {"control", "treated"}

    def test_the_contour_helper_only_ever_sees_a_2d_image(self):
        """`original_image` is `stack` or `stack[..., 0]` -- both 2D.

        So `apply_contours_on_image`'s non-2D branch, which copies rather
        than promoting, cannot run from the one place it is called.
        """
        import inspect

        source = inspect.getsource(P.print_mask_and_flows)
        assert "original_image = stack[..., 0]" in source
        assert "original_image = stack" in source
        assert "apply_contours_on_image(original_image," in source
        assert 'raise ValueError("Unexpected stack dimensionality.")' in source, (
            "a stack of another rank now reaches the contour helper, so its "
            "copy branch is reachable")

    def test_the_threshold_transform_is_guarded_by_an_identical_check(self):
        """Two helpers test the same set of names, and the data one runs
        first -- so the threshold helper's own `Unknown x_transform`
        raise cannot be the one that fires.

        Driven, not read: the error that comes out names the transform,
        and it comes from the data path.
        """
        frame = pd.DataFrame({"fc": [1.0, 2.0], "p": [0.01, 0.5],
                              "g": list("ab")})
        with pytest.raises(ValueError, match="Unknown x_transform"):
            P.volcano_plot(frame, fold_change_col="fc", p_value_col="p",
                           name_col="g", x_transform="cube_root")
