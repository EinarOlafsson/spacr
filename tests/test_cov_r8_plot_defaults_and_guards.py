"""plot.py: the default arguments and the refusals nothing had exercised.

Three contracts, each a parametrised test. Which folder pairs are
drawn, ignored or announced; which axis transforms a volcano plot
accepts and which it refuses by name; and what the plotting functions
hand to the helpers they call. Each row is a helper that does the call
and returns what was observed.

These drive the public functions. Earlier drafts asserted on the
module SOURCE TEXT instead, which passes whatever the code does and
covers nothing; the behaviour is driven here.
"""
from __future__ import annotations

import functools
import inspect

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


def _call(request, helper):
    """Run a case helper, giving it the fixtures named in its signature."""
    names = inspect.signature(helper).parameters
    return helper(**{n: request.getfixturevalue(n) for n in names})


def _arrays_are_found_and_drawn(tmp_path):
    """AN IMAGE AND A MASK, which is what actually draws.

    `plot_images_and_arrays` classifies each side by its unique-value
    count against `threshold`, and only reaches `plt.subplots` when it
    has one continuous image AND one label mask. Two arrays of the same
    kind leave one of them None and the figure is never made.
    """
    import matplotlib.pyplot as plt

    image, mask = tmp_path / "image", tmp_path / "mask"
    image.mkdir()
    mask.mkdir()
    np.save(image / "field1.npy", np.random.rand(64, 64))
    np.save(mask / "field1.npy", np.random.randint(0, 5, (64, 64)))
    plt.close("all")

    P.plot_images_and_arrays([str(image), str(mask)], max_nr=1,
                             randomize=False)

    assert plt.get_fignums(), "nothing was drawn for an image/mask pair"
    axes = plt.gcf().axes
    assert axes[0].images, "the image panel is empty"
    observed = (len(plt.get_fignums()), len(axes))
    plt.close("all")
    return observed


def _two_arrays_of_the_same_kind_draw_nothing(tmp_path):
    """A pair the function cannot classify as image-plus-mask is found,
    grouped, and then silently not drawn."""
    import matplotlib.pyplot as plt

    a, b = _paired_folders(tmp_path)
    plt.close("all")

    P.plot_images_and_arrays([a, b], max_nr=1, randomize=False)

    return plt.get_fignums()


def _a_file_outside_the_image_formats_is_ignored(tmp_path):
    """A folder pair holding only .txt groups nothing, and that must not
    raise or open a blank window."""
    import matplotlib.pyplot as plt

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for folder in (a, b):
        (folder / "notes.txt").write_text("not an image", encoding="utf-8")
    plt.close("all")

    P.plot_images_and_arrays([str(a), str(b)], max_nr=1, randomize=False)

    return plt.get_fignums()


def _an_explicit_extension_list_is_honoured(tmp_path):
    """The pair is two arrays of the same kind, so nothing is DRAWN; an
    explicit extension list is accepted and walks the same files."""
    import matplotlib.pyplot as plt

    a, b = _paired_folders(tmp_path)
    plt.close("all")

    P.plot_images_and_arrays([a, b], extensions=['.npy'], max_nr=1,
                             randomize=False)

    return plt.get_fignums()


def _asking_for_overlay_says_it_uses_the_first_two_folders(tmp_path, capsys):
    """Overlay composes two channels, so a third folder cannot take part."""
    a, b = _paired_folders(tmp_path)
    P.plot_images_and_arrays([a, b], overlay=True, max_nr=1, randomize=False)
    return [line for line in capsys.readouterr().out.splitlines()
            if "first two folders" in line]


def _the_callers_image_is_never_drawn_on_in_place():
    """Contours go onto a copy, whatever shape arrives."""
    pytest.importorskip("cv2")
    mask = np.zeros((16, 16), dtype=np.uint16)
    mask[4:9, 4:9] = 1
    flows = [np.random.rand(16, 16, 3)]

    grey = (np.eye(16) * 255).astype(np.uint8)
    before_grey = grey.copy()
    P.print_mask_and_flows(grey, mask, flows, overlay=True)

    stack = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
    before_stack = stack.copy()
    P.print_mask_and_flows(stack, mask, flows, overlay=True)
    return (bool(np.array_equal(grey, before_grey)),
            bool(np.array_equal(stack, before_stack)))


@pytest.mark.parametrize("helper, expected", [
    pytest.param(_arrays_are_found_and_drawn, (1, 2),
                 id="image-and-mask-draw-one-two-panel-figure"),
    pytest.param(_two_arrays_of_the_same_kind_draw_nothing, [],
                 id="two-arrays-of-same-kind-draw-nothing"),
    pytest.param(_a_file_outside_the_image_formats_is_ignored, [],
                 id="txt-files-ignored"),
    pytest.param(_an_explicit_extension_list_is_honoured, [],
                 id="explicit-extension-list-honoured"),
    pytest.param(_asking_for_overlay_says_it_uses_the_first_two_folders,
                 ["Overlay will only work on the first two folders in the "
                  "list"], id="overlay-announces-first-two-folders"),
    pytest.param(_the_callers_image_is_never_drawn_on_in_place, (True, True),
                 id="callers-image-not-modified"),
])
def test_folder_pairs_are_drawn_ignored_or_announced(request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected


def _positive_frame():
    """All fold changes > 0, so a log transform is legal on them."""
    return pd.DataFrame({"fc": [1.0, 2.0, 3.0],
                         "p": [0.01, 0.5, 0.001],
                         "g": list("abc")})


def _volcano_refusal(frame, transform, threshold, pattern):
    """The part of the ValueError message matching ``pattern``."""
    import re

    with pytest.raises(ValueError) as caught:
        P.volcano_plot(frame, fold_change_col="fc", p_value_col="p",
                       name_col="g", x_transform=transform,
                       fold_change_threshold=threshold)
    return re.search(pattern, str(caught.value)).group(0)


def _an_unknown_transform_is_refused_by_name():
    """Falling through to linear would draw the threshold line in the wrong
    place on a log axis. The message names the value, so a typo in a
    settings file is findable."""
    return _volcano_refusal(_positive_frame(), "cube_root", 2.0,
                            "Unknown x_transform: cube_root")


def _a_negative_fold_change_is_refused_before_the_transform():
    """A log of a negative fold change is not a number, and the message
    says what to do about it rather than letting numpy produce NaN."""
    frame = pd.DataFrame({"fc": [1.0, -1.0], "p": [0.01, 0.5],
                          "g": list("ab")})
    return _volcano_refusal(frame, "log2", 1.0, "requires all fold changes")


def _a_non_positive_threshold_on_a_log_axis_is_refused():
    """log(0) is not a place on the axis."""
    return _volcano_refusal(_positive_frame(), "log2", 0.0, "must be > 0")


def _the_threshold_transform_is_guarded_by_an_identical_check():
    """The data helper runs first, so the threshold helper own raise cannot
    be the one that fires: the error names the transform from the data
    path, with no threshold given."""
    frame = pd.DataFrame({"fc": [1.0, 2.0], "p": [0.01, 0.5],
                          "g": list("ab")})
    with pytest.raises(ValueError) as caught:
        P.volcano_plot(frame, fold_change_col="fc", p_value_col="p",
                       name_col="g", x_transform="cube_root")
    import re

    return re.search("Unknown x_transform", str(caught.value)).group(0)


def _a_named_transform_is_accepted(transform):
    import matplotlib.pyplot as plt

    figure, axes, _rest = P.volcano_plot(
        _positive_frame(), fold_change_col="fc", p_value_col="p",
        name_col="g", x_transform=transform, fold_change_threshold=2.0)

    drawn = bool(axes.collections or axes.lines or axes.patches)
    plt.close(figure)
    return type(figure).__name__, drawn


@pytest.mark.parametrize("helper, expected", [
    pytest.param(_an_unknown_transform_is_refused_by_name,
                 "Unknown x_transform: cube_root", id="unknown-refused-by-name"),
    pytest.param(_a_negative_fold_change_is_refused_before_the_transform,
                 "requires all fold changes", id="negative-fold-change-refused"),
    pytest.param(_a_non_positive_threshold_on_a_log_axis_is_refused,
                 "must be > 0", id="non-positive-log-threshold-refused"),
    pytest.param(_the_threshold_transform_is_guarded_by_an_identical_check,
                 "Unknown x_transform", id="unknown-refused-by-data-check"),
    *[pytest.param(functools.partial(_a_named_transform_is_accepted, t),
                   ("Figure", True), id=f"{t}-accepted-and-drawn")
      for t in ("log2", "log10", "ln", "log")],
])
def test_the_volcano_axis_transform(request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected


def _the_annotation_join_receives_all_four_object_tables(monkeypatch):
    """The call boundary: `tables=None` cannot happen, the caller names
    all four."""
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

    monkeypatch.setattr(io, "_read_and_merge_data", fake_read_and_merge_data)
    monkeypatch.setattr(io, "_read_db", fake_read_db)

    balanced = P.jitterplot_by_annotation("/experiment", "annotation", "value")

    return captured["tables"], set(balanced["annotation"])


def _contour_helper_sees_stack_planes(stack_shape):
    """`original_image` is `stack` or `stack[..., 0]`, both 2D, so the
    outline lands on a 2D plane widened to RGB whatever 2D or 3D stack
    arrives; any other rank is refused."""
    import matplotlib.pyplot as plt

    pytest.importorskip("cv2")
    mask = np.zeros(stack_shape[:2] if len(stack_shape) < 4
                    else stack_shape[1:3])
    mask[4:9, 4:9] = 1
    plt.close("all")
    try:
        P.print_mask_and_flows(np.zeros(stack_shape), mask, None, overlay=True)
    except ValueError as refused:
        plt.close("all")
        return str(refused)
    shown = np.asarray(plt.gcf().axes[1].images[0].get_array()).shape
    plt.close("all")
    return shown


@pytest.mark.parametrize("helper, expected", [
    pytest.param(_the_annotation_join_receives_all_four_object_tables,
                 (["cell", "nucleus", "pathogen", "cytoplasm"],
                  {"control", "treated"}),
                 id="annotation-join-gets-four-tables"),
    pytest.param(lambda: _contour_helper_sees_stack_planes((16, 16)),
                 (16, 16, 3), id="contour-on-2d-stack"),
    pytest.param(lambda: _contour_helper_sees_stack_planes((16, 16, 3)),
                 (16, 16, 3), id="contour-on-first-plane-of-3d-stack"),
    pytest.param(lambda: _contour_helper_sees_stack_planes((2, 16, 16, 3)),
                 "Unexpected stack dimensionality.",
                 id="contour-4d-stack-refused"),
])
def test_helpers_receive_what_the_callers_promise(request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected
