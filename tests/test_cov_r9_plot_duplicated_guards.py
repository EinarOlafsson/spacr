"""Four guards in ``plot.py`` whose other arm the caller already settled.

Each was a second copy of a decision made a few lines earlier, in a
nested helper that could be called with the un-settled value and never
is. They are worth holding for the same reason a duplicated default
always is: the two copies can drift, and the one that runs is not the
one a reader is looking at.

Every guard is held by what it does to a real call, one parametrised
contract each: which folders and files are found by default, which
transform names are refused and how a threshold is converted, and how a
single-channel image gets outlined. Each row is a helper that does the
call and returns what was observed.
"""
from __future__ import annotations

import inspect

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from spacr import plot as P


@pytest.fixture(autouse=True)
def _never_block_or_leak(monkeypatch):
    """No window opens, and no figure outlives its test."""
    monkeypatch.setattr(P.plt, "show", lambda *a, **k: None)
    yield
    P.plt.close("all")


def _call(request, helper):
    """Run a case helper, giving it the fixtures named in its signature."""
    names = inspect.signature(helper).parameters
    return helper(**{n: request.getfixturevalue(n) for n in names})


def _drawn_titles():
    return sorted(figure.axes[0].get_title()
                  for figure in map(P.plt.figure, P.plt.get_fignums()))


def _default_extensions_cover_what_the_pipeline_writes(tmp_path):
    """The default list finds npy, tif and tiff partners (png is found too
    but only npy and tif data are drawn), so a missing format is a folder
    that looks empty."""
    import imageio

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    mask = np.random.default_rng(0).integers(0, 5, (16, 16)).astype(np.uint8)
    image = np.random.default_rng(1).random((16, 16))
    np.save(a / "n_npy.npy", mask)
    np.save(b / "n_npy.npy", image)
    imageio.imwrite(a / "n_tif.tif", mask)
    np.save(b / "n_tif.npy", image)
    imageio.imwrite(a / "n_tiff.tiff", mask)
    np.save(b / "n_tiff.npy", image)
    imageio.imwrite(a / "n_png.png", mask)
    np.save(b / "n_png.npy", image)

    P.plot_images_and_arrays([str(a), str(b)], threshold=10, randomize=False)

    return _drawn_titles()


def _the_public_function_has_no_save_knob():
    return "save" in inspect.signature(P.plot_images_and_arrays).parameters


def _volcano_refusal(transform):
    frame = pd.DataFrame({"fc": [1.0, 2.0], "p": [0.01, 0.2]})
    with pytest.raises(ValueError) as caught:
        P.volcano_plot(frame, fold_change_col="fc", p_value_col="p",
                       x_transform=transform, fold_change_threshold=1.5)
    message = str(caught.value)
    assert "Unknown x_transform" in message
    return message


def _an_unknown_x_transform_is_refused_by_name():
    """Two copies of the same validation existed and the data transform runs
    first, so an unknown name is refused there. Driven because what matters
    is that the refusal happens, not which line it comes from."""
    message = _volcano_refusal("sqrt")
    return "sqrt" in message


def _threshold_lines(alias):
    frame = pd.DataFrame({"fc": [0.5, 1.0, 2.0], "p": [0.01, 0.2, 0.04]})
    figure, axes, _ = P.volcano_plot(
        frame, fold_change_col="fc", p_value_col="p",
        x_transform=alias, fold_change_threshold=np.e, show=False)
    verticals = sorted({round(float(line.get_xdata()[0]), 6)
                        for line in axes.lines
                        if len(set(line.get_xdata())) == 1})
    P.plt.close(figure)
    return verticals


def _outlined_pixels(image, mask):
    P.print_mask_and_flows(image, mask, None, overlay=True)
    shown = np.asarray(P.plt.gcf().axes[1].images[0].get_array())
    return shown


def _a_two_dimensional_image_is_widened_to_rgb_and_outlined():
    """A single-channel frame is the ordinary case for a mask preview, and
    contour drawing needs three channels: the outline lands in the copy."""
    image = np.zeros((16, 16))
    mask = np.zeros((16, 16))
    mask[4:9, 4:9] = 1

    shown = _outlined_pixels(image, mask)

    assert shown.dtype == np.uint8
    return shown.shape, shown[4, 4].tolist(), bool((shown[0, 0] == 0).all())


@pytest.mark.parametrize("helper, expected", [
    pytest.param(_default_extensions_cover_what_the_pipeline_writes,
                 ["n_npy - Mask", "n_tif - Mask", "n_tiff - Mask"],
                 id="default-extensions-find-npy-tif-tiff"),
    pytest.param(_the_public_function_has_no_save_knob, False,
                 id="no-save-parameter"),
    pytest.param(_an_unknown_x_transform_is_refused_by_name, True,
                 id="unknown-x-transform-refused-by-name"),
    pytest.param(lambda: _threshold_lines("ln"), [-1.0, 0.0, 1.0],
                 id="ln-threshold-in-natural-log-units"),
    pytest.param(lambda: _threshold_lines("log"), [-1.0, 0.0, 1.0],
                 id="log-alias-threshold-in-natural-log-units"),
    pytest.param(_a_two_dimensional_image_is_widened_to_rgb_and_outlined,
                 ((16, 16, 3), [255, 0, 0], True),
                 id="grayscale-widened-to-rgb-and-outlined"),
])
def test_guards_whose_other_arm_the_caller_settled(request, helper, expected):
    observed = _call(request, helper)
    assert observed == expected
