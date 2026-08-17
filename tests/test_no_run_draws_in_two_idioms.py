"""A run's figures are all in one visual system.

"i want new versions of all the figures", and then after the first pass, "the
thumbnail in all figures for your new regression plot looks off" -- because a
tile drawn by the old code sat beside a tile drawn by the new one, and two
idioms in one grid reads as a mistake.

THREE SOURCES OF THE OLD IDIOM, found one at a time, each after claiming the
figures were done:

  1. the seven house-style panels shipped and NOTHING CALLED THEM -- a PDF
     was written to disk and the application went on showing what it always
     had;
  2. `spacr/figures/plates.py` and `distributions.py` were built and also
     never called, so a run still drew the wide-and-short plates and the old
     histograms;
  3. `regression_diagnostics.py` drew in seaborn's `deep` -- nine hardcoded
     hexes with no rule behind which one meant what.

This file is the guard against a fourth. It does not check that figures look
nice; it checks that the module a run reaches for is the house-style one.
"""
from __future__ import annotations

import ast
import inspect

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


#: Hues from the palette this repository moved OFF. seaborn's `deep`, which
#: is what every un-converted figure in spaCR was drawn in.
LEGACY_HEXES = ("#4C72B0", "#C44E52", "#55A868", "#8172B2", "#DD8452",
                "#CCB974", "#937860", "#64B5CD", "#DA8BC3")

#: Modules a regression run draws through. Every one must be in the house
#: style; a new one added here without conversion fails this file.
DRAWING_MODULES = ("spacr.regression_diagnostics", "spacr.regression_qc",
                   "spacr.figures.panels", "spacr.figures.plates",
                   "spacr.figures.distributions")


@pytest.mark.parametrize("module_name", DRAWING_MODULES)
def test_no_drawing_module_hardcodes_the_old_palette(module_name):
    """A hex in the source is a colour with no rule behind it.

    The point is not the hue. It is that `#4C72B0` was doing the job of "the
    data" in one panel and "the highlight" in another, so the same colour
    meant two things in one figure sheet.
    """
    from importlib import import_module

    # PARSED, NOT GREPPED. These modules DOCUMENT which old hex used to do
    # what, so a text search matches their own explanation and fails on a
    # module that is already converted -- which is exactly what the first
    # version of this test did.
    tree = ast.parse(inspect.getsource(import_module(module_name)))
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)}
    offenders = sorted(literals & set(LEGACY_HEXES))

    assert not offenders, (
        f"{module_name} still draws in the old palette: {offenders}")


@pytest.mark.parametrize("module_name", DRAWING_MODULES)
def test_no_drawing_module_writes_rcparams_globally(module_name):
    """The style is a context manager. spaCR draws from a long-lived GUI, so
    a global style change restyles every later figure in the session --
    parsed rather than grepped, because these modules DOCUMENT the rule and a
    text search matches their own explanation."""
    from importlib import import_module

    tree = ast.parse(inspect.getsource(import_module(module_name)))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "update"
             and isinstance(node.func.value, ast.Attribute)
             and node.func.value.attr == "rcParams"]

    assert not calls, f"{module_name} writes rcParams globally"


def test_a_run_reaches_for_the_house_style_plates_not_the_old_ones():
    """`plot_plates` is the fallback, not the path. It stays for the case
    where the house-style panel cannot draw -- losing a fit over a figure
    would be the worst possible trade -- but a run tries the new one first."""
    from spacr import ml

    source = inspect.getsource(ml.perform_regression)
    show = source.index("_show_plates")
    legacy = source.index("plot_plates(", show)

    assert show < legacy, (
        "perform_regression calls the legacy plot_plates before the "
        "house-style panel")


def test_a_run_reaches_for_the_house_style_distributions():
    from spacr import ml

    source = inspect.getsource(ml.regression)

    assert "_show_well_distributions" in source
    assert source.index("_show_well_distributions") < \
        source.index("plot_histogram(y,")


def test_the_legacy_volcano_is_off_by_default():
    """"hide my old version behid a boolean that defaults to off"."""
    from spacr.ml import regression

    assert inspect.signature(regression).parameters[
        "legacy_volcano"].default is False


def test_the_diagnostics_draw_inside_the_style_context():
    """rcParams reach an artist when it is CREATED, so a context opened after
    plt.subplots leaves the spines and text at the caller's global style."""
    from spacr import regression_diagnostics

    for name in ("plot_design_diagnostics", "plot_inference_diagnostics",
                 "plot_residual_diagnostics"):
        source = inspect.getsource(getattr(regression_diagnostics, name))
        assert "with figure_style()" in source, name
        assert source.index("with figure_style()") < \
            source.index("plt.subplots("), (
                f"{name} opens the style after creating its figure")


def test_drawing_the_diagnostics_leaves_the_globals_alone():
    """Measured, not asserted from the source."""
    import numpy as np
    import pandas as pd

    from spacr.regression_diagnostics import (plot_design_diagnostics,
                                              plot_inference_diagnostics)

    before = dict(plt.rcParams)
    rng = np.random.default_rng(0)
    plot_design_diagnostics(
        pd.DataFrame(rng.uniform(0, 1, (40, 4)),
                     columns=[f"g{i}" for i in range(4)]),
        block=pd.Series(["p1"] * 20 + ["p2"] * 20))
    p = rng.uniform(0, 1, 120)
    plot_inference_diagnostics(p, adjusted=np.minimum(p * 2, 1), alpha=0.05)
    plt.close("all")

    changed = {k for k in before if str(before[k]) != str(plt.rcParams[k])}
    assert not changed, sorted(changed)
