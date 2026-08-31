"""Two advisory guards in the diagnostics figures.

Both wrap decoration -- a verdict stamp and the house style -- and both
are marked advisory in the source. The rule they share is that a
diagnostic PLOT must still be produced when its ornament cannot be: the
figure carries the science, and losing the whole figure because a style
helper was unavailable would trade the measurement for its typography.

`_house` also documents why it sets ink, type sizes and spines by hand:
the figures are built by `plt.subplots` OUTSIDE a style context in some
callers, and rcParams only reach an artist when it is CREATED.
"""
from __future__ import annotations

import builtins

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from spacr import regression_diagnostics as RD


@pytest.fixture()
def axis():
    figure, ax = plt.subplots()
    try:
        yield ax
    finally:
        plt.close(figure)


class TestStampingTheVerdict:

    def test_a_verdict_is_drawn_onto_the_axis(self, axis):
        RD._stamp(axis, "passed")          # must not raise

    def test_a_stamp_helper_that_will_not_import_is_survived(self, axis,
                                                             monkeypatch):
        """THE UNCOVERED GUARD.

        The stamp is advisory. A diagnostics figure without it is still
        the diagnostic; a diagnostics run that died because the stamp
        could not be drawn would have lost the measurement.
        """
        real = builtins.__import__

        def refuse(name, g=None, l=None, fromlist=(), level=0):
            if "regression_qc" in name or "draw_verdict" in (fromlist or ()):
                raise ImportError("regression_qc is unavailable")
            return real(name, g, l, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", refuse)
        RD._stamp(axis, "passed")          # must not raise

    def test_a_stamp_that_raises_while_drawing_is_survived(self, axis,
                                                           monkeypatch):
        """Not only the import: the guard wraps the call as well."""
        from spacr import regression_qc

        def explode(*_a, **_k):
            raise RuntimeError("no renderer for that text")

        monkeypatch.setattr(regression_qc, "draw_verdict", explode)
        RD._stamp(axis, "passed")


class TestTheHouseStyle:

    def test_an_axis_is_titled_and_labelled(self, axis):
        RD._house(axis, title="Residuals", xlabel="fitted", ylabel="resid")
        assert axis.get_title() == "Residuals"
        assert axis.get_xlabel() == "fitted"
        assert axis.get_ylabel() == "resid"

    def test_no_gridlines_ever(self, axis):
        """`grid(False)` is explicit because a caller with a grid-on
        global style would otherwise put one here."""
        axis.grid(True)
        RD._house(axis)
        assert not any(line.get_visible()
                       for line in axis.get_xgridlines()), (
            "a gridline survived the house style")

    def test_a_style_module_that_will_not_import_still_styles_the_axis(
            self, axis, monkeypatch):
        """THE OTHER UNCOVERED GUARD.

        The ink colour is resolved from the theme when that is
        available, and falls back to the reference role when it is not.
        The axis is still titled either way -- the fallback is a
        different colour, not a missing figure.
        """
        real = builtins.__import__

        def refuse(name, g=None, l=None, fromlist=(), level=0):
            if "figures.style" in name or "resolve_ink" in (fromlist or ()):
                raise ImportError("the style module is unavailable")
            return real(name, g, l, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", refuse)
        RD._house(axis, title="Still titled")
        assert axis.get_title() == "Still titled"

    def test_empty_labels_are_not_written(self, axis):
        """`if title:` and friends -- an empty string is "leave it alone",
        not "set it to nothing"."""
        axis.set_title("kept")
        RD._house(axis)
        assert axis.get_title() == "kept"
