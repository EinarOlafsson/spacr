"""Three PCA guards that the code upstream of them has already ruled out.

None is dead weight: each is one line between a degenerate matrix and a
LinAlgError or an empty plot. But none can fire while the code that
feeds it stays as written, so each is pinned to that code rather than
faked with a monkeypatched internal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.pca_model import (PCAError, PCASpec, component_name,
                                        pca)

pytestmark = pytest.mark.qt


def _frame(n=200, seed=5, constant_columns=0):
    rng = np.random.default_rng(seed)
    data = {
        "plateID": "p1", "rowID": "r1",
        "columnID": [f"c{i % 6 + 1}" for i in range(n)],
        "object_label": range(n),
        "cell_area": rng.lognormal(0.0, 0.3, n),
        "cell_perimeter": rng.lognormal(0.0, 0.25, n),
        "cell_eccentricity": rng.uniform(0.0, 1.0, n),
    }
    for i in range(constant_columns):
        data[f"flat_{i}"] = np.full(n, 3.0)
    return pd.DataFrame(data)


class TestTheMatrixAlwaysHasVarianceLeft:
    """``largest <= 0`` cannot happen, because constants are refused first."""

    def test_a_normal_fit_has_a_positive_largest_singular_value(self):
        result = pca(_frame(), PCASpec(
            features=("cell_area", "cell_perimeter", "cell_eccentricity"),
            n_components=2))

        assert result.n_components >= 1
        assert float(np.max(result.explained_variance)) > 0.0

    def test_every_feature_constant_is_refused_before_the_decomposition(self):
        """THE PIN, part one.

        ``_drop_constant`` raises rather than handing on a matrix with no
        variance -- and it says which columns and what value, which is
        the message a user can act on. The guard downstream would only be
        able to say "no variance left".
        """
        frame = _frame(constant_columns=3)

        with pytest.raises(PCAError, match="constant"):
            pca(frame, PCASpec(features=("flat_0", "flat_1", "flat_2"),
                                   n_components=2))

    def test_one_varying_feature_is_refused_too(self):
        """THE PIN, part two: at least TWO columns reach the decomposition.

        So the standardised matrix always has at least two columns whose
        standard deviation is above tolerance, and its largest singular
        value is therefore above zero.
        """
        frame = _frame(constant_columns=2)

        with pytest.raises(PCAError, match="at least two"):
            pca(frame, PCASpec(features=("cell_area", "flat_0", "flat_1"),
                                   n_components=2))


class TestTheRankIsNeverZero:

    def test_rank_zero_is_arithmetically_impossible_once_largest_is_positive(
            self):
        """THE PIN.

        rank counts singular values above
        ``largest * max(n, p) * eps``. The largest is itself one of them,
        so rank is zero only if ``largest <= largest * max(n, p) * eps``
        -- that is, only if ``max(n, p) * eps >= 1``, which needs an
        array of about 4.5e15 rows or columns. numpy cannot hold one.
        """
        eps = float(np.finfo(float).eps)
        assert eps * 4_000_000_000_000_000 < 1.0, (
            "float64 epsilon changed; the rank guard may now be reachable")

        result = pca(_frame(), PCASpec(
            features=("cell_area", "cell_perimeter", "cell_eccentricity"),
            n_components=3))
        assert result.n_components >= 1, "a positive-variance fit gave rank 0"

    def test_duplicated_columns_reduce_the_rank_without_reaching_zero(self):
        """A genuinely rank-deficient fit still has rank at least one."""
        frame = _frame()
        frame["cell_area_again"] = frame["cell_area"]

        result = pca(frame, PCASpec(
            features=("cell_area", "cell_area_again", "cell_perimeter"),
            n_components=3))

        assert result.n_components >= 1
        assert result.n_components <= 3


class TestSlidingTheOldXOntoY:
    """Clicking a scree bar puts that component on X and slides X to Y."""

    def _panel(self, qtbot):
        from spacr.qt.widgets.pca_view import PCAPanel

        panel = PCAPanel()
        qtbot.addWidget(panel)
        result = pca(_frame(), PCASpec(
            features=("cell_area", "cell_perimeter", "cell_eccentricity"),
            n_components=3))
        panel._result = result
        panel._sync_component_pickers(result)
        return panel

    def test_a_click_slides_the_pair_rather_than_replacing_x_alone(self,
                                                                   qtbot):
        panel = self._panel(qtbot)
        panel._pc_x.setCurrentIndex(panel._pc_x.findData(component_name(0)))
        panel._pc_y.setCurrentIndex(panel._pc_y.findData(component_name(1)))

        panel._on_scree_clicked(2)

        assert panel._pc_x.currentData() == component_name(2)
        assert panel._pc_y.currentData() == component_name(0), (
            "the component the user was already looking at was thrown away")

    def test_clicking_the_component_already_on_x_changes_nothing(self, qtbot):
        """Landing on 'PC3 against PC3' is not a plot."""
        panel = self._panel(qtbot)
        panel._pc_x.setCurrentIndex(panel._pc_x.findData(component_name(1)))
        before = (panel._pc_x.currentData(), panel._pc_y.currentData())

        panel._on_scree_clicked(1)

        assert (panel._pc_x.currentData(), panel._pc_y.currentData()) == before

    def test_both_pickers_always_offer_every_component(self, qtbot):
        """THE PIN.

        ``findData`` on the Y picker cannot fail, because both pickers
        are refilled from the same component list in the same pass -- so
        a name taken from one is always findable in the other. If the Y
        picker is ever narrowed (to exclude the current X, say), the swap
        silently stops sliding and this fails first.
        """
        panel = self._panel(qtbot)

        x_names = [panel._pc_x.itemData(i)
                   for i in range(panel._pc_x.count())]
        y_names = [panel._pc_y.itemData(i)
                   for i in range(panel._pc_y.count())]

        assert x_names == y_names, "the two pickers no longer offer the same set"
        assert x_names, "the pickers were left empty after a fit"
        for name in x_names:
            assert panel._pc_y.findData(name) >= 0
