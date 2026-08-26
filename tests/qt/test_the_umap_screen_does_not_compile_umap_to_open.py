"""Opening the UMAP module must not import umap-learn.

Reading the installed metric names touches `umap`, and importing `umap` makes
numba compile pynndescent. Measured: 9.4 s of a 9.6 s screen construction,
spent so a dropdown nobody had clicked could be complete. On the Apple Silicon
laptop that instruction 268 is about, that is the report -- "the CPU sat at
100% for several minutes just starting the module".

NOTHING IS REMOVED. The static names go in at once so the control works
immediately, and the list is completed from the installed library the first
time it is opened -- by which point a user choosing a metric is about to run
UMAP and needs it loaded anyway.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QComboBox                       # noqa: E402

pytestmark = pytest.mark.qt


def _metric_combo(screen):
    for combo in screen.findChildren(QComboBox):
        if "euclidean" in [combo.itemText(i) for i in range(combo.count())]:
            return combo
    return None


@pytest.fixture
def umap_screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("umap")
    qtbot.addWidget(screen)
    return screen


def test_building_the_screen_does_not_import_umap(qtbot):
    """The whole point. Asserted on a screen built inside the test, so an
    earlier test that opened a metric list cannot make this pass."""
    from spacr.qt.screens.app_screen import AppScreen

    before = "umap.umap_" in sys.modules
    screen = AppScreen("umap")
    qtbot.addWidget(screen)
    if not before:
        assert "umap.umap_" not in sys.modules, (
            "opening the UMAP screen imported umap-learn again")


def test_the_metric_control_works_immediately(umap_screen):
    combo = _metric_combo(umap_screen)
    assert combo is not None, "the metric control is gone"
    assert combo.count() >= 20, "the metric list opened empty"
    assert combo.currentText() == "euclidean"


def test_every_offered_metric_is_a_real_one(umap_screen):
    """The short list must not offer something umap would refuse."""
    from spacr.hyperparam import umap_metrics

    combo = _metric_combo(umap_screen)
    offered = {combo.itemText(i) for i in range(combo.count())}
    installed = set(umap_metrics())
    assert offered <= installed, f"not real metrics: {sorted(offered - installed)}"


def test_opening_it_completes_the_list(umap_screen):
    from spacr.hyperparam import umap_metrics

    combo = _metric_combo(umap_screen)
    before = combo.count()
    combo.showPopup()
    assert combo.count() == len(umap_metrics())
    assert combo.count() >= before


def test_the_selection_survives_the_completion(umap_screen):
    combo = _metric_combo(umap_screen)
    combo.setCurrentText("cosine")
    combo.showPopup()
    assert combo.currentText() == "cosine"


def test_it_completes_only_once(umap_screen):
    """A list that has been filled must not be rebuilt on every click --
    and one that could not be filled must not retry forever."""
    combo = _metric_combo(umap_screen)
    combo.showPopup()
    filled = combo.count()
    combo.showPopup()
    combo.showPopup()
    assert combo.count() == filled


def test_the_static_list_is_a_subset_not_a_copy():
    """If umap-learn ever drops a name the static list still offers, this
    test is the thing that says so."""
    from spacr.hyperparam import UMAP_METRICS, umap_metrics

    installed = set(umap_metrics())
    missing = sorted(set(UMAP_METRICS) - installed)
    assert not missing, f"offered up front but not installed: {missing}"
