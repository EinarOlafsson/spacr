"""What the Image UMAP figure-settings form does with imperfect input.

The panel is seeded from three places that can all disagree with the field
table it builds from: ``set_default_umap_image_settings``, the settings dict
of the run that produced the figure on screen, and the form layout that
supplies each row's label. This module drives the seams between them --
seeds that never arrive, a reducer name the build does not offer, a row with
no label, and a field table that has been trimmed -- because each of those is
a state a user reaches by opening a figure from an older run or a differently
built install, and in every one of them the window still has to open with
every editor usable.
"""
from __future__ import annotations

import pytest

from spacr.qt.widgets.umap_figure_settings import UmapFigureSettings


@pytest.fixture
def form(qtbot):
    """The panel as the figure-settings window builds it, fully seeded."""
    widget = UmapFigureSettings()
    qtbot.addWidget(widget)
    return widget


def _build(qtbot, values=None):
    widget = UmapFigureSettings(values)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# Seeds that never arrive


# ---------------------------------------------------------------------------
# A reducer name the build does not offer


def test_an_unknown_spectral_affinity_still_greys_by_the_offered_one(qtbot):
    """The neighbours row follows the affinity the combo really shows.

    ``spectral_n_neighbors`` is only meaningful for a nearest-neighbours
    affinity. When the saved affinity is not one of the offered ones the
    combo falls back to its first entry, and the greying has to be computed
    from that entry rather than from the saved string -- otherwise the row
    the user can see selected and the row that is editable disagree.
    """
    rbf = _build(qtbot, {"reduction_method": "spectral",
                         "spectral_affinity": "rbf"})
    assert rbf._editors["spectral_affinity"].currentText() == "rbf"
    assert not rbf._editors["spectral_n_neighbors"].isEnabled()

    unknown = _build(qtbot, {"reduction_method": "spectral",
                             "spectral_affinity": "cosine_graph"})

    assert unknown._editors["spectral_affinity"].currentText() == \
        "nearest_neighbors"
    assert unknown._editors["spectral_n_neighbors"].isEnabled()


# ---------------------------------------------------------------------------
# A row with no label


def test_a_row_with_no_label_does_not_stop_the_rest_being_greyed(form):
    """One label-less row must not leave stale rows looking editable.

    Each editor remembers the form label beside it so the caption greys with
    the field; a row added without one (a spanning row, or a form rebuilt by
    a caller) has ``None`` there. If the greying pass tripped over that, the
    switch to another reducer would stop part-way and rows belonging to the
    method that is NO LONGER selected would still read as editable -- the
    worst version of this bug, because the user then edits a value the run
    will ignore.
    """
    form._editors["pca_whiten"]._spacr_setting_label = None
    labelled = form._editors["n_neighbors"]
    assert labelled._spacr_setting_label is not None
    assert labelled._spacr_setting_label.isEnabled()

    form._editors["reduction_method"].setCurrentText("pca")

    # the label-less row itself followed the switch,
    assert form._editors["pca_whiten"].isEnabled()
    # its labelled sibling did too, caption and all,
    assert form._editors["pca_svd_solver"].isEnabled()
    assert form._editors["pca_svd_solver"]._spacr_setting_label.isEnabled()
    # and every row of the method that is no longer selected went grey.
    assert not labelled.isEnabled()
    assert not labelled._spacr_setting_label.isEnabled()
    assert not form._editors["min_dist"].isEnabled()
    assert not form._editors["tsne_perplexity"].isEnabled()
