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
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox

from spacr.qt.widgets import umap_figure_settings as ufs
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


def test_seeds_that_never_arrive_open_every_number_at_its_own_floor(
        monkeypatch, qtbot):
    """A window that cannot be seeded must still be a usable window.

    ``_defaults`` swallows a settings module that will not answer, and then
    every numeric field is handed ``None``. If the editors refused to build
    on that, an install whose defaults raise would have no Image UMAP figure
    settings at all -- the user would get an exception where the panel should
    be instead of spin boxes sitting at the bottom of their own ranges, which
    is a state they can simply type over.
    """
    import spacr.settings as settings_module

    seeded = _build(qtbot)
    assert seeded._editors["n_neighbors"].value() == 1000
    assert seeded._editors["min_dist"].value() == pytest.approx(0.1)
    assert seeded._editors["tsne_perplexity"].value() == pytest.approx(30.0)

    def _refuse(values):
        raise RuntimeError("no defaults on this install")

    monkeypatch.setattr(settings_module, "set_default_umap_image_settings",
                        _refuse)

    blank = _build(qtbot)

    # int fields: no seed, so the spin box sits at the field's own low bound.
    assert isinstance(blank._editors["n_neighbors"], QSpinBox)
    assert blank._editors["n_neighbors"].value() == 2
    assert blank._editors["image_nr"].value() == 0
    assert blank._editors["dot_size"].value() == 1
    # float fields, likewise.
    assert isinstance(blank._editors["min_dist"], QDoubleSpinBox)
    assert blank._editors["min_dist"].value() == pytest.approx(0.0)
    assert blank._editors["tsne_perplexity"].value() == pytest.approx(0.01)
    assert blank._editors["img_zoom"].value() == pytest.approx(0.001)
    # and the whole field table is still present and readable.
    assert set(blank.values()) == {f.key for f in ufs.IMAGE_UMAP_FIELDS}
    assert blank.values()["n_neighbors"] == 2


def test_an_unseeded_choice_and_an_unseeded_text_row_are_not_the_same_blank(
        monkeypatch, qtbot):
    """Blank means "first offered option" for a combo, "None" for a line.

    ``values()`` is what the next run is given, so an unseeded combo has to
    hand back a real method name rather than an empty string -- a run started
    with ``reduction_method=""`` reduces nothing -- while an unseeded text
    row has to hand back ``None`` so the run falls back to its own default
    instead of filtering on the empty column name.
    """
    import spacr.settings as settings_module

    def _refuse(values):
        raise RuntimeError("no defaults on this install")

    monkeypatch.setattr(settings_module, "set_default_umap_image_settings",
                        _refuse)

    blank = _build(qtbot)
    values = blank.values()

    assert isinstance(blank._editors["reduction_method"], QComboBox)
    assert values["reduction_method"] == "umap"
    assert values["clustering"] == "dbscan"
    assert values["pca_svd_solver"] == "auto"
    assert isinstance(blank._editors["filter_by"], QLineEdit)
    assert blank._editors["filter_by"].text() == ""
    assert values["filter_by"] is None
    assert values["row_limit"] is None

    # and the blank is a blank, not a broken row: typed into, the very same
    # two editors hand the run a stripped column name and a real integer.
    blank._editors["filter_by"].setText("  channel_1  ")
    blank._editors["row_limit"].setText("2500")
    typed = blank.values()

    assert typed["filter_by"] == "channel_1"
    assert typed["row_limit"] == 2500


# ---------------------------------------------------------------------------
# A reducer name the build does not offer


def test_a_reduction_method_this_build_does_not_offer_falls_back_to_umap(
        qtbot):
    """An old settings dict must not silently select the wrong reducer.

    Method names come out of saved settings, so a figure produced by a build
    that offered a reducer this one does not (or a hand-edited CSV) reaches
    the combo as a string with no matching entry. Selecting nothing would
    leave ``currentText()`` empty and propagate ``reduction_method=""`` to
    the next run; falling back to the first entry keeps the panel honest
    about what it will actually do, and the greying follows that fallback.
    """
    known = _build(qtbot, {"reduction_method": "  TSNE "})
    assert known._editors["reduction_method"].currentText() == "tsne"
    assert known._editors["tsne_perplexity"].isEnabled()
    assert not known._editors["n_neighbors"].isEnabled()

    unknown = _build(qtbot, {"reduction_method": "wavelet"})

    assert unknown._editors["reduction_method"].currentText() == "umap"
    assert unknown.values()["reduction_method"] == "umap"
    assert unknown._editors["n_neighbors"].isEnabled()
    assert not unknown._editors["tsne_perplexity"].isEnabled()


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


# ---------------------------------------------------------------------------
# A trimmed field table


def test_a_field_table_without_the_reducer_combos_still_builds_and_greys(
        monkeypatch, qtbot):
    """The field table is data, and a build without a reducer row must open.

    ``IMAGE_UMAP_FIELDS`` is the single source of what the panel offers, and
    the two rows the greying pass listens to -- the reduction method and the
    spectral affinity -- are entries in it like any other. A panel built from
    a table that does not carry them has nothing to connect to; if that were
    an error rather than a fallback, trimming the table (a build that hides
    the reducer choice, a caller that offers only the display half) would
    take the whole Image UMAP settings window down with it.
    """
    full = _build(qtbot)
    assert isinstance(full._editors["reduction_method"], QComboBox)
    assert isinstance(full._editors["spectral_affinity"], QComboBox)
    assert full.values()["reduction_method"] == "umap"

    dropped = {"reduction_method", "spectral_affinity"}
    monkeypatch.setattr(ufs, "IMAGE_UMAP_FIELDS", tuple(
        f for f in ufs.IMAGE_UMAP_FIELDS if f.key not in dropped))

    trimmed = _build(qtbot)
    values = trimmed.values()

    assert "reduction_method" not in values
    assert "spectral_affinity" not in values
    assert "n_neighbors" in values
    # With no reducer combo the pass falls back to the umap family, so the
    # umap rows stay editable and every other family is greyed.
    assert trimmed._editors["n_neighbors"].isEnabled()
    assert trimmed._editors["min_dist"].isEnabled()
    assert not trimmed._editors["tsne_perplexity"].isEnabled()
    assert not trimmed._editors["pca_whiten"].isEnabled()
    assert not trimmed._editors["spectral_n_neighbors"].isEnabled()


def test_a_trimmed_table_still_emits_the_settings_it_does_carry(
        monkeypatch, qtbot):
    """A panel with no reducer row still has to report its edits.

    The debounced ``settings_changed`` payload is the whole dict the applier
    reads to decide which tier changed. Losing the reducer rows must not
    stop the signal or shrink it to nothing, or a user editing the dot size
    on such a build would see the graph never follow.
    """
    monkeypatch.setattr(ufs, "IMAGE_UMAP_FIELDS", tuple(
        f for f in ufs.IMAGE_UMAP_FIELDS
        if f.key not in {"reduction_method", "spectral_affinity"}))
    trimmed = _build(qtbot)
    seen = []
    trimmed.settings_changed.connect(seen.append)

    trimmed._editors["dot_size"].setValue(123)
    trimmed.flush()

    assert len(seen) == 1
    assert seen[0]["dot_size"] == 123
    assert "reduction_method" not in seen[0]
    assert seen[0]["n_neighbors"] == trimmed._editors["n_neighbors"].value()
