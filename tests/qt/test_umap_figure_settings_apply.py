"""Applying Image UMAP figure settings to a finished figure, and the form.

"Applies now" has to apply to the figure already on screen and must not move
a point.  The styling half runs against a real matplotlib figure with real
collections, the redraw half against a real payload carrying a real
embedding, and the form against the field table it is built from -- including
seeds that never arrive, reducer names the build does not offer and a trimmed
field table.  One general test per unit, each walking a case table with
pytest's ``subtests`` fixture.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (  # noqa: E402
    QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox,
)

from spacr.qt.widgets import umap_figure_settings as ufs  # noqa: E402
from spacr.qt.widgets.umap_figure_settings import (  # noqa: E402
    UmapFigureSettings, apply_to_figure, redraw_umap_figure,
    restyle_umap_figure,
)

pytestmark = pytest.mark.qt


def _scatter():
    """A figure with one three-colour scatter and one outline, as plotted."""
    from matplotlib.figure import Figure

    fig = Figure()
    axes = fig.add_subplot(111)
    axes.scatter([0.0, 1.0, 2.0], [0.0, 1.0, 0.5],
                 c=["#ff0000", "#00ff00", "#0000ff"])
    axes.plot([0.0, 1.0], [0.0, 1.0])
    return fig


def _payload():
    """Two clusters of six, with labels and image-less records."""
    rng = np.random.default_rng(0)
    embedding = np.vstack([rng.normal(size=(6, 2)),
                           rng.normal(loc=6.0, size=(6, 2))])
    return {"embedding": embedding,
            "labels": np.array([0] * 6 + [1] * 6),
            "records": [{"image": None} for _ in range(12)]}


def _form(qtbot, values=None) -> UmapFigureSettings:
    widget = UmapFigureSettings(values)
    qtbot.addWidget(widget)
    return widget


def _refuse(*_args, **_kwargs):
    raise RuntimeError("refused")


def _no_defaults(monkeypatch):
    import spacr.settings as settings_module

    monkeypatch.setattr(settings_module, "set_default_umap_image_settings",
                        _refuse)


# ---------------------------------------------------------------------------
# restyling what is already drawn
# ---------------------------------------------------------------------------

def test_restyling_sets_the_artists_and_survives_one_that_refuses(subtests):
    """Size, clamped opacity, outline and a reversible fixed colour."""
    with subtests.test(msg="no figure is no work"):
        assert restyle_umap_figure(None, {"dot_size": 20}) is False

    with subtests.test(msg="size, opacity and outline land on the artists"):
        fig = _scatter()
        assert restyle_umap_figure(fig, {"dot_size": 33, "point_alpha": 0.25,
                                         "outline_width": 4.0}) is True
        collection = fig.get_axes()[0].collections[0]
        assert list(collection.get_sizes()) == [33.0]
        assert collection.get_alpha() == 0.25
        assert fig.get_axes()[0].lines[0].get_linewidth() == 4.0

    for case_id, values, alpha, width in [
            ("an opacity above one is clamped", {"point_alpha": 5.0}, 1.0,
             None),
            ("an opacity below zero is clamped", {"point_alpha": -2.0}, 0.0,
             None),
            ("an outline below the floor is raised", {"outline_width": 0.0},
             None, 0.1)]:
        with subtests.test(msg=case_id):
            fig = _scatter()
            assert restyle_umap_figure(fig, values) is True
            if alpha is not None:
                assert fig.get_axes()[0].collections[0].get_alpha() == alpha
            if width is not None:
                assert fig.get_axes()[0].lines[0].get_linewidth() == width

    with subtests.test(msg="a fixed colour can be taken back off"):
        fig = _scatter()
        collection = fig.get_axes()[0].collections[0]
        original = collection.get_facecolor().copy()
        restyle_umap_figure(fig, {"point_color": "red"})
        np.testing.assert_array_equal(collection.get_facecolor(),
                                      [[1.0, 0.0, 0.0, 1.0]])
        restyle_umap_figure(fig, {"point_color": "cluster"})
        np.testing.assert_array_equal(collection.get_facecolor(), original)

    for method, values in [("set_sizes", {"dot_size": 10}),
                           ("set_alpha", {"point_alpha": 0.5}),
                           ("set_facecolor", {"point_color": "red"})]:
        with subtests.test(msg=f"an artist refusing {method}"):
            fig = _scatter()
            setattr(fig.get_axes()[0].collections[0], method, _refuse)
            assert restyle_umap_figure(fig, values) is False

    with subtests.test(msg="an outline refusing a width"):
        fig = _scatter()
        fig.get_axes()[0].lines[0].set_linewidth = _refuse
        assert restyle_umap_figure(fig, {"outline_width": 2.0}) is False


# ---------------------------------------------------------------------------
# redrawing from the same embedding
# ---------------------------------------------------------------------------

def test_a_redraw_keeps_every_point_where_it_was(monkeypatch, tmp_path,
                                                 subtests):
    """Missing or mismatched inputs refuse; a redraw never moves a point."""
    fig = _scatter()
    mismatched = _payload()
    mismatched["labels"] = np.array([0, 1])
    cases = [
        ("no figure", None, {}),
        ("a payload that is not a mapping", fig, "not a mapping"),
        ("an empty embedding list", fig, {"embedding": []}),
        ("a zero-row embedding", fig, {"embedding": np.zeros((0, 2))}),
        ("an embedding without labels", fig,
         {"embedding": np.zeros((12, 2))}),
        ("labels for another embedding", fig, mismatched),
    ]
    for case_id, figure, payload in cases:
        with subtests.test(msg=f"{case_id} is not redrawn"):
            axes = tuple(fig.get_axes())
            assert redraw_umap_figure(figure, payload, {}) is False
            assert tuple(fig.get_axes()) == axes

    with subtests.test(msg="the redrawn figure keeps the coordinates"):
        fig, payload = _scatter(), _payload()
        assert redraw_umap_figure(fig, payload, {"figuresize": 6.0}) is True
        drawn = np.vstack([c.get_offsets()
                           for c in fig.get_axes()[0].collections])
        np.testing.assert_allclose(np.sort(drawn, axis=0),
                                   np.sort(payload["embedding"], axis=0))
        assert tuple(fig.get_size_inches()) == (6.0, 6.0)
        assert fig._spacr_umap_payload is payload

    with subtests.test(msg="the plot labels win over the raw labels"):
        fig, payload = _scatter(), _payload()
        payload["plot_labels"] = np.array([7] * 12)
        payload["labels"] = np.array([0, 1])
        assert redraw_umap_figure(fig, payload, {}) is True

    with subtests.test(msg="a figure that will not resize is still redrawn"):
        fig = _scatter()
        monkeypatch.setattr(fig, "set_size_inches", _refuse)
        assert redraw_umap_figure(fig, _payload(), {"figuresize": 8.0}) is True

    with_images = _payload()
    with_images["records"] = [{"image": str(tmp_path / f"{i}.png")}
                              for i in range(12)]
    with subtests.test(msg="the image overlay is drawn once"):
        drawn = []
        monkeypatch.setattr("spacr.utils.plot_umap_images",
                            lambda *args, **kwargs: drawn.append(args))
        assert redraw_umap_figure(_scatter(), with_images,
                                  {"plot_images": True, "image_nr": 4}) is True
        assert len(drawn) == 1
        assert drawn[0][4] == 4

    with subtests.test(msg="an overlay that fails keeps the embedding"):
        monkeypatch.setattr("spacr.utils.plot_umap_images", _refuse)
        assert redraw_umap_figure(_scatter(), with_images,
                                  {"plot_images": True}) is True


def test_the_tier_of_the_changed_key_decides_the_work(subtests):
    """Nothing changed is no work; a rerun-only change touches nothing."""
    cases = [
        ("nothing changed", {"dot_size": 20}, {"dot_size": 20}, ""),
        ("a style change restyles", {"dot_size": 20}, {"dot_size": 10},
         "style"),
        ("a redraw setting redraws", {"figuresize": 7.0},
         {"figuresize": 10.0}, "redraw"),
        ("a rerun setting alone", {"n_neighbors": 30}, {"n_neighbors": 15},
         ""),
    ]
    for case_id, values, previous, expected in cases:
        with subtests.test(msg=case_id):
            assert apply_to_figure(_scatter(), _payload(), values,
                                   previous=previous) == expected


# ---------------------------------------------------------------------------
# the form
# ---------------------------------------------------------------------------

def test_the_form_opens_usable_whatever_its_seeds(qtbot, monkeypatch,
                                                  subtests):
    """Missing seeds sit at each field's floor; unknown names fall back."""
    with subtests.test(msg="seeded from the shipped defaults"):
        seeded = _form(qtbot)
        assert seeded._editors["n_neighbors"].value() == 1000
        assert seeded._editors["min_dist"].value() == pytest.approx(0.1)
        assert seeded._editors["tsne_perplexity"].value() == 30.0

    for case_id, values, method, enabled, greyed in [
            ("a known reducer, spaced and capitalised",
             {"reduction_method": "  TSNE "}, "tsne", "tsne_perplexity",
             "n_neighbors"),
            ("a reducer this build does not offer",
             {"reduction_method": "wavelet"}, "umap", "n_neighbors",
             "tsne_perplexity")]:
        with subtests.test(msg=case_id):
            form = _form(qtbot, values)
            assert form._editors["reduction_method"].currentText() == method
            assert form.values()["reduction_method"] == method
            assert form._editors[enabled].isEnabled()
            assert not form._editors[greyed].isEnabled()

    _no_defaults(monkeypatch)

    with subtests.test(msg="seeds that never arrive open at each floor"):
        blank = _form(qtbot)
        floors = {"n_neighbors": (QSpinBox, 2), "image_nr": (QSpinBox, 0),
                  "dot_size": (QSpinBox, 1), "min_dist": (QDoubleSpinBox, 0.0),
                  "tsne_perplexity": (QDoubleSpinBox, 0.01),
                  "img_zoom": (QDoubleSpinBox, 0.001)}
        for key, (kind, floor) in floors.items():
            assert isinstance(blank._editors[key], kind), key
            assert blank._editors[key].value() == pytest.approx(floor), key
        assert set(blank.values()) == {f.key for f in ufs.IMAGE_UMAP_FIELDS}
        assert blank.values()["n_neighbors"] == 2

    with subtests.test(msg="an unseeded combo is its first option"):
        blank = _form(qtbot)
        values = blank.values()
        assert isinstance(blank._editors["reduction_method"], QComboBox)
        assert (values["reduction_method"], values["clustering"],
                values["pca_svd_solver"]) == ("umap", "dbscan", "auto")

    with subtests.test(msg="an unseeded text row is None until typed"):
        blank = _form(qtbot)
        assert isinstance(blank._editors["filter_by"], QLineEdit)
        assert blank._editors["filter_by"].text() == ""
        assert blank.values()["filter_by"] is None
        assert blank.values()["row_limit"] is None
        blank._editors["filter_by"].setText("  channel_1  ")
        blank._editors["row_limit"].setText("2500")
        assert blank.values()["filter_by"] == "channel_1"
        assert blank.values()["row_limit"] == 2500
        blank._editors["row_limit"].setText("all")
        assert blank.values()["row_limit"] is None


def test_the_form_emits_once_per_real_change_even_when_trimmed(
        qtbot, monkeypatch, subtests):
    """Debounced, not repeated, nothing for a round trip; trimming is safe."""
    with subtests.test(msg="a change is emitted once"):
        form = _form(qtbot)
        seen = []
        form.settings_changed.connect(seen.append)
        form._editors["dot_size"].setValue(form._editors["dot_size"].value()
                                           + 5)
        form.flush()
        form.flush()
        assert len(seen) == 1
        assert seen[0] == form.values()

    with subtests.test(msg="setting a value back emits nothing"):
        form = _form(qtbot)
        seen = []
        form.settings_changed.connect(seen.append)
        original = form._editors["dot_size"].value()
        form._editors["dot_size"].setValue(original + 5)
        form._editors["dot_size"].setValue(original)
        form.flush()
        assert seen == []

    with subtests.test(msg="a missing editor is skipped"):
        form = _form(qtbot)
        del form._editors["n_neighbors"]
        form._refresh_reducer_fields()
        assert "n_neighbors" not in form.values()

    monkeypatch.setattr(ufs, "IMAGE_UMAP_FIELDS", tuple(
        f for f in ufs.IMAGE_UMAP_FIELDS
        if f.key not in {"reduction_method", "spectral_affinity"}))

    with subtests.test(msg="a table without the reducer combos greys umap"):
        trimmed = _form(qtbot)
        values = trimmed.values()
        assert "reduction_method" not in values
        assert "spectral_affinity" not in values
        assert {key: trimmed._editors[key].isEnabled() for key in (
            "n_neighbors", "min_dist", "tsne_perplexity", "pca_whiten",
            "spectral_n_neighbors")} == {
            "n_neighbors": True, "min_dist": True, "tsne_perplexity": False,
            "pca_whiten": False, "spectral_n_neighbors": False}

    with subtests.test(msg="a trimmed table still emits what it carries"):
        trimmed = _form(qtbot)
        seen = []
        trimmed.settings_changed.connect(seen.append)
        trimmed._editors["dot_size"].setValue(123)
        trimmed.flush()
        assert len(seen) == 1
        assert seen[0]["dot_size"] == 123
        assert "reduction_method" not in seen[0]
        assert seen[0]["n_neighbors"] == trimmed._editors["n_neighbors"].value()
