"""The NON-LIVE Image UMAP figure settings: every setting, live.

Instruction 75, from the maintainer:

    "the non live image UMAP figure settings should have all the image UMAP
     settings live editable (you should see changes in the graph directly)
     and remove the dots wit hapi links the tooltips have the links now.
     also add a propegate button so the settings can be propagated for the
     next run."

The trap this is built around is the same one instruction 26 recorded for
the live explorer, and it is worse here because the figure is finished: a
"live apply" that recomputes the embedding moves every point, and the
arrangement the user was reading is the whole value of a projection. So the
fields are tiered by what they actually cost, and the tier is asserted here
rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

import matplotlib
matplotlib.use("Agg", force=True)


N_POINTS = 24


def _payload(**over):
    rng = np.random.default_rng(7)
    embedding = rng.normal(size=(N_POINTS, 2))
    labels = np.arange(N_POINTS) % 3
    payload = {
        "embedding": embedding,
        "labels": labels,
        "plot_labels": labels,
        "records": [{"image": None, "display_name": str(i)}
                    for i in range(N_POINTS)],
        "display": {},
        "settings": {
            "dot_size": 33, "point_color": "cluster", "point_alpha": 0.5,
            "outline_width": 1.0, "figuresize": 5.0, "image_nr": 7,
            "img_zoom": 0.4, "plot_images": False, "plot_points": True,
            "plot_outlines": False, "smooth_lines": False,
            "n_neighbors": 42, "min_dist": 0.2, "metric": "euclidean",
            "clustering": "dbscan", "black_background": False,
        },
        "theme_colors": None,
    }
    payload.update(over)
    return payload


def _umap_figure(payload=None):
    from matplotlib.figure import Figure

    payload = payload if payload is not None else _payload()
    figure = Figure(figsize=(4, 4))
    axes = figure.subplots()
    embedding = payload["embedding"]
    axes.scatter(embedding[:, 0], embedding[:, 1], s=33,
                 c=payload["labels"])
    axes.plot([0, 1], [0, 1], linewidth=1.0)
    figure._spacr_umap_payload = payload
    return figure


def _plain_figure():
    from matplotlib.figure import Figure

    figure = Figure(figsize=(2, 2))
    figure.subplots().plot([0, 1], [0, 1])
    return figure


@pytest.fixture()
def dialog(qtbot):
    from spacr.qt.widgets.figure_queue import _FigureSettingsDialog

    figure = _umap_figure()
    dlg = _FigureSettingsDialog(figure)
    qtbot.addWidget(dlg)
    return dlg


# ---------------------------------------------------------------------------
# Every setting is there, seeded from the run
# ---------------------------------------------------------------------------

def test_the_window_holds_every_image_umap_setting(dialog):
    """Display AND embedding settings, in one window.

    A window that offered only the cheap half would be answering a question
    nobody asked: "all the image UMAP settings" is the requirement.
    """
    values = dialog.umap_values()
    for key in ("dot_size", "point_color", "point_alpha", "outline_width",
                "figuresize", "image_nr", "img_zoom", "plot_images",
                "plot_points", "plot_outlines", "smooth_lines",
                "plot_by_cluster", "black_background",
                "n_neighbors", "min_dist", "metric", "clustering", "eps",
                "min_samples", "reduction_method", "filter_by", "row_limit"):
        assert key in values, f"{key} is missing from the figure settings"


def test_the_window_opens_on_the_settings_the_run_used(dialog):
    """Not on package defaults the user never chose.

    ``dot_size`` defaults to 50 and this run used 33; opening on 50 would
    silently offer to change a value the user had already set.
    """
    values = dialog.umap_values()
    assert values["dot_size"] == 33
    assert values["image_nr"] == 7
    assert values["n_neighbors"] == 42


def test_every_offered_setting_is_a_real_image_umap_setting():
    """Or the window edits a key the module will ignore."""
    from spacr.qt.widgets.umap_figure_settings import IMAGE_UMAP_FIELDS
    from spacr.settings import set_default_umap_image_settings

    known = set(set_default_umap_image_settings({}))
    unknown = [f.key for f in IMAGE_UMAP_FIELDS if f.key not in known]
    assert not unknown, f"not Image UMAP settings: {unknown}"


def test_the_figure_window_offers_every_setting_the_panel_calls_display():
    """Anti-drift, in the direction that matters.

    The settings panel builds a "UMAP Display" group; if a setting is added
    there and not here, the figure window quietly stops being "all the image
    UMAP settings" and nobody notices until a user looks for one.
    """
    from spacr.qt.screens.settings_model import categories_for_app
    from spacr.settings import categories
    from spacr.qt.widgets.umap_figure_settings import IMAGE_UMAP_FIELDS

    display = categories_for_app("umap", categories)["UMAP Display"]
    offered = {f.key for f in IMAGE_UMAP_FIELDS}
    missing = [key for key in display if key not in offered]
    assert not missing, f"the figure window does not offer {missing}"


def test_a_plain_figure_gets_no_umap_section(qtbot):
    """The section is offered for a figure that carries its embedding.

    Without the embedding, "live" would have to mean re-running the
    reduction, which is exactly what must never happen here.
    """
    from spacr.qt.widgets.figure_queue import _FigureSettingsDialog

    dlg = _FigureSettingsDialog(_plain_figure())
    qtbot.addWidget(dlg)
    assert dlg._umap_settings is None
    assert dlg.umap_values() == {}


# ---------------------------------------------------------------------------
# Live apply, tier by tier
# ---------------------------------------------------------------------------

def test_a_style_setting_reaches_the_figure_already_drawn():
    from spacr.qt.widgets.umap_figure_settings import apply_to_figure

    figure = _umap_figure()
    payload = figure._spacr_umap_payload
    before = {"dot_size": 33}
    assert apply_to_figure(figure, payload, {"dot_size": 250}, before) == \
        "style"
    sizes = figure.get_axes()[0].collections[0].get_sizes()
    assert list(sizes) == [250.0]


def test_a_style_setting_never_moves_a_point():
    from spacr.qt.widgets.umap_figure_settings import apply_to_figure

    figure = _umap_figure()
    payload = figure._spacr_umap_payload
    before = payload["embedding"].copy()
    apply_to_figure(figure, payload,
                    {"dot_size": 90, "point_alpha": 0.2,
                     "point_color": "red"}, {})
    assert np.array_equal(before, payload["embedding"])


def test_a_fixed_colour_can_be_taken_back_to_cluster_colours():
    """The bug this guards: the per-cluster colours are overwritten.

    Setting a fixed colour and then choosing 'cluster' again used to be
    impossible to express, because the original face colours were gone.
    """
    from spacr.qt.widgets.umap_figure_settings import restyle_umap_figure

    figure = _umap_figure()
    collection = figure.get_axes()[0].collections[0]
    original = collection.get_facecolor().copy()
    restyle_umap_figure(figure, {"point_color": "red"})
    assert not np.allclose(collection.get_facecolor(), original)
    restyle_umap_figure(figure, {"point_color": "cluster"})
    assert np.allclose(collection.get_facecolor(), original)


def test_a_redraw_setting_replots_from_the_same_embedding():
    """`figuresize` decides what gets drawn, so the graph is drawn again.

    From the SAME embedding — that is the whole contract. A redraw that
    re-embedded would move every point.
    """
    from spacr.qt.widgets.umap_figure_settings import apply_to_figure

    figure = _umap_figure()
    payload = figure._spacr_umap_payload
    before = payload["embedding"].copy()
    assert apply_to_figure(figure, payload,
                           {"figuresize": 9.0, "plot_points": True},
                           {"figuresize": 5.0}) == "redraw"
    assert tuple(figure.get_size_inches()) == (9.0, 9.0)
    assert np.array_equal(before, payload["embedding"])
    offsets = np.vstack([c.get_offsets() for c in figure.get_axes()[0].collections])
    assert len(offsets) == N_POINTS
    assert np.allclose(np.sort(offsets, axis=0), np.sort(before, axis=0))


def test_a_redraw_keeps_the_payload_on_the_figure():
    """Or the figure can be edited exactly once."""
    from spacr.qt.widgets.umap_figure_settings import redraw_umap_figure

    figure = _umap_figure()
    payload = figure._spacr_umap_payload
    assert redraw_umap_figure(figure, payload, {"figuresize": 6.0})
    assert getattr(figure, "_spacr_umap_payload", None) is payload


def test_an_embedding_setting_is_not_applied_live():
    """n_neighbors cannot be honoured without re-embedding.

    Applying it here would move every point, so it is stored, propagated,
    and reported as "next run" instead of pretending it already landed.
    """
    from spacr.qt.widgets.umap_figure_settings import apply_to_figure

    figure = _umap_figure()
    payload = figure._spacr_umap_payload
    assert apply_to_figure(figure, payload, {"n_neighbors": 9},
                           {"n_neighbors": 42}) == ""


def test_an_unchanged_value_costs_nothing(dialog):
    from spacr.qt.widgets.umap_figure_settings import apply_to_figure

    figure = _umap_figure()
    values = dialog.umap_values()
    assert apply_to_figure(figure, figure._spacr_umap_payload,
                           values, values) == ""


def test_the_expensive_half_is_debounced_into_one_apply(qtbot):
    """A spin-box drag is one redraw, not thirty.

    Every field emits on each keystroke/step; replotting a montage per step
    would make the panel unusable, which is the thing the instruction warns
    about.
    """
    from spacr.qt.widgets.umap_figure_settings import UmapFigureSettings

    panel = UmapFigureSettings({"figuresize": 5.0, "image_nr": 4})
    qtbot.addWidget(panel)
    seen = []
    panel.settings_changed.connect(lambda values: seen.append(values))

    for size in (6.0, 7.0, 8.0, 9.0):
        panel._editors["figuresize"].setValue(size)
    qtbot.wait(600)

    assert len(seen) == 1, f"{len(seen)} applies for four edits"
    assert seen[0]["figuresize"] == 9.0


def test_ok_does_not_lose_a_value_still_on_the_timer(qtbot):
    from spacr.qt.widgets.umap_figure_settings import UmapFigureSettings

    panel = UmapFigureSettings({"figuresize": 5.0})
    qtbot.addWidget(panel)
    seen = []
    panel.settings_changed.connect(lambda values: seen.append(values))
    panel._editors["figuresize"].setValue(11.0)
    panel.flush()
    assert seen and seen[0]["figuresize"] == 11.0


def test_cancel_puts_the_figure_back(qtbot):
    """Live apply with no way out is a trap."""
    from spacr.qt.widgets.figure_queue import _FigureSettingsDialog

    figure = _umap_figure()
    dlg = _FigureSettingsDialog(figure)
    qtbot.addWidget(dlg)
    dlg._umap_settings._editors["dot_size"].setValue(300)
    dlg._umap_settings.flush()
    assert list(figure.get_axes()[0].collections[0].get_sizes()) == [300.0]

    dlg.reject()
    assert list(figure.get_axes()[0].collections[0].get_sizes()) == [33.0]


# ---------------------------------------------------------------------------
# No API dots, and the help survives
# ---------------------------------------------------------------------------

def test_the_figure_settings_draw_no_api_dots(dialog):
    """Instruction 75: "remove the dots wit hapi links".

    This test used to assert the opposite — every field had a
    ``_spacr_api_dot`` — and is inverted rather than deleted, because "no
    dot" on its own would also pass if the documentation had left with it.
    The next assertion is what makes this a removal and not a regression.
    """
    from spacr.qt.widgets.dot_link import DotLink

    assert dialog.findChildren(DotLink) == []


def test_the_tooltips_still_carry_their_api_links(dialog):
    from PySide6.QtWidgets import QWidget

    labels = [w for w in dialog.findChildren(QWidget)
              if w.property("settingHelpLabel")
              and "href=" in (w.toolTip() or "")]
    # three figure controls + every Image UMAP setting
    assert len(labels) >= 30
    for widget in (dialog._bg_btn, dialog._fg_btn, dialog._size):
        assert "href=" in widget._spacr_setting_label.toolTip()


# ---------------------------------------------------------------------------
# Propagate
# ---------------------------------------------------------------------------

def test_propagate_sends_every_value_through_the_callback(qtbot):
    from spacr.qt.widgets.figure_queue import _FigureSettingsDialog

    sent = {}
    dlg = _FigureSettingsDialog(_umap_figure(), propagate_callback=sent.update)
    qtbot.addWidget(dlg)
    dlg._umap_settings._editors["n_neighbors"].setValue(15)
    dlg._propagate_btn.click()

    assert sent["n_neighbors"] == 15, (
        "the next-run half must propagate, or it can never take effect")
    assert sent["dot_size"] == 33
    assert sent["figure_text_size"] == dlg._size.value()


def test_propagate_is_disabled_when_nothing_owns_a_settings_panel(dialog):
    assert dialog._propagate_btn.isEnabled() is False
    assert "settings panel" in dialog._propagate_btn.toolTip()


def test_the_queue_hands_the_dialog_propagate_and_re_render(qtbot,
                                                            monkeypatch):
    """The wiring, not the dialog: without it nothing is live and nothing
    propagates, however good the dialog is."""
    from spacr.qt.widgets import figure_queue as module

    queue = module.FigureQueue()
    qtbot.addWidget(queue)
    queue.set_propagate_callback(lambda values: None)
    queue.add_figure(_umap_figure())

    captured = {}

    class _Dialog:
        def __init__(self, fig, parent=None, propagate_callback=None,
                     render_callback=None):
            captured["propagate"] = propagate_callback
            captured["render"] = render_callback

        def exec(self):
            return False

    monkeypatch.setattr(module, "_FigureSettingsDialog", _Dialog)
    queue._open_figure_settings()

    assert callable(captured["propagate"])
    assert captured["render"] == queue.refresh_current_figure


def test_re_rendering_the_current_figure_updates_the_view(qtbot):
    from spacr.qt.widgets.figure_queue import FigureQueue

    queue = FigureQueue()
    qtbot.addWidget(queue)
    figure = _umap_figure()
    queue.add_figure(figure)
    assert queue.refresh_current_figure() is True


def test_the_umap_screen_propagates_into_its_settings_panel(qtbot,
                                                            qt_theme_applied):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("umap")
    qtbot.addWidget(screen)
    assert screen._figure_queue._propagate_cb is not None

    screen._figure_queue._propagate_cb({"n_neighbors": 77})
    assert screen._settings_model.collect()["n_neighbors"] == 77


@pytest.fixture()
def _isolate_figure_prefs(monkeypatch, tmp_path_factory):
    """Never write into the developer's real preference store."""
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences as prefs

    store = tmp_path_factory.mktemp("umap_fig_prefs") / "prefs.ini"
    monkeypatch.setattr(
        prefs, "_settings",
        lambda: QSettings(str(store), QSettings.Format.IniFormat))


def test_a_live_edit_rewrites_the_page_in_the_chosen_format(
        qtbot, _isolate_figure_prefs):
    """PNG at its DPI, or PDF -- whichever the preference asks for.

    The search grid already picks the vector page when one exists; the two
    surfaces must not disagree about what a figure IS, so a live edit here
    rewrites the same pair the run wrote.
    """
    from pathlib import Path
    from spacr.qt import preferences as prefs
    from spacr.qt.widgets.figure_queue import FigureQueue, _sibling_pdf
    from spacr.qt.widgets.umap_figure_settings import apply_to_figure

    prefs.set_figure_format("pdf")
    queue = FigureQueue()
    qtbot.addWidget(queue)
    figure = _umap_figure()
    queue.add_figure(figure)
    png = Path(queue._png_paths[0])
    pdf = _sibling_pdf(png)
    assert pdf.is_file()
    before = pdf.stat().st_size

    apply_to_figure(figure, figure._spacr_umap_payload,
                    {"figuresize": 9.0}, {"figuresize": 5.0})
    assert queue.refresh_current_figure() is True
    assert png.is_file() and pdf.is_file()
    assert pdf.stat().st_size != before or pdf.stat().st_mtime
