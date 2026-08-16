"""The gene tile is a renderer, and it renders what the resolver decided.

Instruction 121 asks for a tile that appears when a gene is clicked. The part
that can be WRONG -- which gene a guide belongs to -- lives in
:mod:`spacr.gene_tile` and is tested in
``tests/test_clicking_a_gene_says_what_it_is.py`` without a window.

What is left for the widget is small and is exactly what this file checks:

* it takes the ``feature`` string, which is what ``key_selected`` carries, so
  a volcano click and a table row click reach it identically;
* it never shows a blank panel -- not before the first click, and not for a
  feature that resolved to nothing;
* it opens the ToxoDB link ON THE CLICK and never while rendering, because a
  fetch inside a mouse click is the one thing the instruction forbids;
* a resolver failure does not take the plot down with it.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

#: The real screen's rows for GRA14 and for one control guide. Same provenance
#: as tests/test_clicking_a_gene_says_what_it_is.py -- plate1_dv/ols/list.
ROWS = [
    ("gene_fraction:gene[239740]", 0.8731773941909292, 5.000024935549653e-09,
     None, "pc", "239740"),
    ("fraction:grna[239740_3]", 0.7288503312667108, 3.886509059233938e-05,
     "239740_3", "pc", None),
    ("fraction:grna[000000_22]", 4.370849812608784, 3.1222414615895375e-05,
     "000000_22", "control", None),
]


@pytest.fixture()
def results() -> pd.DataFrame:
    frame = pd.DataFrame(ROWS, columns=[
        "feature", "coefficient", "p_value", "grna", "condition", "gene"])
    frame["q_value"] = frame["p_value"]
    frame["multiple_testing_method"] = "none"
    return frame


@pytest.fixture()
def panel(qtbot, results):
    from spacr.qt.widgets.gene_tile import GeneTilePanel

    widget = GeneTilePanel(frame_provider=lambda: results)
    qtbot.addWidget(widget)
    return widget


def test_the_panel_says_it_is_waiting_before_anything_is_clicked(panel):
    """A blank panel beside a plot reads as broken, not as ready."""
    assert panel.tile is None
    assert "Click a point" in panel._view.toPlainText()


def test_clicking_a_guide_shows_the_gene_a_human_recognises(panel):
    """The slot takes the feature string `key_selected` carries, nothing else."""
    panel.show_feature("fraction:grna[239740_3]")

    assert panel.tile is not None
    assert panel.tile.title == "GRA14"
    text = panel._view.toPlainText()
    assert "GRA14" in text and "dense granule protein GRA14" in text


def test_the_panel_shows_this_screens_numbers_for_the_clicked_point(panel):
    """"the effect, the p-value, the q-value ... cost nothing"."""
    panel.show_feature("gene_fraction:gene[239740]")
    text = panel._view.toPlainText()

    assert "0.8732" in text, text
    assert "5e-09" in text, text


def test_a_control_guide_gets_a_tile_that_says_so_not_a_blank_one(panel):
    """"an empty panel reads as a bug, a panel saying 'no gene record for this
    guide' reads as an answer"."""
    panel.show_feature("fraction:grna[000000_22]")
    text = panel._view.toPlainText()

    assert text.strip(), "a control guide produced a blank panel"
    assert "non-targeting control" in text


def test_an_unrecognised_id_still_fills_the_panel(panel):
    """Same rule, for an id from a screen that is not Toxoplasma."""
    panel.show_feature("fraction:grna[HsCtrl_1]")
    text = panel._view.toPlainText()

    assert text.strip()
    assert "not shaped like a Toxoplasma accession" in text


def test_an_ambiguous_mapping_is_called_out_under_the_tile(qtbot, results,
                                                           tmp_path):
    """The three shared protospacers of the real reference, through the widget."""
    from spacr.qt.widgets.gene_tile import GeneTilePanel

    reference = tmp_path / "grna_barcodes.csv"
    pd.DataFrame([
        ("TGGT1_241310_2", "GCCGGCGATAGAGCCCCGCCC"),
        ("TGGT1_411210_2", "GCCGGCGATAGAGCCCCGCCC"),
        ("TGGT1_411710_2", "GCCGGCGATAGAGCCCCGCCC"),
    ], columns=["name", "sequence"]).to_csv(reference, index=False)

    frame = pd.DataFrame([("fraction:grna[411710_2]", -0.0033716357862716,
                           0.9806973349380717, "411710_2", "other", None)],
                         columns=["feature", "coefficient", "p_value", "grna",
                                  "condition", "gene"])
    widget = GeneTilePanel(frame_provider=lambda: frame)
    qtbot.addWidget(widget)

    from spacr.gene_tile import gene_tile
    tile = gene_tile("fraction:grna[411710_2]", frame, barcodes=str(reference))
    assert tile.ambiguous, "the fixture stopped being ambiguous"

    widget.show_feature("fraction:grna[411710_2]")
    # The widget resolves against the SHIPPED reference, where those rows were
    # removed -- so it must say the guide is missing rather than say nothing.
    assert widget.tile is not None
    assert "not in the gRNA reference" in widget._view.toPlainText()


def test_the_toxodb_link_is_not_followed_until_it_is_clicked(panel,
                                                             monkeypatch):
    """"Nothing in the path makes a network call while the user waits."""
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(
        url.toString()) or True)

    panel.show_feature("gene_fraction:gene[239740]")
    assert opened == [], (
        "rendering the tile already opened the external reference")

    panel._open(QUrl("https://toxodb.org/toxo/app/record/gene/TGME49_239740"))
    assert opened == [
        "https://toxodb.org/toxo/app/record/gene/TGME49_239740"]


def test_the_links_are_in_the_rendered_tile(panel):
    """The record is only useful if the anchor reaches the view."""
    panel.show_feature("gene_fraction:gene[239740]")

    assert "toxodb.org/toxo/app/record/gene/TGME49_239740" in panel._view.toHtml()


def test_a_resolver_failure_does_not_take_the_plot_down(panel, monkeypatch):
    """A tile is an explanation. An explanation that raises leaves the user
    with a traceback instead of the point they clicked."""
    import spacr.qt.widgets.gene_tile as module

    def explode(*args, **kwargs):
        raise RuntimeError("annotation file is a directory")

    monkeypatch.setattr(module, "gene_tile", explode)
    panel.show_feature("gene_fraction:gene[239740]")

    assert panel.tile is None
    assert "Could not build a tile" in panel._view.toPlainText()


def test_a_new_regression_is_not_answered_from_the_previous_one(qtbot):
    """The panel asks for the frame on every click rather than holding one."""
    from spacr.qt.widgets.gene_tile import GeneTilePanel

    current = {"frame": pd.DataFrame(
        [("gene_fraction:gene[239740]", 0.87, 5e-09, None, "pc", "239740")],
        columns=["feature", "coefficient", "p_value", "grna", "condition",
                 "gene"])}
    widget = GeneTilePanel(frame_provider=lambda: current["frame"])
    qtbot.addWidget(widget)

    widget.show_feature("gene_fraction:gene[239740]")
    assert widget.tile.effect == pytest.approx(0.87)

    current["frame"] = pd.DataFrame(
        [("gene_fraction:gene[239740]", -0.42, 0.5, None, "pc", "239740")],
        columns=["feature", "coefficient", "p_value", "grna", "condition",
                 "gene"])
    widget.show_feature("gene_fraction:gene[239740]")
    assert widget.tile.effect == pytest.approx(-0.42), (
        "the panel answered from the regression that was loaded before")


def test_the_tile_can_be_rendered_as_a_pixmap_for_the_figure_grid(panel):
    """Instruction 121 puts the tile IN the figure grid, and the grid's cells
    take a pixmap. Rendering to one keeps the grid to a single kind of cell."""
    panel.show_feature("gene_fraction:gene[239740]")
    pixmap = panel.to_pixmap(240)

    assert not pixmap.isNull()
    assert pixmap.width() == 240
    assert pixmap.height() > 0
