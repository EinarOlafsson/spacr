"""A tile is an explanation, so nothing it touches may take the plot down.

The panel sits beside the volcano and the results table and is fed by their
``key_selected`` signal. Everything it reaches for on the way to a tile can be
broken -- the host's frame provider, the resolver itself -- and in both cases
the reader has to keep the point they clicked. The alternative is a traceback
where an explanation should be.

The links are the other half of the contract: a URL is built on render and
followed only on the click, so a tile never puts a network round trip inside
a mouse press.
"""
from __future__ import annotations

import logging
import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QUrl

from spacr.qt.widgets import gene_tile as gt
from spacr.qt.widgets.gene_tile import IDLE_TEXT, TILE_WIDTH, GeneTilePanel

pytestmark = pytest.mark.qt

_COLUMNS = ["feature", "coefficient", "p_value", "grna", "condition", "gene"]
_FEATURE = "gene_fraction:gene[239740]"


def _frame(rows):
    frame = pd.DataFrame(rows, columns=_COLUMNS)
    frame["q_value"] = frame["p_value"]
    frame["multiple_testing_method"] = "none"
    return frame


# -- the host is allowed to be broken ----------------------------------------

def test_a_frame_provider_that_raises_still_produces_a_tile(qtbot, caplog):
    """A broken host costs the numbers, not the identity of the gene.

    The provider is the host's, not the tile's. When it throws, the tile is
    still built from the feature string alone and the reader keeps the point
    they clicked.
    """
    def _explode():
        raise RuntimeError("the results screen has already been torn down")

    panel = GeneTilePanel(frame_provider=_explode)
    qtbot.addWidget(panel)

    with caplog.at_level(logging.ERROR, logger="spacr.qt.gene_tile"):
        panel.show_feature(_FEATURE)

    assert panel.tile is not None
    assert panel.feature == _FEATURE
    assert "239740" in panel._view.toPlainText()
    assert any("could not reach the results frame" in record.message
               for record in caplog.records)


def test_a_resolver_that_raises_says_so_and_leaves_the_plot_alone(qtbot,
                                                                  caplog,
                                                                  monkeypatch):
    """A failure to resolve is reported in the panel, in the panel's colours."""
    def _explode(key, frame):
        raise ValueError("the packaged reference is unreadable")

    monkeypatch.setattr(gt, "gene_tile", _explode)

    panel = GeneTilePanel(frame_provider=lambda: None)
    qtbot.addWidget(panel)
    shown = []
    panel.tile_shown.connect(shown.append)

    with caplog.at_level(logging.ERROR, logger="spacr.qt.gene_tile"):
        panel.show_feature(_FEATURE)

    assert panel.tile is None
    assert panel.feature == _FEATURE
    assert _FEATURE in panel._view.toPlainText()
    assert panel._status.text() == ""
    assert shown == [], "a tile that could not be built is not announced"


def test_a_recovered_selection_clears_the_error_message(qtbot, monkeypatch):
    """The failure notice belongs to one click and must not outlive it."""
    calls = {"n": 0}
    real = gt.gene_tile

    def _first_one_explodes(key, frame):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("nope")
        return real(key, frame)

    monkeypatch.setattr(gt, "gene_tile", _first_one_explodes)

    panel = GeneTilePanel(frame_provider=lambda: None)
    qtbot.addWidget(panel)
    panel.show_feature(_FEATURE)
    assert "unaffected" in panel._view.toPlainText()

    panel.show_feature(_FEATURE)

    assert panel.tile is not None
    assert "unaffected" not in panel._view.toPlainText()


# -- the waiting state --------------------------------------------------------

def test_a_panel_with_no_selection_says_it_is_waiting(qtbot):
    """A blank panel beside a plot reads as broken, so it is never blank."""
    panel = GeneTilePanel()
    qtbot.addWidget(panel)

    assert panel.tile is None
    assert panel.feature == ""
    assert IDLE_TEXT.split(",")[0] in panel._view.toPlainText()
    assert panel._status.text() == ""


def test_clearing_takes_the_panel_back_to_waiting(qtbot):
    panel = GeneTilePanel(frame_provider=lambda: None)
    qtbot.addWidget(panel)
    panel.show_feature(_FEATURE)
    assert panel.tile is not None

    panel.clear()

    assert panel.tile is None
    assert panel.feature == ""
    assert IDLE_TEXT.split(",")[0] in panel._view.toPlainText()


def test_a_panel_with_no_provider_still_resolves_the_identity(qtbot):
    """No frame is not an error: identity resolves, the numbers are missing."""
    panel = GeneTilePanel()
    qtbot.addWidget(panel)
    shown = []
    panel.tile_shown.connect(shown.append)

    panel.show_feature(_FEATURE)

    assert panel.tile is not None
    assert shown == [panel.feature]


# -- the status line ----------------------------------------------------------

def test_an_unresolvable_term_puts_its_first_gap_in_the_status_line(qtbot):
    """Nothing landed on a gene, so the reason takes the status line."""
    panel = GeneTilePanel(frame_provider=lambda: None)
    qtbot.addWidget(panel)

    panel.show_feature("not_a_term_spacr_knows")

    assert panel.tile is not None
    assert panel.tile.resolved is False
    assert panel.tile.unresolved
    assert panel._status.text() != ""


def test_a_resolved_term_leaves_the_status_line_empty(qtbot):
    """Nothing to report means nothing under the tile."""
    frame = _frame([(_FEATURE, 0.87, 5e-09, None, "pc", "239740")])
    panel = GeneTilePanel(frame_provider=lambda: frame)
    qtbot.addWidget(panel)

    panel.show_feature(_FEATURE)

    assert panel.tile.resolved is True
    assert panel.tile.ambiguous is False
    assert panel._status.text() == ""


# -- translation --------------------------------------------------------------

def test_a_language_change_rewrites_the_prose_and_keeps_the_gene_id(qtbot):
    """The words are translated; the accession is data and is not.

    A catalog that "translated" TGGT1_239740 would be renaming a gene.
    """
    from spacr.qt.i18n import tr

    heading = tr("gene id", "is")
    assert heading != "gene id", "the Icelandic catalog answers for this row"

    panel = GeneTilePanel(frame_provider=lambda: None)
    qtbot.addWidget(panel)
    panel.show_feature(_FEATURE)
    english = panel._view.toPlainText()
    assert "gene id" in english

    panel.retranslate_dynamic_content("is")

    icelandic = panel._view.toPlainText()
    assert heading in icelandic
    assert "gene id" not in icelandic
    assert "239740" in icelandic
    assert panel.feature == _FEATURE


def test_the_waiting_sentence_is_translated_too(qtbot, monkeypatch):
    """Even the idle prompt goes through the catalog."""
    monkeypatch.setattr(gt, "tr",
                        lambda source, language=None, **values: "BID EFTIR")

    panel = GeneTilePanel()
    qtbot.addWidget(panel)

    assert "BID EFTIR" in panel._view.toPlainText()


def test_the_failure_notice_is_translated_too(qtbot, monkeypatch):
    """Including the sentence shown when the resolver could not answer."""
    def _explode(key, frame):
        raise ValueError("nope")

    monkeypatch.setattr(gt, "gene_tile", _explode)
    monkeypatch.setattr(
        gt, "tr",
        lambda source, language=None, **values: source.format(**values).upper()
        if values else source.upper())

    panel = GeneTilePanel()
    qtbot.addWidget(panel)
    panel.show_feature(_FEATURE)

    assert "THE PLOT IS" in panel._view.toPlainText()


# -- the links ----------------------------------------------------------------

def test_a_link_is_followed_on_the_click_and_never_on_the_render(qtbot,
                                                                 monkeypatch):
    """Building the URL is string formatting; opening it is a user action."""
    opened = []

    class _Desktop:
        @staticmethod
        def openUrl(url):        # noqa: N802 (Qt naming)
            opened.append(url.toString())
            return True

    monkeypatch.setattr(gt, "QDesktopServices", _Desktop)

    panel = GeneTilePanel(frame_provider=lambda: None)
    qtbot.addWidget(panel)
    panel.show_feature(_FEATURE)

    assert opened == [], "rendering a tile must open nothing"

    panel._view.anchorClicked.emit(QUrl("https://toxodb.org/gene/TGGT1_239740"))

    assert opened == ["https://toxodb.org/gene/TGGT1_239740"]


def test_the_view_never_follows_a_link_by_itself(qtbot):
    """The browser is told not to, so the click reaches the panel's slot."""
    panel = GeneTilePanel()
    qtbot.addWidget(panel)

    assert panel._view.openLinks() is False
    assert panel._view.openExternalLinks() is False
    assert panel._view.property("i18nSkipText") is True


# -- the same record as a grid cell -------------------------------------------

def test_the_tile_renders_to_a_pixmap_the_grid_can_take(qtbot):
    """The figure grid takes pixmaps, so the tile gives it one."""
    frame = _frame([(_FEATURE, 0.87, 5e-09, None, "pc", "239740")])
    panel = GeneTilePanel(frame_provider=lambda: frame)
    qtbot.addWidget(panel)
    panel.show_feature(_FEATURE)

    pixmap = panel.to_pixmap()

    assert not pixmap.isNull()
    assert pixmap.width() == TILE_WIDTH
    assert pixmap.height() >= 1


def test_a_narrower_cell_gets_a_narrower_pixmap(qtbot):
    """The width is the caller's; the height follows the text at that width."""
    panel = GeneTilePanel(frame_provider=lambda: None)
    qtbot.addWidget(panel)
    panel.show_feature(_FEATURE)

    narrow = panel.to_pixmap(120)
    wide = panel.to_pixmap(400)

    assert narrow.width() == 120
    assert wide.width() == 400
    assert narrow.height() >= wide.height()


def test_a_zero_width_request_still_yields_a_drawable_pixmap(qtbot):
    """A grid that asks for nothing gets one pixel, never a null pixmap."""
    panel = GeneTilePanel()
    qtbot.addWidget(panel)

    pixmap = panel.to_pixmap(0)

    assert not pixmap.isNull()
    assert pixmap.width() == 1
    assert pixmap.height() >= 1
