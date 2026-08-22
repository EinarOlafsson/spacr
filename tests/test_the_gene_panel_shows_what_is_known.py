"""The gene panel: what the clicked dot IS, beside what this screen measured.

Instruction 121, the widget half. The two things it composes are already
tested without a window -- :mod:`spacr.gene_tile` in
``tests/test_clicking_a_gene_says_what_it_is.py`` and :mod:`spacr.gene_facts`
in ``tests/test_a_clicked_gene_carries_its_annotation.py`` -- so what is left
here is exactly what only a widget can get wrong:

* THE GUI THREAD DOES NOT READ FILES. The warm-up runs on a worker thread and
  its result is delivered ON the GUI thread, which is the project rule about
  relaying ``finished`` through a bound-method signal. Both halves are
  asserted, the second by comparing thread objects.
* A CONTROL THAT CANNOT DO ANYTHING IS GREYED OUT AND SAYS WHY -- instruction
  106. Four states, four sentences, all driven.
* NOTHING EMPTY. A control guide, a covariate and a gene with no annotation
  each produce a sentence; none of them produces a panel of blank fields.
* THE PANEL IS NOT A SECOND SOURCE OF TRUTH. The effect and the p-value shown
  are the frame's, and moving the frame moves them.
* ONE CLICK BUILDS ONE TILE. `table.key_selected` is the funnel both the
  volcano and the table pass through, so a second connection would build the
  tile twice for every click.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from tests.test_clicking_a_gene_says_what_it_is import (  # noqa: E402
    AMBIGUOUS_ROWS, CLEAN_ROWS, REAL_ROWS)

pytestmark = pytest.mark.qt

#: The real screen's coefficient columns, as transcribed once in the resolver
#: suite. Imported rather than re-typed: two transcriptions of one file is two
#: chances to be wrong about it, and only one of them would ever be checked.
COLUMNS = ["feature", "coefficient", "p_value", "grna", "condition", "gene",
           "n_grna", "n_gene"]


@pytest.fixture()
def results() -> pd.DataFrame:
    frame = pd.DataFrame(REAL_ROWS, columns=COLUMNS)
    frame["q_value"] = frame["p_value"]
    frame["multiple_testing_method"] = "none"
    return frame


@pytest.fixture()
def screen_reference(tmp_path) -> str:
    """The gRNA reference AS THE SCREEN WAS COUNTED, duplicates included."""
    path = tmp_path / "grna_barcodes.csv"
    pd.DataFrame(AMBIGUOUS_ROWS + CLEAN_ROWS,
                 columns=["name", "sequence"]).to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def panel(qtbot, results):
    """A panel warmed inline, so a test reads like one thread."""
    from spacr.qt.widgets.gene_panel import GenePanel

    widget = GenePanel(frame_provider=lambda: results, threaded=False)
    qtbot.addWidget(widget)
    # SHOWN, because nothing warms a panel nobody shows -- and showing it is
    # what a user does.
    #
    # The warm-up starts a QThread, and Qt calls abort() when a running
    # QThread is destroyed. A panel that is never shown is never closed
    # either, so neither `closeEvent` nor `QApplication.aboutToQuit` fires:
    # `QApplication([]); RegressionResultsPanel()` was SIGABRT. Gating every
    # start on first show fixes the class rather than the case, and a shown
    # panel is inside a running event loop by definition, which is the state
    # both guards need.
    widget.show()
    return widget


def _lower(widget) -> str:
    """What the "what spaCR knows" half of the panel says, as plain text."""
    return widget._known.toPlainText()


# --------------------------------------------------------------------------- #
#  The warm-up: off the GUI thread, delivered on it
# --------------------------------------------------------------------------- #

def test_the_annotation_loads_on_a_worker_thread_and_lands_on_the_gui_thread(
        qtbot, results):
    """The rule this panel exists to keep, and the rule that is easy to break.

    Cold, the tables are 360 ms of CSV reading. They must not be read inside a
    mouse press -- and the handler that puts the result on screen must run on
    the GUI thread, which it only does because JobRunner relays the worker's
    `finished` through a Signal whose receiver is a bound method.
    """
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication

    from spacr import gene_facts
    from spacr.qt.widgets.gene_panel import GenePanel

    gene_facts.clear_cache()
    widget = GenePanel(frame_provider=lambda: results, threaded=True)
    qtbot.addWidget(widget)
    widget.show()

    assert not widget.is_warm(), (
        "the annotation was already loaded before the event loop ran, so it "
        "was read on the GUI thread")

    seen = []
    widget.annotation_ready.connect(
        lambda _n: seen.append(
            QThread.currentThread() is QApplication.instance().thread()))
    with qtbot.waitSignal(widget.annotation_ready, timeout=60000):
        pass

    assert widget.is_warm()
    assert seen == [True], (
        "the result was delivered on the worker thread; a widget touched "
        "from there is undefined behaviour")
    assert len(widget.annotation_columns()) == 23


def test_before_it_is_warm_the_panel_says_it_is_loading(qtbot, results):
    """Not blank. A blank pane beside a plot reads as broken, not as busy."""
    from spacr import gene_facts
    from spacr.qt.widgets.gene_panel import GenePanel, LOADING_TEXT

    gene_facts.clear_cache()
    widget = GenePanel(frame_provider=lambda: results, threaded=True)
    qtbot.addWidget(widget)
    widget.show()

    assert LOADING_TEXT in _lower(widget)
    widget.show_feature("fraction:grna[239740_3]")
    assert LOADING_TEXT in _lower(widget), (
        "a click that beat the warm-up showed an empty annotation block "
        "instead of saying the tables were still loading")
    assert not widget.topology_button.isEnabled()
    assert "still loading" in widget.topology_button.toolTip()

    with qtbot.waitSignal(widget.annotation_ready, timeout=60000):
        pass
    assert "dense granule protein GRA14" in _lower(widget), (
        "the click that arrived first was never re-answered once the "
        "annotation landed")


def test_a_warmed_panel_reads_no_file_on_a_click(panel, monkeypatch):
    """The whole payoff: a click is a dictionary lookup."""
    def refuse(*args, **kwargs):
        raise AssertionError("the GUI thread read a file on a click")

    monkeypatch.setattr(pd, "read_csv", refuse)
    panel.show_feature("fraction:grna[239740_3]")
    assert "GRA14" in _lower(panel)


def test_no_click_opens_a_socket(panel, monkeypatch):
    """"Nothing in the path makes a network call while the user waits"."""
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("a click opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    panel.show_feature("fraction:grna[239740_3]")
    assert "ToxoDB" in panel.summary._view.toPlainText()


# --------------------------------------------------------------------------- #
#  What a click puts on the panel
# --------------------------------------------------------------------------- #

def test_clicking_a_guide_shows_the_gene_and_then_everything_known_about_it(
        panel):
    """Identity first, then this screen, then what spaCR already knew."""
    panel.show_feature("fraction:grna[239740_3]")

    assert panel.tile.title == "GRA14"
    lower = _lower(panel)
    for expected in ("dense granule protein GRA14",   # product
                     "dense granules",                # hyperLOPIT compartment
                     "SP+TM",                         # DeepTMHMM class
                     "residues 1–34 (34 aa)",         # the signal peptide span
                     "in vivo (liver)",               # a published screen
                     "tachyzoite"):                   # stage expression
        assert expected in lower, f"{expected!r} is not on the panel:\n{lower}"


def test_the_screens_own_numbers_come_from_the_frame_and_not_from_anywhere_else(
        qtbot, results):
    """THE PANEL IS NOT A SECOND SOURCE OF TRUTH.

    Driven, not asserted about: the frame's coefficient is moved to a value
    no table on disk carries, and the panel has to follow it. A panel that
    recomputed or re-read would print the old number.
    """
    from spacr.qt.widgets.gene_panel import GenePanel

    moved = results.copy()
    row = moved["feature"] == "gene_fraction:gene[239740]"
    moved.loc[row, "coefficient"] = -12.3456
    moved.loc[row, "p_value"] = 0.75

    widget = GenePanel(frame_provider=lambda: moved, threaded=False)
    qtbot.addWidget(widget)
    widget.show()
    widget.show_feature("gene_fraction:gene[239740]")

    assert widget.tile.effect == pytest.approx(-12.3456)
    assert "-12.35" in widget.summary._view.toPlainText()


def test_a_gene_with_only_part_of_an_annotation_shows_only_that_part(panel):
    """411710 has four fitness scores, no name and no product.

    Its identity block is ABSENT rather than present and blank -- blank fields
    read as "measured, found nothing".
    """
    panel.show_feature("gene_fraction:gene[411710]")
    lower = _lower(panel)

    assert "CRISPR fitness screens" in lower
    assert "identity" not in lower
    assert "nan" not in lower.lower()


def test_a_non_targeting_control_says_it_is_one(panel):
    """"a panel saying "no gene record for this guide" reads as an answer"."""
    panel.show_feature("fraction:grna[000000_22]")
    lower = _lower(panel)

    assert "non-targeting control" in lower
    assert "nothing to annotate" in lower
    assert "CRISPR fitness screens" not in lower, (
        "a control guide was given gene 000000's annotation block")


def test_a_model_covariate_says_it_is_not_a_gene(panel):
    panel.show_feature("Intercept")
    assert "model covariate" in _lower(panel)
    assert not panel.facts


def test_an_unrecognised_string_says_what_it_could_not_resolve(panel):
    panel.show_feature("total_nonsense")
    lower = _lower(panel)
    assert lower.strip()
    assert "total_nonsense" in lower or "not" in lower.lower()


def test_a_resolver_failure_does_not_take_the_panel_down(panel, monkeypatch):
    """A tile is an explanation. An explanation that raises leaves a traceback
    where the user expected the point they clicked."""
    import spacr.qt.widgets.gene_tile as renderer

    # Patched where it is USED, not where it is defined: the renderer binds
    # `gene_tile` at import, so patching the source module would leave the
    # working function in place and the test would pass on nothing.
    monkeypatch.setattr(renderer, "gene_tile", _explode)
    panel.show_feature("fraction:grna[239740_3]")

    assert panel.tile is None
    assert "could not be resolved" in _lower(panel)
    assert not panel.topology_button.isEnabled()


def _explode(*args, **kwargs):
    raise RuntimeError("resolver is broken")


def test_the_ambiguous_guide_gets_one_annotation_block_per_gene(
        panel, results, screen_reference):
    """Three genes, three blocks, each named.

    Three products under one heading would read as one protein with three
    names, which is the opposite of what the ambiguity means. The reference
    is the SCREEN's own, where the eight shared rows still live.
    """
    from spacr.gene_tile import gene_tile

    tile = gene_tile("fraction:grna[411710_2]", results,
                     barcodes=screen_reference)
    assert tile.ambiguous and len(tile.candidates) == 3

    panel._render_known(tile)
    lower = _lower(panel)

    assert [known.gene for known in panel.facts] == [
        candidate.gene for candidate in tile.candidates]
    assert lower.count("what spaCR knows about") == 3
    assert "hypothetical protein" in lower, (
        "241310 has a product description and the ambiguous block dropped it")


# --------------------------------------------------------------------------- #
#  The one control on the panel, and why it is off
# --------------------------------------------------------------------------- #

def test_the_topology_button_is_off_with_a_reason_before_anything_is_clicked(
        panel):
    """Instruction 106: greyed out AND says why, never inert and silent."""
    assert not panel.topology_button.isEnabled()
    assert panel.topology_reason() == panel.topology_button.toolTip()
    assert "Click a gene first" in panel.topology_reason()


def test_the_topology_button_is_off_for_a_protein_with_no_segments(panel):
    """241310 is GLOB: no signal peptide, no helix. Its topology table would
    be a header and nothing else, and a button that writes that is a button
    that lied."""
    panel.show_feature("gene_fraction:gene[241310]")
    assert not panel.topology_button.isEnabled()
    reason = panel.topology_reason()
    assert "241310" in reason and "no signal peptide" in reason


def test_the_topology_button_turns_on_for_a_protein_that_has_segments(panel):
    panel.show_feature("fraction:grna[239740_3]")
    assert panel.topology_button.isEnabled()
    assert panel.topology_reason() == ""
    assert "DeepTMHMM" in panel.topology_button.toolTip()


def test_the_button_says_so_when_the_annotation_is_not_installed(
        qtbot, results, monkeypatch):
    """An install without the bundled tables is a state, not a failure."""
    from spacr import annotation, gene_facts
    from spacr.qt.widgets.gene_panel import GenePanel

    monkeypatch.setattr(annotation, "columns", lambda: [])
    gene_facts.clear_cache()
    try:
        widget = GenePanel(frame_provider=lambda: results, threaded=False)
        qtbot.addWidget(widget)
        widget.show()
        widget.show_feature("fraction:grna[239740_3]")

        assert not widget.topology_button.isEnabled()
        assert "not installed" in widget.topology_button.toolTip()
        assert "not installed" in _lower(widget)
    finally:
        gene_facts.clear_cache()


def test_saving_the_topology_writes_this_genes_segments(panel, tmp_path):
    """Straight through `annotation.supplementary`, which is the function that
    defines what that table is."""
    panel.show_feature("fraction:grna[239740_3]")
    path = tmp_path / "topology.csv"

    assert panel.save_topology(path)
    written = pd.read_csv(path)
    assert len(written) == 1
    assert str(written.loc[0, "gene_nr"]) == "239740"
    assert written.loc[0, "sp_length"] == 34


def test_saving_with_nothing_clicked_writes_nothing(panel, tmp_path):
    path = tmp_path / "nothing.csv"
    assert not panel.save_topology(path)
    assert not path.exists()


# --------------------------------------------------------------------------- #
#  Lifetime and the figure grid
# --------------------------------------------------------------------------- #

def test_clearing_the_panel_puts_it_back_to_waiting(panel):
    panel.show_feature("fraction:grna[239740_3]")
    panel.clear()

    assert panel.tile is None
    assert not panel.facts
    assert not panel.topology_button.isEnabled()
    assert "appears here" in _lower(panel)


def test_the_same_table_is_not_warmed_twice(panel, results):
    """A panel is re-set_frame'd whenever the gene/guide filter moves, and a
    QThread per redraw is a thread for no new genes at all."""
    assert panel.warm_for(results)
    assert not panel.warm_for(results)
    assert not panel.warm_for(pd.DataFrame({"coefficient": [1.0]}))
    assert not panel.warm_for(None)


def test_the_pixmap_carries_both_halves(panel):
    """`_FigureCell` takes a QPixmap, so the tile can be a tile in the figure
    grid without the grid growing a second kind of cell."""
    panel.show_feature("fraction:grna[239740_3]")
    whole = panel.to_pixmap(240)
    top = panel.summary.to_pixmap(240)

    assert whole.width() == 240
    assert whole.height() > top.height(), (
        "the pixmap is the upper half alone; the annotation did not reach "
        "the figure grid")


# --------------------------------------------------------------------------- #
#  Wired into the results panel
# --------------------------------------------------------------------------- #

@pytest.fixture()
def results_panel(qtbot, results):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    widget = RegressionResultsPanel()
    qtbot.addWidget(widget)
    assert widget.set_frame(results, "")
    return widget


def test_the_gene_tab_is_the_panel_with_the_annotation_on_it(results_panel):
    from spacr.qt.widgets.gene_panel import GenePanel

    assert isinstance(results_panel.gene, GenePanel)
    labels = [results_panel.tabs.tabText(i)
              for i in range(results_panel.tabs.count())]
    assert "Gene" in labels


def test_one_click_builds_one_tile(results_panel, qtbot):
    """Connected on the TABLE and not also on the volcano: `key_selected` on
    the table is the funnel both directions already pass through, so a second
    connection would build the tile twice for every click."""
    built = []
    results_panel.gene.tile_shown.connect(built.append)

    results_panel.volcano.key_selected.emit("fraction:grna[239740_3]")
    qtbot.wait(10)

    assert built == ["fraction:grna[239740_3]"], built


def test_a_new_table_does_not_leave_the_previous_screens_gene_on_the_tile(
        results_panel, results):
    """A plot re-rings a point; the tile keeps a whole paragraph about a gene
    from a screen the user is no longer looking at."""
    results_panel.table.select_key("fraction:grna[239740_3]")
    assert results_panel.gene.tile is not None

    other = results.loc[results["feature"] != "fraction:grna[239740_3]"].copy()
    assert results_panel.set_frame(other, "")

    assert results_panel.gene.tile is None
    assert "GRA14" not in _lower(results_panel.gene)
