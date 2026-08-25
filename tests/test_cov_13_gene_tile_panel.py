"""The tile panel follows a re-run's frame, and says when a guide is ambiguous.

Two things the panel owes the reader. It must answer from the regression that
is loaded NOW -- a panel still explaining the previous run's genes is worse
than a blank one, because it looks right. And when a protospacer sits under
more than one gene name, the status line under the tile has to say so; the
effect shown belongs to all of them equally and the tile alone reads as if it
belongs to the one the counting pipeline happened to record.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

_COLUMNS = ["feature", "coefficient", "p_value", "grna", "condition", "gene"]


def _frame(rows):
    frame = pd.DataFrame(rows, columns=_COLUMNS)
    frame["q_value"] = frame["p_value"]
    frame["multiple_testing_method"] = "none"
    return frame


def test_a_new_provider_replaces_the_run_the_panel_answers_from(qtbot):
    """After a re-run the panel must read the new frame, not the old one.

    The panel is deliberately given a callable rather than a frame so it
    cannot keep answering from a finished regression. Re-pointing it is how a
    host hands it the next run, and if the setter did not take, every number
    in the tile would belong to the previous fit while looking current.
    """
    from spacr.qt.widgets.gene_tile import GeneTilePanel

    first = _frame([("gene_fraction:gene[239740]", 0.87, 5e-09,
                     None, "pc", "239740")])
    second = _frame([("gene_fraction:gene[239740]", -0.25, 0.5,
                      None, "pc", "239740")])

    panel = GeneTilePanel(frame_provider=lambda: first)
    qtbot.addWidget(panel)
    panel.show_feature("gene_fraction:gene[239740]")
    assert "0.87" in panel._view.toPlainText()

    panel.set_frame_provider(lambda: second)
    panel.show_feature("gene_fraction:gene[239740]")
    text = panel._view.toPlainText()

    assert "-0.25" in text
    assert "0.87" not in text


def test_dropping_the_provider_leaves_the_panel_answering_without_a_frame(qtbot):
    """``None`` must mean "no run loaded", not a crash on the next click.

    A host clearing the workspace passes ``None``; the panel still has to
    render something for a clicked feature rather than raise inside a slot.
    """
    from spacr.qt.widgets.gene_tile import GeneTilePanel

    panel = GeneTilePanel(frame_provider=lambda: _frame([]))
    qtbot.addWidget(panel)

    panel.set_frame_provider(None)
    panel.show_feature("gene_fraction:gene[239740]")

    assert panel.tile is not None
    assert panel._view.toPlainText().strip()


def test_an_ambiguous_guide_says_so_in_the_status_line(qtbot, tmp_path,
                                                       monkeypatch):
    """The line under the tile must name the ambiguity, not stay empty.

    The tile body lists every candidate gene, which on its own reads like
    extra detail about one result. The status line is what tells the reader
    the reads cannot be told apart, so it has to be filled for the ambiguous
    case and empty for the ordinary one.
    """
    from spacr.gene_tile import gene_tile as resolve
    from spacr.qt.widgets import gene_tile as widget_module

    reference = tmp_path / "grna_barcodes.csv"
    pd.DataFrame([
        ("TGGT1_241310_2", "GCCGGCGATAGAGCCCCGCCC"),
        ("TGGT1_411210_2", "GCCGGCGATAGAGCCCCGCCC"),
        ("TGGT1_411710_2", "GCCGGCGATAGAGCCCCGCCC"),
    ], columns=["name", "sequence"]).to_csv(reference, index=False)

    monkeypatch.setattr(
        widget_module, "gene_tile",
        lambda key, frame: resolve(key, frame, barcodes=str(reference)))

    frame = _frame([("fraction:grna[411710_2]", -0.0034, 0.98,
                     "411710_2", "other", None)])
    panel = widget_module.GeneTilePanel(frame_provider=lambda: frame)
    qtbot.addWidget(panel)

    panel.show_feature("fraction:grna[411710_2]")

    assert panel.tile.ambiguous, "the fixture stopped being ambiguous"
    assert "ambiguous mapping" in panel._status.text()


def test_an_unambiguous_guide_leaves_the_status_line_empty(qtbot):
    """A clean resolution must not decorate itself with a warning."""
    from spacr.qt.widgets.gene_tile import GeneTilePanel

    frame = _frame([("fraction:grna[239740_3]", 0.73, 3.9e-05,
                     "239740_3", "pc", None)])
    panel = GeneTilePanel(frame_provider=lambda: frame)
    qtbot.addWidget(panel)

    panel.show_feature("fraction:grna[239740_3]")

    assert panel.tile.ambiguous is False
    assert panel._status.text() == ""
