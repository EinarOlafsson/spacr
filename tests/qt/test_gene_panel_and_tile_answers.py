"""The gene panel and the gene tile answer from the run loaded now, in words.

Both panels are handed a frame provider rather than a frame, so re-pointing
them is how a host hands over the next run: the numbers shown must follow it,
and ``None`` must mean "no run loaded" rather than a crash.  Nothing is ever
blank -- a term whose genes have no record says so, an ambiguous guide says so
on the status line and a clean one leaves it empty -- and saving the topology
reports the path it wrote, writes nothing on a cancel, and names the file and
the reason when the write fails.

One general test per panel, each walking a case table with pytest's
``subtests`` fixture; every case builds its own panel.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from tests.test_clicking_a_gene_says_what_it_is import REAL_ROWS  # noqa: E402

pytestmark = pytest.mark.qt

TERM = "fraction:grna[239740_3]"
PANEL_COLUMNS = ["feature", "coefficient", "p_value", "grna", "condition",
                 "gene", "n_grna", "n_gene"]
TILE_COLUMNS = ["feature", "coefficient", "p_value", "grna", "condition",
                "gene"]


def _frame(rows, columns):
    frame = pd.DataFrame(rows, columns=columns)
    frame["q_value"] = frame["p_value"]
    frame["multiple_testing_method"] = "none"
    return frame


def test_the_gene_panel_follows_its_provider_and_reports_every_save(
        qtbot, tmp_path, monkeypatch, subtests):
    """Provider swaps reach the tile; saves say what happened, or nothing."""
    from spacr import gene_facts
    from spacr.qt.widgets import gene_panel as gp

    results = _frame(REAL_ROWS, PANEL_COLUMNS)

    def make():
        widget = gp.GenePanel(frame_provider=lambda: results, threaded=False)
        qtbot.addWidget(widget)
        widget.show()
        return widget

    with subtests.test(msg="a new provider reaches the half that reads it"):
        panel = make()
        moved = results.copy()
        moved["coefficient"] = moved["coefficient"] * -1.0
        panel.set_frame_provider(lambda: moved)
        panel.show_feature(TERM)
        assert panel.summary._frame_provider() is moved

    with subtests.test(msg="a term with no gene record says so"):
        panel = make()
        with monkeypatch.context() as patch:
            patch.setattr(gene_facts, "facts_for", lambda genes: {})
            panel.show_feature(TERM)
        assert "No gene number could be parsed" in panel._known.toPlainText()
        assert panel._facts == ()

    def answer_with(path, record):
        def dialog(parent, caption, suggested, filter_):
            record.append(suggested)
            return (str(path) if path else ""), filter_
        return dialog

    blocker = tmp_path / "blocker"
    blocker.write_text("this is a file, not a folder", encoding="utf-8")
    cases = [
        # id, chosen path (None: cancelled), written, status mentions
        ("a chosen file is written and named", tmp_path / "out" / "topo.csv",
         True, None),
        ("a cancel writes nothing and says nothing", None, False, None),
        ("a failed write names the file and why", blocker / "topology.csv",
         False, "Could not write"),
    ]
    for case_id, target, written, failure in cases:
        with subtests.test(msg=case_id):
            panel = make()
            panel.show_feature(TERM)
            before = panel._status.text()
            asked: list = []
            if target is not None and written:
                target.parent.mkdir(parents=True, exist_ok=True)
            work = tmp_path / f"cwd_{len(list(tmp_path.iterdir()))}"
            work.mkdir()
            with monkeypatch.context() as patch:
                patch.setattr(gp.QFileDialog, "getSaveFileName",
                              staticmethod(answer_with(target, asked)))
                patch.chdir(work)
                panel._ask_to_save_topology()
            assert len(asked) == 1
            assert "239740" in asked[0]
            assert list(work.iterdir()) == []
            if target is None:
                assert panel._status.text() == before
            else:
                assert target.exists() is written
                assert str(target) in panel._status.text()
                if failure:
                    assert failure in panel._status.text()


def test_the_gene_tile_panel_answers_from_the_current_run(
        qtbot, tmp_path, monkeypatch, subtests):
    """The tile reads the provider of now and flags only an ambiguous guide."""
    from spacr.gene_tile import gene_tile as resolve
    from spacr.qt.widgets import gene_tile as widget_module

    def make(provider):
        panel = widget_module.GeneTilePanel(frame_provider=provider)
        qtbot.addWidget(panel)
        return panel

    with subtests.test(msg="a new provider replaces the run"):
        first = _frame([("gene_fraction:gene[239740]", 0.87, 5e-09, None,
                         "pc", "239740")], TILE_COLUMNS)
        second = _frame([("gene_fraction:gene[239740]", -0.25, 0.5, None,
                          "pc", "239740")], TILE_COLUMNS)
        panel = make(lambda: first)
        panel.show_feature("gene_fraction:gene[239740]")
        assert "0.87" in panel._view.toPlainText()
        panel.set_frame_provider(lambda: second)
        panel.show_feature("gene_fraction:gene[239740]")
        text = panel._view.toPlainText()
        assert "-0.25" in text
        assert "0.87" not in text

    with subtests.test(msg="dropping the provider still answers"):
        panel = make(lambda: _frame([], TILE_COLUMNS))
        panel.set_frame_provider(None)
        panel.show_feature("gene_fraction:gene[239740]")
        assert panel.tile is not None
        assert panel._view.toPlainText().strip()

    reference = tmp_path / "grna_barcodes.csv"
    pd.DataFrame([
        ("TGGT1_241310_2", "GCCGGCGATAGAGCCCCGCCC"),
        ("TGGT1_411210_2", "GCCGGCGATAGAGCCCCGCCC"),
        ("TGGT1_411710_2", "GCCGGCGATAGAGCCCCGCCC"),
    ], columns=["name", "sequence"]).to_csv(reference, index=False)
    cases = [
        # id, row, own reference, ambiguous
        ("an ambiguous guide says so on the status line",
         ("fraction:grna[411710_2]", -0.0034, 0.98, "411710_2", "other",
          None), True, True),
        ("an unambiguous guide leaves the status line empty",
         (TERM, 0.73, 3.9e-05, "239740_3", "pc", None), False, False),
    ]
    for case_id, row, own_reference, ambiguous in cases:
        with subtests.test(msg=case_id):
            frame = _frame([row], TILE_COLUMNS)
            with monkeypatch.context() as patch:
                if own_reference:
                    patch.setattr(widget_module, "gene_tile",
                                  lambda key, frame: resolve(
                                      key, frame, barcodes=str(reference)))
                panel = make(lambda: frame)
                panel.show_feature(row[0])
            assert panel.tile.ambiguous is ambiguous
            if ambiguous:
                assert "ambiguous mapping" in panel._status.text()
            else:
                assert panel._status.text() == ""
