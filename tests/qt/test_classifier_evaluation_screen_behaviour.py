"""The Classifier Evaluation workbench: scanning, loading, the matrix, a cell, opening crops.

Discovery and loading run on workers (or inline when unthreaded); every failure
ends with an exact sentence on the status line (in the error colour) and a
cleared panel.  The confusion matrix shows counts with each cell's share of its
row on a tooltip; a bundle with no object key keeps the normalised matrix and
says why it cannot be clicked.  A cell splits its objects at the "sure at"
threshold, which re-splits only a cell that is open, and either half is routed
to the crop opener with its keys, reason and per-key scores, or the reason it
was not is shown.  One real four-object bundle: each (true, predicted) pair of
the 2x2 matrix holds exactly one object.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spacr.classifier_evaluation import (
    LeakageReport,
    evaluate_predictions,
    write_evaluation_bundle,
)
from spacr.qt.screens import classifier_evaluation as CE
from spacr.qt.screens.classifier_evaluation import ClassifierEvaluationScreen

pytestmark = pytest.mark.qt

PROMPT = "Click a cell of the matrix to see the objects it counted."
LOADED = "Loaded 4 held-out prediction(s) · leakage audit: passed."


@pytest.fixture
def evaluation_root(tmp_path):
    """One bundle: 4 objects, one in each cell of the 2x2 confusion matrix."""
    paths = ["plate1_A01_1_1.png", "plate1_A02_1_2.png",
             "plate2_B01_1_3.png", "plate2_B02_1_4.png"]
    evaluation = evaluate_predictions(
        [0, 1, 0, 1],
        np.asarray([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]]),
        paths,
        classes=["negative", "positive"],
        fold_ids=[1, 1, 2, 2],
        calibration_method="none",
        calibration_bins=4,
    )
    report = LeakageReport(
        group_by="well", train_samples=2, validation_samples=2,
        overlap_counts={"exact": 0, "augmentation_family": 0, "object": 0,
                        "field": 0, "well": 0, "plate": 1},
        examples={"exact": [], "augmentation_family": [], "object": [],
                  "field": [], "well": [], "plate": ["plate1"]},
        split_name="outer_1",
    )
    return write_evaluation_bundle(
        tmp_path / "model" / "evaluation", evaluation,
        leakage_reports=[report]).parents[1]


@pytest.fixture
def make(qtbot, qt_theme_applied):
    """Build a workbench; ``root`` given means scanned (and so loaded) inline."""
    def _build(root=None, threaded=False):
        widget = ClassifierEvaluationScreen(threaded=threaded)
        qtbot.addWidget(widget)
        if root is not None:
            widget._source.setText(str(root))
            widget.scan()
        return widget
    return _build


def _error_style():
    return f"color: {CE.active_palette()['error']};"


def _raise(exc):
    def _raiser(*_args, **_kwargs):
        raise exc
    return _raiser


def _texts(listing):
    return [listing.item(index).text() for index in range(listing.count())]


def _table(table):
    return [[None if table.item(row, column) is None
             else table.item(row, column).text()
             for column in range(table.columnCount())]
            for row in range(table.rowCount())]


def test_scan_reports_what_it_found_or_why_it_found_nothing(make, monkeypatch,
                                                            evaluation_root,
                                                            tmp_path, subtests):
    """No folder, no bundles and a failed discovery each clear the panel and say so."""
    empty = tmp_path / "no_runs_here"
    empty.mkdir()

    with subtests.test(msg="scanning with no folder chosen asks for one"):
        screen = make()
        screen.scan()
        assert screen._status.text() == "Choose a results folder first."
        assert screen._status.styleSheet() == _error_style()
        assert screen.bundles == []

    with subtests.test(msg="a folder with bundles loads the first"):
        screen = make(evaluation_root)
        assert screen.bundles == CE.find_evaluation_bundles(evaluation_root)
        assert [screen._bundle_choice.itemText(i)
                for i in range(screen._bundle_choice.count())] == ["evaluation"]
        assert screen._status.text() == LOADED
        assert screen._status.styleSheet() == ""
        assert screen._busy is False
        assert screen._copy_path.isEnabled() and screen._open_folder.isEnabled()

    with subtests.test(msg="a folder with no bundles"):
        screen = make(evaluation_root)
        screen._source.setText(str(empty))
        screen.scan()
        assert screen._status.text() == (
            "No evaluation_manifest.json files were found.")
        assert screen._status.styleSheet() == _error_style()
        assert screen.bundle is None
        assert screen._confusion.rowCount() == 0
        assert screen._bundle_choice.count() == 0
        assert not screen._copy_path.isEnabled()

    with subtests.test(msg="a discovery that raised"):
        screen = make()
        with monkeypatch.context() as patch:
            patch.setattr(CE, "find_evaluation_bundles",
                          _raise(PermissionError("results is 0700")))
            screen._source.setText(str(tmp_path))
            screen.scan()
        assert screen._status.text() == (
            "Could not scan classifier evaluations: "
            "PermissionError: results is 0700")
        assert screen.last_error == "PermissionError: results is 0700"
        assert screen._status.styleSheet() == _error_style()
        assert screen.bundle is None and screen.bundles == []

    with subtests.test(msg="a bundle outside the root is labelled by its path"):
        manifests = CE.find_evaluation_bundles(evaluation_root)
        screen = make()
        with monkeypatch.context() as patch:
            patch.setattr(CE, "find_evaluation_bundles",
                          lambda root: list(manifests))
            screen._source.setText(str(empty))
            screen.scan()
        assert [screen._bundle_choice.itemText(i)
                for i in range(screen._bundle_choice.count())] == [
            str(manifests[0].parent)]
        assert screen._status.text() == LOADED

    with subtests.test(msg="a scan while one is running is ignored"):
        screen = make()
        screen._source.setText(str(evaluation_root))
        screen._busy = True
        screen._status.setText("busy")
        screen.scan()
        assert screen._status.text() == "busy"
        assert screen.bundles == []
        screen._busy = False


def test_loading_a_bundle_fills_the_panel_or_clears_it(make, monkeypatch,
                                                       evaluation_root,
                                                       subtests):
    """A parse failure clears every view; an idle load with nothing chosen is a no-op."""
    with subtests.test(msg="the tables are empty before anything is loaded"):
        screen = make()
        assert screen._predictions_frame().empty
        screen._render_predictions()
        assert screen._predictions.rowCount() == 0
        screen.show_cell("negative", "positive")
        assert screen._cell is None
        assert screen._cell_summary.text() == PROMPT

    with subtests.test(msg="a selector with nothing in it loads nothing"):
        screen = make()
        screen._load_selected_bundle()
        assert screen.bundle is None
        assert screen._busy is False
        assert screen._status.text() == (
            "Choose or drop a classifier run folder, then select Scan.")

    with subtests.test(msg="a load while one is running is ignored"):
        screen = make(evaluation_root)
        loaded = screen.bundle
        screen._busy = True
        with monkeypatch.context() as patch:
            patch.setattr(CE, "load_evaluation_bundle",
                          _raise(AssertionError("must not be read")))
            screen._load_selected_bundle()
        screen._busy = False
        assert screen.bundle is loaded
        assert screen._status.text() == LOADED

    with subtests.test(msg="a loaded bundle fills every table"):
        screen = make(evaluation_root)
        assert screen._predictions.rowCount() == 4
        assert len(screen._predictions_frame()) == 4
        assert json.loads(screen._overview.toPlainText())["results"] == str(
            screen.bundle["path"])

    with subtests.test(msg="a bundle that cannot be parsed"):
        screen = make(evaluation_root)
        screen.show_cell("negative", "positive")
        with monkeypatch.context() as patch:
            patch.setattr(CE, "load_evaluation_bundle", _raise(
                json.JSONDecodeError("bad manifest", "{", 0)))
            screen._load_selected_bundle()
        assert screen.bundle is None
        assert screen._status.text() == (
            "Could not load classifier evaluation: JSONDecodeError: "
            "bad manifest: line 1 column 1 (char 0)")
        assert screen._status.styleSheet() == _error_style()
        assert screen._cell is None
        assert screen._confusion.rowCount() == 0
        assert screen._predictions.rowCount() == 0
        assert screen._overview.toPlainText() == ""
        assert not screen._copy_path.isEnabled()


def test_the_confusion_matrix_shows_counts_with_row_shares(make, monkeypatch,
                                                           evaluation_root,
                                                           subtests):
    """Counts in the cells, shares on tooltips, and a holed cell skipped."""
    screen = make(evaluation_root)
    share = "1 object(s) · 50.0% of this row"

    with subtests.test(msg="a loaded matrix"):
        assert _table(screen._confusion) == [["negative", "1", "1"],
                                             ["positive", "1", "1"]]
        assert [[screen._confusion.item(row, column).toolTip()
                 for column in (1, 2)] for row in (0, 1)] == [[share] * 2] * 2
        assert screen._confusion_ranking.text() == (
            "Your worst confusion is negative → positive: 1 object(s), 50% of "
            "all errors and 50% of everything truly negative.\nThen positive "
            "→ negative: 1 object(s), 50% of all errors and 50% of everything "
            "truly positive.")
        assert screen._threshold.value() == pytest.approx(0.75)

    with subtests.test(msg="a cell with no item is skipped, not a traceback"):
        original = screen._render_frame

        def holed(table, frame, **kwargs):
            original(table, frame, **kwargs)
            if table is screen._confusion and table.rowCount():
                table.takeItem(0, 1)

        screen._threshold.setValue(0.5)
        with monkeypatch.context() as patch:
            patch.setattr(screen, "_render_frame", holed)
            screen._render_confusion(dict(screen.bundle.get("summary") or {}))
        assert screen._confusion.item(0, 1) is None
        assert screen._confusion.item(0, 2).toolTip() == share
        assert screen._confusion.item(1, 1).toolTip() == share
        assert screen._confusion.item(1, 2).toolTip() == share
        assert screen._confusion_ranking.text().startswith(
            "Your worst confusion is negative → positive")
        assert screen._threshold.value() == pytest.approx(0.75)
        assert screen._cell is None

    with subtests.test(msg="a bundle that cannot be broken down by object"):
        screen.show_cell("negative", "positive")
        with monkeypatch.context() as patch:
            patch.setattr(CE.cx, "confusion_counts", _raise(
                CE.cx.ConfusionError("no object key column in this bundle")))
            screen._render_confusion(dict(screen.bundle.get("summary") or {}))
        assert screen._confusion_ranking.text() == (
            "This bundle cannot be broken down by object: "
            "no object key column in this bundle")
        assert screen._cell is None
        assert _table(screen._confusion) == [
            [str(index)] + [CE._cell_text(value) for value in row]
            for index, row in screen.bundle["confusion_normalized"].iterrows()]
        assert screen._confusion.rowCount() == 2


def test_the_sure_at_threshold_resplits_only_an_open_cell(make,
                                                          evaluation_root,
                                                          subtests):
    """With nothing open the control is inert; with a cell open it moves the split."""
    screen = make(evaluation_root)

    with subtests.test(msg="no cell open"):
        screen._clear_cell()
        screen._threshold.setValue(0.40)
        assert screen._cell is None
        assert screen._cell_summary.text() == PROMPT
        assert screen._high_list.count() == 0 and screen._low_list.count() == 0

    with subtests.test(msg="the open cell is split at the control's value"):
        # plate2_B01 is a negative called positive at 0.60 confidence.
        screen.show_cell("negative", "positive")
        assert screen._cell.threshold == pytest.approx(0.40)
        assert list(screen._cell.keys("high")) == ["plate2_B01_1_3.png"]
        assert screen._cell.low.empty
        assert _texts(screen._high_list) == ["0.600  plate2_B01_1_3.png"]
        assert _texts(screen._low_list) == ["(none)"]
        assert screen._high_open.text() == "Open 1 sure and wrong"
        assert screen._low_open.text() == "Nothing to open"

    with subtests.test(msg="raising the bar moves the object to the low half"):
        screen._threshold.setValue(0.75)
        assert screen._cell.threshold == pytest.approx(0.75)
        assert screen._cell.high.empty
        assert list(screen._cell.keys("low")) == ["plate2_B01_1_3.png"]
        assert _texts(screen._high_list) == ["(none)"]
        assert _texts(screen._low_list) == ["0.600  plate2_B01_1_3.png"]
        assert screen._cell_summary.text() == (
            "annotated negative · predicted positive — 1 errors: 0 at "
            "confidence ≥ 0.75 (suspect the label) and 1 below it (suspect "
            "the boundary).")
        assert screen._cell_breakdown.text() == (
            "well: 1 object(s) across 1 well(s); worst is plate2_B01 with 1 "
            "(100%).\nplate: 1 object(s) across 1 plate(s); worst is plate2 "
            "with 1 (100%).")
        assert screen._low_open.text() == "Open 1 unsure and wrong"


def test_open_cell_routes_a_half_to_the_opener_or_says_why_not(make,
                                                               monkeypatch,
                                                               evaluation_root,
                                                               subtests):
    """The opener gets keys, reason and scores; every refusal is a status line."""
    screen = make(evaluation_root)
    calls = []

    def opener(keys, **kwargs):
        calls.append((list(keys), kwargs))
        return "opened"

    with subtests.test(msg="the unsure half of an error"):
        calls.clear()
        screen.show_cell("negative", "positive")
        cell = screen._cell
        with monkeypatch.context() as patch:
            patch.setattr(CE, "open_objects", opener)
            assert screen._open_low() == "opened"
        assert calls == [(["plate2_B01_1_3.png"], {
            "reason": cell.reason("low"), "source": CE.LINK_SOURCE,
            "context": {"scores": {"plate2_B01_1_3.png": 0.6},
                        "threshold": 0.75, "true_class": "negative",
                        "predicted_class": "positive", "which": "low"}})]

    with subtests.test(msg="the sure half of a correct cell"):
        calls.clear()
        screen.show_cell("negative", "negative")
        with monkeypatch.context() as patch:
            patch.setattr(CE, "open_objects", opener)
            assert screen._open_high() == "opened"
        assert calls[0][0] == ["plate1_A01_1_1.png"]
        assert calls[0][1]["context"]["scores"] == {"plate1_A01_1_1.png": 0.9}
        assert calls[0][1]["context"]["which"] == "high"

    with subtests.test(msg="scores that cannot be read still open the crops"):
        calls.clear()
        screen.show_cell("negative", "negative")
        with monkeypatch.context() as patch:
            patch.setattr(CE, "open_objects", opener)
            patch.setattr(CE.cx, "CONFIDENCE_COLUMN", "a_column_nobody_wrote")
            assert screen.open_cell("high") == "opened"
        assert calls[0][0] == ["plate1_A01_1_1.png"]
        assert calls[0][1]["context"]["scores"] == {}

    refusals = [
        # (case id, patch target, attribute, replacement, status, error style)
        ("the cell cannot list its keys", "cell", "keys",
         _raise(CE.cx.ConfusionError("no confidence column to rank by")),
         "no confidence column to rank by", True),
        ("the opener raises", CE, "open_objects",
         _raise(RuntimeError("no crop viewer")),
         "Could not open those crops: no crop viewer", True),
        ("the half is empty", "cell", "keys", lambda self, which: [],
         "That list is empty.", False),
    ]
    for case_id, target, attribute, replacement, status, error in refusals:
        with subtests.test(msg=case_id):
            screen.show_cell("negative", "negative")
            owner = type(screen._cell) if target == "cell" else target
            with monkeypatch.context() as patch:
                patch.setattr(owner, attribute, replacement)
                assert screen.open_cell("high") is None
            assert screen._status.text() == status
            assert screen._status.styleSheet() == (
                _error_style() if error else "")

    with subtests.test(msg="no cell open"):
        screen._clear_cell()
        assert screen.open_cell("high") is None
        assert screen.open_cell("low") is None


def test_fill_list_previews_the_head_and_says_how_many_it_left_out(
        make, evaluation_root, subtests):
    """An empty frame says (none); a long one ends with a count of the rest."""
    screen = make(evaluation_root)
    listing = screen._high_list
    size = CE.LIST_PREVIEW + 7
    cases = [
        ("an empty frame", pd.DataFrame(), ["(none)"]),
        ("no object key column still lists the rows",
         pd.DataFrame({"score": [0.9, 0.8]}), ["    ?    "] * 2),
        ("a key column with confidences",
         pd.DataFrame({"basename": ["a.png", "b.png"],
                       "confidence": [0.91234, float("nan")]}),
         ["0.912  a.png", "    ?    b.png"]),
        ("a long list says how many it did not show",
         pd.DataFrame({"basename": [f"o{i}.png" for i in range(size)]}),
         [f"    ?    o{i}.png" for i in range(CE.LIST_PREVIEW)]
         + ["… 7 more; open them to see all"]),
    ]
    for case_id, frame, expected in cases:
        with subtests.test(msg=case_id):
            screen._fill_list(listing, frame, 0.75)
            assert _texts(listing) == expected


def test_a_threaded_scan_and_load_land_on_the_gui_thread(make,
                                                         evaluation_root,
                                                         qtbot):
    """Discovery and parsing both run off the GUI thread and report back."""
    widget = make(threaded=True)
    widget._source.setText(str(evaluation_root))
    widget.scan()
    qtbot.waitUntil(lambda: widget.bundle is not None and not widget._busy,
                    timeout=10000)
    assert widget.bundles == CE.find_evaluation_bundles(evaluation_root)
    assert _table(widget._confusion) == [["negative", "1", "1"],
                                         ["positive", "1", "1"]]
    assert widget._status.text() == LOADED
    qtbot.waitUntil(lambda: widget._jobs == [], timeout=10000)
    widget.close()
