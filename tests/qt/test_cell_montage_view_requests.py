"""The cell montage: ``load``'s refusals and routes, and the panel's controls.

A montage that draws the wrong cells looks exactly like one that draws the
right cells, so every path where the panel or its worker-thread loader is
handed less than it needs must end in a stated refusal (asserted verbatim), a
disabled control, or a fallback the caption can name -- never an exception.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Tuple

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QMessageBox  # noqa: E402

from spacr.qt.widgets import cell_montage_view as cmv  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def make_view(qtbot):
    """Build montage views that stop their loader when the test ends."""
    made = []

    def build(**kwargs):
        widget = cmv.CellMontageView(threaded=False, **kwargs)
        qtbot.addWidget(widget)
        made.append(widget)
        return widget

    yield build
    for widget in made:
        widget.shutdown()


def _empty_db(path):
    sqlite3.connect(str(path)).close()
    return str(path)


def test_load_refuses_by_naming_what_is_missing(tmp_path, monkeypatch, subtests):
    """No database, no fractions, a folder without regression data, unreadable counts: each is a sentence."""
    from spacr import cell_montage as cm

    database = _empty_db(tmp_path / "measurements.db")
    (tmp_path / "run").mkdir()
    broken = tmp_path / "broken.csv"
    broken.write_text("this is not a counts table\n", encoding="utf-8")
    counts = tmp_path / "counts.csv"
    counts.write_text("prc,grna,count\n", encoding="utf-8")

    def request(**over):
        fields = dict(name="guide_1", effect=0.5, results_path="",
                      count_csvs=(), databases=(database,))
        fields.update(over)
        return cmv.MontageRequest(**fields)

    cases = {
        "no-database": (request(databases=()),
                        "No measurement database is attached to this run's input "
                        "table, so there are no per-object rows and no crops. "
                        "Attach one to a plate row first."),
        "no-guide-fractions": (request(),
                               "No per-well guide fractions are available: there is "
                               "no run folder holding regression_data.csv, and no "
                               "count CSV is attached to the input table. Either "
                               "one is enough."),
        "folder-without-regression-data": (
            request(results_path=str(tmp_path / "run")),
            f"{tmp_path / 'run' / 'regression_data.csv'} does not exist. A montage "
            "needs regression_data.csv from the regression results folder."),
        "unreadable-count-csv": (
            request(count_csvs=(str(broken),)),
            "The per-well guide fractions could not be built from the count CSVs: "
            "broken.csv has no grna or count column; it has ['this is not a counts "
            "table', 'plateID']. They are the regression's own input -- the same "
            "files the fit read -- so a run that produced coefficients has them."),
    }
    for case, (req, error) in cases.items():
        with subtests.test(msg=case):
            answer = cmv.load(req)
            assert (answer.unavailable, answer.error) == (True, error)

    def raising(error):
        def _raise(*_a, _error=error):
            raise _error
        return _raise

    patched = {
        "count-csv-montage-error": (
            "fractions_from_counts", cm.MontageError("the count file names no well"),
            request(count_csvs=(str(counts),)), "the count file names no well"),
        "count-csv-unexpected-error": (
            "fractions_from_counts", MemoryError("the count file is too large"),
            request(count_csvs=(str(counts),)), None),
        "run-folder-montage-error": (
            "read_well_guide_fractions", cm.MontageError("no fractions here"),
            request(results_path="/nowhere/at/all"), "no fractions here"),
        "run-folder-unexpected-error": (
            "read_well_guide_fractions", RuntimeError("the disk went away"),
            request(results_path="/nowhere/at/all"),
            "Could not read the per-well guide fractions: the disk went away"),
    }
    for case, (reader, error, req, expected) in patched.items():
        with subtests.test(msg=case):
            with monkeypatch.context() as patch:
                patch.setattr(cm, reader, raising(error))
                answer = cmv.load(req)
            assert answer.unavailable is True
            if expected is None:
                assert "Could not build the guide fractions" in answer.error
                assert "too large" in answer.error
            else:
                assert answer.error == expected


@dataclass(frozen=True)
class _FakePlan:
    """The shape ``load`` needs back from the selector, and nothing more."""

    objects: Any
    notes: Tuple[str, ...] = ()

    def rows(self):
        return [row for _index, row in self.objects.iterrows()]


class _FakeSource:
    """A crop source that answers with a flat grey square per row."""

    def get_many(self, rows):
        return [np.full((4, 4, 3), 7, dtype=np.uint8) for _ in rows]


def test_load_follows_the_crop_route_and_hands_the_picker_its_inputs(
        tmp_path, monkeypatch, subtests):
    """Route requirements refuse or annotate; attributed/multivariate picking gets matched effects and the grid."""
    from spacr import cell_montage, crops
    from spacr.cell_montage import CropSourceChoice, RouteRequirements

    objects = pd.DataFrame({"object_id": [1, 2], "prc": ["p1_r1_c1", "p1_r1_c2"],
                            "pred": [0.8, 0.2]})
    counts = pd.DataFrame({"prc": ["p1_r1_c1", "p1_r1_c2"],
                           "grna": ["TGGT1_225160_1", "TGGT1_225160_2"],
                           "fraction": [0.9, 0.1]})
    seen: dict = {}
    monkeypatch.setattr(cell_montage, "fractions_from_counts", lambda _p: counts)
    monkeypatch.setattr(cell_montage, "load_montage_objects",
                        lambda *a, **k: objects.copy())
    monkeypatch.setattr(crops, "reanchor_frame", lambda frame, _root: (
        frame, SimpleNamespace(describe=lambda: "")))

    def remember(_objects, _counts, name, effect, **kwargs):
        seen.clear()
        seen.update(kwargs, name=name)
        return _FakePlan(objects=_objects)

    monkeypatch.setattr(cell_montage, "select_montage", remember)

    def route(requirements):
        monkeypatch.setattr(cell_montage, "resolve_montage_crop_source",
                            lambda *a, **k: CropSourceChoice(
                                source=_FakeSource(), kind="merged",
                                reason="merged arrays", available=True,
                                requirements=requirements))

    database = tmp_path / "p1.db"
    database.write_bytes(b"")
    count_csv = tmp_path / "counts.csv"
    count_csv.write_text("prc,grna,count\n", encoding="utf-8")

    def request(picture=None, crop_shape="object"):
        return cmv.MontageRequest(
            name="225160", effect=0.8, level="gene", results_path="",
            databases=(str(database),), count_csvs=(str(count_csv),),
            picture=picture or {}, crop_shape=crop_shape)

    with subtests.test(msg="missing-requirement-refused"):
        route(RouteRequirements(route="merged-bbox", shapes=("bbox",),
                                missing=("no channel list",)))
        answer = cmv.load(request())
        assert answer.unavailable is True
        assert "no channel list" in answer.error
    with subtests.test(msg="uncuttable-shape-stated"):
        route(RouteRequirements(route="merged-bbox", shapes=("bbox",),
                                detail="a coordinate table"))
        answer = cmv.load(request(crop_shape="object"))
        notes = "\n".join(answer.plans[0].notes)
        assert "'object' crop shape was asked for" in notes
        assert "'bbox'" in notes
        assert answer.shapes == ("bbox",)
    with subtests.test(msg="no-requirements-narrow-nothing"):
        route(None)
        answer = cmv.load(request())
        assert len(answer.plans) == 1
        assert (answer.shapes, answer.shape_reason) == ((), "")
    picking = {
        "attributed-matches-the-measured-prefix": (
            "attributed", {"225160_1": 0.8, "225160_2": -0.1}, None,
            {"TGGT1_225160_1": 0.8, "TGGT1_225160_2": -0.1}),
        "no-matching-effects-carry-none": (
            "assigned", {"nothing_matches": 1.0}, None, None),
        "multivariate-gets-the-grid": (
            "multivariate", {"225160_1": 0.8},
            pd.DataFrame({"cell_area": [0.4]}, index=["225160_1"]),
            {"TGGT1_225160_1": 0.8}),
    }
    for case, (mode, effects, grid, expected) in picking.items():
        with subtests.test(msg=case):
            route(None)
            monkeypatch.setattr(cell_montage, "effects_from_results",
                                lambda _path, _e=effects: _e)
            if grid is not None:
                monkeypatch.setattr(cell_montage, "effects_grid_from_results",
                                    lambda _path, _g=grid: _g)
            cmv.load(request(picture={"cell_picking": mode}))
            assert seen["picking"] == mode
            assert seen["effects"] == expected
            if grid is not None:
                assert seen["effects_grid"] is grid


def test_the_panel_reads_count_and_score_files_off_the_input_table(make_view,
                                                                  subtests):
    """A throwing or absent table names no files; repeated files are listed once."""
    def raising():
        raise RuntimeError("the input table is gone")

    rows = [{"plate": "p1", "score": "s1.csv", "count": "c1.csv",
             "database": "p1.db"},
            {"plate": "p2", "score": "s1.csv", "count": "", "database": ""}]
    for case, kwargs, files in (
            ("provider-raises", {"database_provider": raising}, ((), ())),
            ("no-provider", {}, ((), ())),
            ("attached-files", {"database_provider": lambda: rows},
             (("c1.csv",), ("s1.csv",)))):
        with subtests.test(msg=case):
            view = make_view(**kwargs)
            assert (view.count_csvs(), view.score_csvs()) == files


def test_the_panel_says_what_to_do_before_there_is_a_montage(
        make_view, tmp_path, monkeypatch, subtests):
    """Build, compare and save with nothing picked say so; multivariate needs a sweep or a downgrade to rank."""
    with subtests.test(msg="nothing-selected"):
        view = make_view()
        view._announce()
        assert view.status_text() == view.NOTHING_SELECTED == (
            "Click a coefficient — a dot on the volcano or a row in the "
            "coefficient table — to see the cells behind it.")
        assert view.build() is False
        assert view.status_text() == view.NOTHING_SELECTED
    with subtests.test(msg="compare-with-nothing-picked"):
        view = make_view()
        assert view.compare_a_measurement() is None
        assert view.status_text() == (
            "Show some cells first — the comparison groups them by what the "
            "picker chose, and nothing is picked yet.")
    with subtests.test(msg="save-with-no-montage"):
        view = make_view()
        assert view.save() is None
        assert view.status_text() == "There is no montage to save yet."
    with subtests.test(msg="mode-line-without-a-status-label"):
        view = make_view()
        before = view.status_text()
        monkeypatch.setattr(view, "_status", None)
        view._on_mode_changed()
        assert view.status_text() == before
        assert cmv.picture_source_label(view.picture_mode())

    run = tmp_path / "ols_3"
    run.mkdir()
    results = run / "results.csv"
    results.write_text("feature,coefficient\n", encoding="utf-8")
    frame = pd.DataFrame({"feature": ["fraction:gene[GRA14]", "Intercept"],
                          "coefficient": [0.8, 1.0]})
    rows = [{"plate": "p1", "score": str(tmp_path / "scores.csv"),
             "count": str(tmp_path / "counts.csv"),
             "database": _empty_db(tmp_path / "p1.db")}]
    for case, answer, built in (("multivariate-declined", "cancel", False),
                                ("multivariate-downgraded", "rank", True)):
        with subtests.test(msg=case):
            view = make_view(frame_provider=lambda: frame,
                             results_provider=lambda: str(results),
                             database_provider=lambda: rows)
            view.set_coefficient("fraction:gene[GRA14]")
            view._picture_settings = dict(view.picture_settings())
            view._picture_settings["cell_picking"] = "multivariate"
            view._ask_about_multivariate = lambda _shortfall, _a=answer: _a
            assert view.build() is built
            if built:
                assert view._picking_override == "rank"


def test_writing_scores_into_databases_asks_first_and_reports_what_merged(
        make_view, tmp_path, monkeypatch, subtests):
    """Nothing without both halves; the prompt names every file, Cancel is default, closing is no."""
    def answering(button=None, close=False, seen=None):
        def exec_(self):
            if seen is not None:
                seen.update(text=self.text(), detail=self.informativeText(),
                            default=self.defaultButton().text())
            if close:
                self.reject()
            else:
                self.buttons()[button].click()
            return 0
        return exec_

    with subtests.test(msg="nothing-to-merge"):
        view = make_view()
        assert view.write_scores_into_the_databases() == {}
        assert view.status_text() == ("Nothing to merge: attach at least one "
                                      "database and load at least one "
                                      "per-object score file.")
    with subtests.test(msg="prompt-names-every-file-cancel-default"):
        view = make_view()
        seen = {}
        monkeypatch.setattr(QMessageBox, "exec", answering(1, seen=seen))
        assert view._ask_before_writing(["/data/p1.db"],
                                        ["/data/scores.csv"]) is False
        assert "1 measurement database?" in seen["text"]
        assert "scores.csv" in seen["detail"] and "p1.db" in seen["detail"]
        assert seen["default"] == "Cancel"
    for case, exec_, answer in (("closed-is-not-consent", answering(close=True), False),
                                ("write-pressed", answering(0), True)):
        with subtests.test(msg=case):
            view = make_view()
            monkeypatch.setattr(QMessageBox, "exec", exec_)
            assert view._ask_before_writing(["/data/p1.db"], ["/a.csv"]) is answer

    scores = tmp_path / "plate1_dv.csv"
    pd.DataFrame({"path": [f"img{i}.png" for i in range(4)],
                  "pred": [i / 4 for i in range(4)],
                  "cv_predictions": [0, 1, 0, 1],
                  "prc": ["plate1_r1_c1"] * 4,
                  "object": list(range(4))}).to_csv(scores, index=False)
    with subtests.test(msg="default-confirmation-asks"):
        asked = []

        def refuse(self):
            asked.append(self.text())
            self.buttons()[1].click()
            return 0

        monkeypatch.setattr(QMessageBox, "exec", refuse)
        view = make_view(database_provider=lambda: [
            {"plate": "p1", "score": str(scores),
             "database": _empty_db(tmp_path / "asked.db")}])
        assert view.write_scores_into_the_databases() == {}
        assert len(asked) == 1
    with subtests.test(msg="no-database-takes-the-scores"):
        broken = tmp_path / "not-a-database.db"
        broken.write_text("this is not sqlite", encoding="utf-8")
        table = [{"plate": "plate1", "database": str(broken), "score": str(scores)}]
        view = make_view(database_provider=lambda: table)
        assert view.write_scores_into_the_databases(confirm=lambda *_a: True) == {}
        assert "not-a-database.db" in view.status_text()
        assert "matched" not in view.status_text()
        database = tmp_path / "measurements.db"
        with sqlite3.connect(database) as db:
            db.execute("CREATE TABLE png_list (file_name, plateID, rowID, "
                       "columnID, fieldID, object_label, png_path)")
            for i in range(4):
                db.execute("INSERT INTO png_list VALUES (?,?,?,?,?,?,?)",
                           (f"img{i}.png", "plate1", "r1", "c1", "1", i,
                            str(tmp_path / f"img{i}.png")))
        table[:] = [{"plate": "plate1", "database": str(database),
                     "score": str(scores)}]
        written = view.write_scores_into_the_databases(confirm=lambda *_a: True)
        assert list(written) == [str(database)]
        assert "rows matched" in view.status_text()


def test_settings_written_back_to_the_hidden_controls_keep_what_they_cannot_hold(
        make_view, qtbot, subtests):
    """Nulls and unholdable values are skipped, combos match by text, the cap box spans the editor's range."""
    from spacr.qt.widgets.picture_settings_dialog import PictureSettingsDialog

    with subtests.test(msg="unholdable-value-skipped"):
        view = make_view()
        cap = view._read_widgets()["cap"]
        view._write_back({"channels": [1, 2], "cap": "not a number"})
        assert view._channels.text() == "1, 2"
        assert view._read_widgets()["cap"] == cap == 2000
    with subtests.test(msg="null-not-written"):
        view = make_view()
        view._channels.setText("r, g, b")
        view._write_back({"channels": None})
        assert view._channels.text() == "r, g, b"
    with subtests.test(msg="combo-matched-by-text"):
        view = make_view()
        view._object.addItem("organelle")
        view._write_back({"object_type": "organelle"})
        assert view._object.currentText() == "organelle"
    with subtests.test(msg="cap-box-holds-every-offered-cap"):
        view = make_view()
        dialog = PictureSettingsDialog(view.picture_settings())
        qtbot.addWidget(dialog)
        offered = dialog._editors["cap"]
        assert (view._cap.maximum(), view._cap.minimum()) == (offered.maximum(), 1)
        offered.setValue(20000)
        view._write_back({"cap": offered.value()})
        assert view._cap.value() == view._read_widgets()["cap"] == 20000
        assert 20000 in view._load_signature()


def test_the_shape_control_follows_what_the_crop_route_can_cut(make_view, subtests):
    """An uncuttable shape moves the control off it; no shape at all greys it with the reason."""
    from PySide6.QtCore import QStringListModel

    with subtests.test(msg="moves-off-an-uncuttable-shape"):
        view = make_view()
        view._shape.setCurrentIndex(view._shape.findData("mask"))
        view._apply_shape_availability(cmv.MontageLoad(
            plans=(object(),), shapes=("bbox",),
            shape_reason="this route has no masks"))
        assert (str(view._shape.currentData()), view._shape.isEnabled()) == (
            "bbox", True)
    with subtests.test(msg="no-shape-disables"):
        view = make_view()
        view._apply_shape_availability(cmv.MontageLoad(
            plans=(object(),), shapes=(),
            shape_reason="the PNGs were cut at run time"))
        assert view._shape.isEnabled() is False
        assert "cut at run time" in view._shape.toolTip()
    with subtests.test(msg="combo-without-an-item-model"):
        view = make_view()
        view._shape.setModel(QStringListModel(["object", "bbox"]))
        view._apply_shape_availability(cmv.MontageLoad(
            plans=(object(),), shapes=("bbox",),
            shape_reason="this route has no masks"))
        assert view._shape.isEnabled() is True
