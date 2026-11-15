"""The QC dashboard's on-disk readers and the screen that caches them.

Every reader in :mod:`spacr.qt.widgets.qc_summary` does a listing, a stat and
a parse -- nothing is recomputed -- so the cases are what is on disk: a file
that will not parse, a database that will not open, a report with a field
missing, and the clean control.  Each is built as a real file under
``tmp_path`` and read by the real reader; every row asserts the exact verdict
and headline the card shows.  The last test drives the dashboard screen's
cache, which must re-read whenever its fingerprint cannot be taken.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from spacr.measurement_schema import MEASUREMENT_STAMP_COLUMNS
from spacr.qt.widgets import qc_summary as qc
from tests.refused_access import refused


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload),
                    encoding="utf-8")
    return path


def _db(path, tables):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        for name, frame in tables.items():
            frame.to_sql(name, db, index=False)
    return path


def test_project_root_reads_whatever_source_a_screen_was_given(tmp_path, subtests):
    """A list is read as its first entry, a file as its folder, nothing as no root."""
    db = tmp_path / "measurements.db"
    db.write_bytes(b"")
    cases = {
        "list-of-folders": ([str(tmp_path), "/elsewhere"], str(tmp_path)),
        "empty-list": ([], ""),
        "none": (None, ""),
        "empty-string": ("", ""),
        "file-stands-for-its-folder": (str(db), str(tmp_path)),
        "path-that-does-not-exist": (str(tmp_path / "gone"), ""),
        "folder": (str(tmp_path), str(tmp_path)),
    }
    for case, (src, expected) in cases.items():
        with subtests.test(msg=case):
            assert qc._project_root(src) == expected


def test_the_leakage_reader_names_the_boundaries_that_leaked(tmp_path, subtests):
    """An empty audit is not a pass, a leak names its boundary, a broken file is an error."""
    cases = {
        "no-reports": ({"reports": []}, "missing",
                       "leakage.json holds no split reports.", []),
        "one-boundary-leaked": (
            {"reports": [{"group_by": "well", "passed": False},
                         {"group_by": "plate", "passed": True}]},
            "fail", "1 of 2 split boundaries leaked (well).", ["upper bound"]),
        "all-held": (
            {"reports": [{"group_by": "well", "passed": True}]}, "ok",
            "All 1 split boundaries held; no related samples crossed a split.",
            []),
    }
    for case, (payload, verdict, headline, detail) in cases.items():
        with subtests.test(msg=case):
            root = tmp_path / case
            path = _write_json(root / "leakage.json", payload)
            card = qc._read_leakage(str(root))
            assert (card.verdict, card.headline) == (verdict, headline)
            assert card.source == str(path)
            assert len(card.detail) == len(detail)
            for fragment, line in zip(detail, card.detail):
                assert fragment in line
    with subtests.test(msg="unparsable"):
        root = tmp_path / "broken"
        _write_json(root / "leakage.json", "{not json")
        card = qc._read_leakage(str(root))
        assert card.verdict == "error"
        assert card.headline.startswith("Could not read leakage.json: ")


def test_the_units_reader_tells_mixed_unstamped_and_consistent_tables_apart(
        tmp_path, subtests):
    """One run per table is ok; two runs in one table is a failure naming it."""
    one_stamp = {name: ["one"] for name in MEASUREMENT_STAMP_COLUMNS}
    two_stamps = {name: ["a", "b"] for name in MEASUREMENT_STAMP_COLUMNS}
    cases = {
        "only-bookkeeping-tables": (
            {"settings": pd.DataFrame({"key": ["a"], "value": ["b"]}),
             "png_list": pd.DataFrame({"png_path": ["/a.png"]})},
            "missing", "The measurements database holds no object tables."),
        "two-runs-in-one-table": (
            {"cell": pd.DataFrame({"cell_area": [1.0, 2.0], **two_stamps})},
            "fail",
            "1 table(s) hold rows measured in different units (cell)."),
        "no-stamp": (
            {"cell": pd.DataFrame({"cell_area": [1.0, 2.0]})}, "warn",
            "1 table(s) carry no units stamp, so a 2-D and a 3-D run cannot be "
            "told apart in them."),
        "stamped-beside-unstamped": (
            {"cell": pd.DataFrame({"cell_area": [1.0], **one_stamp}),
             "notes": pd.DataFrame({"text": ["x"]})},
            "ok", "1 measurement table(s) are internally consistent in units "
                  "and dimensionality."),
    }
    for case, (tables, verdict, headline) in cases.items():
        with subtests.test(msg=case):
            db = _db(tmp_path / case / "measurements" / "measurements.db", tables)
            card = qc._read_units(str(tmp_path / case))
            assert (card.verdict, card.headline) == (verdict, headline)
            assert card.source == str(db)
            if verdict == "fail":
                assert "no query over the table can tell them apart" in \
                    card.detail[0]
    with subtests.test(msg="database-at-the-project-root"):
        db = _db(tmp_path / "flat" / "measurements.db",
                 {"cell": pd.DataFrame({"cell_area": [2.0], **one_stamp})})
        card = qc._read_units(str(tmp_path / "flat"))
        assert (card.source, card.verdict) == (str(db), "ok")
    with subtests.test(msg="will-not-open"):
        db = _db(tmp_path / "locked" / "measurements" / "measurements.db",
                 {"cell": pd.DataFrame({"area": [1.0]})})
        with refused(db, 0o000, windows="read"):
            card = qc._read_units(str(tmp_path / "locked"))
        assert card.verdict == "error"
        assert card.headline.startswith("Could not open measurements.db")


def test_the_plate_reader_carries_the_verdict_and_its_detail(tmp_path, subtests):
    """Detail may be a list or one string; an unparsable file is an error card."""
    cases = {
        "list-detail": (
            {"verdict": "warn", "headline": "Edge wells are dimmer.",
             "detail": ["row A is 30% dimmer", "column 1 too"]},
            "warn", "Edge wells are dimmer.",
            ["row A is 30% dimmer", "column 1 too"]),
        "single-string-detail": ({"detail": "one sentence"}, "missing",
                                 "Plate QC ran.", ["one sentence"]),
    }
    for case, (payload, verdict, headline, detail) in cases.items():
        with subtests.test(msg=case):
            _write_json(tmp_path / case / "plate_qc.json", payload)
            card = qc._read_plate(str(tmp_path / case))
            assert (card.verdict, card.headline, card.detail) == (
                verdict, headline, detail)
    with subtests.test(msg="unparsable"):
        _write_json(tmp_path / "broken" / "plate_qc.json", "{not json")
        card = qc._read_plate(str(tmp_path / "broken"))
        assert card.verdict == "error"
        assert card.headline.startswith("Could not read plate_qc.json: ")


def test_the_agreement_reader_bands_kappa_conventionally(tmp_path, subtests):
    """Kappa >= 0.6 ok, >= 0.4 warn, else fail; no kappa or a word is not a verdict."""
    cases = {
        "no-kappa": ({"band": "substantial"}, "missing",
                     "The agreement report holds no kappa.", 0),
        "kappa-is-a-word": ({"kappa": "high"}, "error",
                            "kappa is not a number: 'high'", 0),
        "kappa-0.81": ({"kappa": 0.81, "band": "measured"}, "ok",
                       "Annotator agreement kappa = 0.81 (measured).", 0),
        "kappa-0.55": ({"kappa": 0.55, "band": "measured"}, "warn",
                       "Annotator agreement kappa = 0.55 (measured).", 1),
        "kappa-0.2": ({"kappa": 0.2, "band": "measured"}, "fail",
                      "Annotator agreement kappa = 0.20 (measured).", 1),
    }
    for case, (payload, verdict, headline, n_detail) in cases.items():
        with subtests.test(msg=case):
            _write_json(tmp_path / case / "agreement.json", payload)
            card = qc._read_agreement(str(tmp_path / case))
            assert (card.verdict, card.headline) == (verdict, headline)
            assert len(card.detail) == n_detail
    with subtests.test(msg="unparsable"):
        _write_json(tmp_path / "broken" / "agreement.json", "{not json")
        card = qc._read_agreement(str(tmp_path / "broken"))
        assert card.verdict == "error"
        assert card.headline.startswith("Could not read agreement.json: ")


def test_the_image_qc_classifier_card_counts_flags_and_lists_the_benchmark(
        tmp_path, monkeypatch, subtests):
    """No usable report gives no card; a scored report lists every class and the benchmark."""
    from spacr.image_quality import REPORT

    scored = {"policy": {"image_qc_classifier_threshold": 0.5},
              "fields": [{"channels": [{"p_debris": 0.9}, {"p_debris": 0.1}]}]}
    for case, body in {"unparsable": "{not json", "no-fields": {"fields": []},
                       "no-probabilities": {
                           "policy": {"image_qc_classifier_threshold": 0.5},
                           "fields": [{"channels": [{"mean": 1.0}]}]}}.items():
        with subtests.test(msg=case):
            _write_json(tmp_path / case / REPORT, body)
            assert qc._read_image_qc_classifier(str(tmp_path / case)) is None
    with subtests.test(msg="no-report"):
        assert qc._read_image_qc_classifier(str(tmp_path / "empty")) is None

    classes = ["out_of_focus: 0 of 2 channels at or above 0.5; mean p 0.00",
               "saturated: 0 of 2 channels at or above 0.5; mean p 0.00",
               "debris: 1 of 2 channels at or above 0.5; mean p 0.50",
               "bubble: 0 of 2 channels at or above 0.5; mean p 0.00",
               "empty: 0 of 2 channels at or above 0.5; mean p 0.00"]
    with subtests.test(msg="benchmark-listed"):
        path = _write_json(tmp_path / "scored" / REPORT, scored)
        pd.DataFrame({"class": ["debris"], "precision": [0.75]}).to_csv(
            path.parent / "image_qc_benchmark.csv", index=False)
        card = qc._read_image_qc_classifier(str(tmp_path / "scored"))
        assert (card.key, card.title, card.verdict, card.source) == (
            "image_qc_classifier", "Image QC classifier", "warn", str(path))
        assert card.headline == "2 channels scored; 1 class flags at threshold 0.5."
        assert card.detail == classes + ["Benchmark class debris, precision 0.75"]
    with subtests.test(msg="benchmark-unreadable"):
        import spacr.tabular as tabular

        def broken(*_a, **_k):
            raise ValueError("bad table")

        monkeypatch.setattr(tabular, "read_table", broken)
        card = qc._read_image_qc_classifier(str(tmp_path / "scored"))
        assert card.detail == classes + ["Could not read the benchmark: bad table"]
        assert card.mtime == path.stat().st_mtime


def test_the_dashboard_screen_rereads_unless_its_fingerprint_proves_nothing_changed(
        qtbot, tmp_path, monkeypatch, subtests):
    """No fingerprint means no cache; a usable one stops the second read; browse sets the source."""
    pytest.importorskip("PySide6")
    from spacr import seg_qc
    from spacr.qt.screens import qc_dashboard
    from spacr.qt.screens.qc_dashboard import QCDashboardScreen

    def screen_with_reader():
        calls = []

        def read(src):
            calls.append(src)
            return qc.Dashboard(root=str(src), verdict="pass",
                                headline="nothing to report")

        screen = QCDashboardScreen(threaded=False, reader=read)
        qtbot.addWidget(screen)
        return screen, calls

    def refuse(_src):
        raise RuntimeError("the qc folder cannot be listed")

    unfingerprintable = {
        "scorecard-scan-fails": refuse,
        "scorecard-vanished": lambda _src: (str(tmp_path / "qc" / "gone.csv"),),
    }
    for case, finder in unfingerprintable.items():
        with subtests.test(msg=case):
            monkeypatch.setattr(seg_qc, "find_scorecards", finder)
            screen, calls = screen_with_reader()
            screen.set_source(str(tmp_path))
            screen.refresh()
            assert screen._fingerprint(str(tmp_path)) is None
            assert calls == [str(tmp_path), str(tmp_path)]
    monkeypatch.undo()

    with subtests.test(msg="fingerprint-honoured"):
        project = tmp_path / "project"
        (project / "measurements").mkdir(parents=True)
        (project / "measurements" / "measurements.db").write_bytes(b"x")
        screen, calls = screen_with_reader()
        screen.set_source(str(project))
        assert screen.refresh() is False
        assert calls == [str(project)]
        assert "nothing on disk has changed" in screen.status_text().lower()

    with subtests.test(msg="empty-read-keeps-the-last-dashboard"):
        screen, _calls = screen_with_reader()
        screen.set_source(str(tmp_path))
        before = screen.as_text()
        screen._on_read(None)
        assert screen.dashboard().headline == "nothing to report"
        assert screen.as_text() == before

    for case, answer, expected_calls in (
            ("browse-to-a-folder", str(tmp_path), [str(tmp_path)]),
            ("browse-cancelled", "", [])):
        with subtests.test(msg=case):
            screen, calls = screen_with_reader()
            start = tmp_path / "start"
            start.mkdir(exist_ok=True)
            if case == "browse-cancelled":
                screen.set_source(str(start))
                calls.clear()
            monkeypatch.setattr(qc_dashboard.QFileDialog, "getExistingDirectory",
                                staticmethod(lambda *a, _v=answer, **k: _v))
            screen._on_browse()
            assert calls == expected_calls
            assert screen.source() == (answer or str(start))
