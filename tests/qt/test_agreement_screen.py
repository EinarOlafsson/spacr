"""Annotator Agreement — the Tools module that scores annotation columns.

Everything runs offscreen against a *real* temporary ``measurements.db``
whose ``png_list`` carries two or three annotation columns, written the
way ``spacr.qt.annotate_engine`` writes them: an INTEGER column per pass,
NULL where the annotator has not looked yet.

The properties this suite pins:

* it is **registered** as a Tools app with a title, an intro and a real icon;
* the κ table and confusion matrix **populate from a real database**,
  with the numbers the arithmetic says they should have;
* the **disagreement review list** contains exactly the rows the
  annotators labelled differently — abstentions are not disagreements;
* a database with only one annotation column, or none, **reports inline**
  instead of crashing, and no code path ever opens a modal dialog (which
  would hang a headless run forever — see ``test_db_browser.py``).
"""
from __future__ import annotations

import hashlib
import math
import os
import sqlite3

import pytest

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QLabel

from spacr.qt.screens.agreement import (
    DEFAULT_REVIEW_LIMIT,
    AgreementScreen,
    format_kappa,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_META = ("png_path", "file_name", "plateID", "rowID", "columnID", "fieldID",
         "prcfo", "cell_id")


class _Db:
    """Handle on the temporary run folder + what went into the database."""

    def __init__(self, src, path, crops, rows):
        self.src = str(src)      # run folder the user types
        self.path = str(path)    # <src>/measurements/measurements.db
        self.crops = crops       # png_path per row
        self.rows = rows         # label tuples per row

    def digest(self):
        with open(self.path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    def siblings(self):
        d = os.path.dirname(self.path)
        return sorted(os.listdir(d))


def build_db(tmp_path, annotators, rows, write_crops=True, name="run"):
    """Create ``<src>/measurements/measurements.db`` + the crop PNGs.

    :param annotators: annotation column names.
    :param rows: per-row label tuples; ``None`` = that annotator abstained.
    :param write_crops: also write a real 8×8 PNG per row, so the review
        panel has something to render.
    """
    src = tmp_path / name
    meas = src / "measurements"
    pngs = src / "cell_png"
    meas.mkdir(parents=True)
    pngs.mkdir(parents=True)
    db_path = meas / "measurements.db"

    crops = []
    for i in range(len(rows)):
        crop = pngs / f"plate1_A01_1_{i}.png"
        if write_crops:
            image = QImage(8, 8, QImage.Format_RGB32)
            image.fill(0xFF3366AA)
            assert image.save(str(crop), "PNG")
        crops.append(str(crop))

    con = sqlite3.connect(db_path)
    try:
        meta_sql = ", ".join(f'"{c}" TEXT' for c in _META)
        con.execute(f"CREATE TABLE png_list ({meta_sql})")
        for col in annotators:
            con.execute(f'ALTER TABLE png_list ADD COLUMN "{col}" INTEGER')
        placeholders = ", ".join("?" * (len(_META) + len(annotators)))
        con.executemany(
            f"INSERT INTO png_list VALUES ({placeholders})",
            [(crops[i], os.path.basename(crops[i]), "plate1", "r1", "c1",
              "f1", f"plate1_A01_1_o{i}", f"o{i}", *labels)
             for i, labels in enumerate(rows)])
        con.commit()
    finally:
        con.close()
    return _Db(src, db_path, crops, rows)


#: 10 rows: 6 unanimous, 2 disagreements, 1 partial (agreeing), 1 untouched.
TWO_ROWS = [
    (1, 1),
    (1, 1),
    (1, 2),      # disagreement  -> row 2
    (2, 2),
    (2, 1),      # disagreement  -> row 4
    (1, None),   # abstention, not a disagreement
    (None, None),
    (2, 2),
    (1, 1),
    (2, 2),
]


@pytest.fixture
def two_db(tmp_path):
    """alice vs bob: 8 shared rows, 6 agreeing -> κ = 0.500 exactly.

        p_o = 6/8                                   = 0.750
        alice marginals = 4/8, 4/8                  (1, 2)
        bob   marginals = 4/8, 4/8
        p_e = 0.5*0.5 + 0.5*0.5                     = 0.500
        κ   = (0.75 - 0.50) / (1 - 0.50)            = 0.500
    """
    return build_db(tmp_path, ["alice", "bob"], TWO_ROWS)


@pytest.fixture
def three_db(tmp_path):
    """Three annotators — the screen must switch to Fleiss' κ."""
    rows = [
        (1, 1, 1),
        (1, 1, 2),
        (2, 2, 2),
        (1, 2, 2),
        (1, 1, None),
        (None, None, None),
        (2, 1, None),
    ]
    return build_db(tmp_path, ["alice", "bob", "carol"], rows, name="run3")


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Blow up loudly if any code path under test opens a modal dialog.

    Copied from ``tests/qt/test_db_browser.py``: a QMessageBox in an error
    path hangs the whole headless suite, and this screen has a lot of
    error paths (no database, one column, missing crop).
    """
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError(
            "a modal dialog was opened — errors must be reported inline")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("exec", "exec_", "open", "show"):
        monkeypatch.setattr(QMessageBox, name, _boom, raising=False)
    monkeypatch.setattr(QDialog, "exec", _boom, raising=False)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    yield


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A synchronous screen — the report is computed inline."""
    w = AgreementScreen(threaded=False)
    qtbot.addWidget(w)
    return w


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_agreement_is_registered_under_results_and_qc_as_alpha():
    """Both axes, which are no longer the same kind of thing.

    #16i briefly made "Alpha modules" a SECTION and filed this screen
    there; #16j put it back under Results & QC — scoring annotation
    passes is reading a result — and kept alpha as a maturity stage that
    the Home tile draws as a hover colour."""
    from spacr.qt.app import APPS
    entry = next((a for a in APPS if a[0] == "agreement"), None)
    assert entry is not None, "agreement missing from APPS"
    key, name, desc, section = entry
    assert name == "Annotator Agreement"
    assert desc and desc.strip()
    from spacr.qt.app import SECTION_RESULTS, app_stage
    assert section == SECTION_RESULTS, (
        f"agreement filed under {section!r}; scoring annotation passes "
        "is reading a result")
    # `spacr.qt.maturity` reassessed every alpha module against the
    # evidence in the repository and this one no longer qualifies; the
    # reason is recorded beside the decision. Applied here because the
    # promotions land in `register_self_registering_modules`, which every
    # launch calls but a bare test process may not have. `apply` alone,
    # not the whole registration pass: it touches only APP_STAGE, so it
    # cannot re-register a module a test has deliberately removed.
    from spacr.qt import maturity
    maturity.apply()
    assert app_stage(key) == "stable", (
        "130 tests and 431 assertions, a shipped lesson and a README "
        "section: the tile colour has to say that")


def test_agreement_has_a_title_and_an_intro():
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES
    assert APP_TITLES.get("agreement", "").strip() == "Annotator Agreement"
    intro = APP_INTROS.get("agreement", "")
    assert len(intro.split()) >= 10, f"intro is too thin: {intro!r}"
    assert intro.endswith(".")


def test_agreement_icon_resolves_to_a_real_resource_file():
    from spacr.qt import app as qt_app
    assert "agreement" in qt_app._ICON_OVERRIDES, (
        "agreement needs an _ICON_OVERRIDES entry — no binary was added")
    here = os.path.dirname(os.path.abspath(qt_app.__file__))
    path = os.path.normpath(os.path.join(
        here, "..", "resources", "icons", qt_app._ICON_OVERRIDES["agreement"]))
    assert os.path.isfile(path), f"missing icon file: {path}"


def test_icon_provider_returns_a_non_null_icon(qtbot, qt_theme_applied):
    from PySide6.QtGui import QIcon
    from spacr.qt.app import _icon_for_app
    icon = _icon_for_app("agreement")
    assert isinstance(icon, QIcon)
    assert not icon.isNull(), "agreement icon is null — the PNG failed to load"


def test_sidebar_lists_annotator_agreement(qtbot, qt_theme_applied):
    from PySide6.QtWidgets import QPushButton
    from spacr.qt.app import Sidebar
    bar = Sidebar()
    qtbot.addWidget(bar)
    labels = {b.accessibleName() for b in bar.findChildren(QPushButton)}
    assert "Annotator Agreement" in labels


def test_main_window_builds_the_agreement_screen(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    win._on_nav_selected("agreement")
    assert isinstance(win._stack.currentWidget(), AgreementScreen)


def test_screen_builds_offscreen_without_raising(qtbot, qt_theme_applied):
    w = AgreementScreen()
    qtbot.addWidget(w)
    assert w.database_path() == ""
    assert w.available_columns() == []
    assert w.report() is None
    assert not w._btn_compute.isEnabled()
    assert any("read-only" in lbl.text().lower()
               for lbl in w.findChildren(QLabel)), \
        "the screen must tell the user it cannot modify annotations"


# ---------------------------------------------------------------------------
# Opening a database + picking columns
# ---------------------------------------------------------------------------

def test_opening_a_run_folder_lists_the_annotation_columns(screen, two_db):
    assert screen.set_database(two_db.src) is True
    assert screen.database_path() == os.path.abspath(two_db.path)
    assert screen.available_columns() == ["alice", "bob"]
    # Two columns is the common case: both ticked so Compute works at once.
    assert screen.selected_columns() == ["alice", "bob"]
    assert screen._btn_compute.isEnabled()
    assert screen.last_error == ""


def test_opening_the_database_file_directly_works(screen, two_db):
    assert screen.set_database(two_db.path) is True
    assert screen.available_columns() == ["alice", "bob"]


def test_database_opened_signal_carries_the_resolved_path(qtbot, screen, two_db):
    with qtbot.waitSignal(screen.database_opened, timeout=2000) as blocker:
        screen.set_database(two_db.src)
    assert blocker.args[0] == os.path.abspath(two_db.path)


def test_typing_a_path_and_pressing_return_opens_it(screen, two_db):
    screen._path_edit.setText(two_db.src)
    screen._path_edit.returnPressed.emit()
    assert screen.database_path() == os.path.abspath(two_db.path)


def test_the_file_pickers_feed_set_database(screen, two_db, monkeypatch):
    """The pickers are the only QFileDialog use — and they are not modal
    message boxes, so they get a real (stubbed) round trip here."""
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (two_db.path, "")))
    screen._pick_database()
    assert screen.database_path() == os.path.abspath(two_db.path)

    screen.set_database("")     # reset
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: two_db.src))
    screen._pick_run_folder()
    assert screen.database_path() == os.path.abspath(two_db.path)


def test_cancelling_a_file_picker_changes_nothing(screen, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    screen._pick_database()
    screen._pick_run_folder()
    assert screen.database_path() == ""
    assert screen.last_error == ""


def test_metadata_columns_are_not_offered_as_annotators(screen, three_db):
    screen.set_database(three_db.src)
    assert screen.available_columns() == ["alice", "bob", "carol"]
    for meta in ("png_path", "plateID", "prcfo", "cell_id"):
        assert meta not in screen.available_columns()


def test_selecting_columns_by_name(screen, three_db):
    screen.set_database(three_db.src)
    assert screen.select_columns(["alice", "carol"]) is True
    assert screen.selected_columns() == ["alice", "carol"]


def test_selecting_an_unknown_column_reports_inline(screen, three_db):
    screen.set_database(three_db.src)
    assert screen.select_columns(["alice", "nobody"]) is False
    assert "nobody" in screen.status_text()
    assert screen.last_error


# ---------------------------------------------------------------------------
# The kappa table + confusion matrix
# ---------------------------------------------------------------------------

def test_kappa_table_populates_from_a_real_database(screen, two_db):
    """8 shared rows, 6 agreeing, balanced marginals -> κ = +0.500."""
    screen.set_database(two_db.src)
    assert screen.compute() is True
    rows = screen.kappa_rows()
    assert len(rows) == 1
    a, b, n, abstain, raw, kappa, band = rows[0]
    assert (a, b) == ("alice", "bob")
    assert n == "8"
    assert abstain == "1"          # the (1, NULL) row, not a disagreement
    assert raw == "75.0%"
    assert kappa == "+0.500"
    assert band == "moderate"
    report = screen.report()
    assert report.overall_method == "Cohen's κ"
    assert report.overall_kappa == pytest.approx(0.5)


def test_confusion_matrix_populates(screen, two_db):
    """[[3, 1], [1, 3]] over the 8 rows both annotators labelled."""
    screen.set_database(two_db.src)
    screen.compute()
    grid = screen.confusion_rows()
    assert grid == [["1", "3", "1"], ["2", "1", "3"]]
    header = screen._confusion_table.horizontalHeaderItem(0).text()
    assert "alice" in header and "bob" in header


def test_summary_states_both_kappa_and_raw_agreement(screen, two_db):
    screen.set_database(two_db.src)
    screen.compute()
    summary = screen.summary_text()
    assert "+0.500" in summary
    assert "75.0%" in summary
    assert "Landis" in summary, "the band must be labelled as a convention"
    assert "8" in summary
    status = screen.status_text()
    assert "Cohen" in status and "+0.500" in status


def test_three_annotators_switch_to_fleiss_and_three_pairs(screen, three_db):
    screen.set_database(three_db.src)
    assert screen.select_columns(["alice", "bob", "carol"]) is True
    assert screen.compute() is True
    assert len(screen.kappa_rows()) == 3
    report = screen.report()
    assert report.overall_method == "Fleiss' κ"
    assert report.overall_kappa == pytest.approx(1 / 3)
    assert "+0.333" in screen.summary_text()
    # Each pair gets its own confusion matrix, selectable from the combo.
    assert screen._pair_combo.count() == 3
    screen._pair_combo.setCurrentIndex(2)
    assert screen.confusion_rows()


def test_clicking_a_kappa_row_shows_that_pairs_confusion(screen, three_db):
    screen.set_database(three_db.src)
    screen.select_columns(["alice", "bob", "carol"])
    screen.compute()
    screen._kappa_table.setCurrentCell(2, 0)
    assert screen._pair_combo.currentIndex() == 2
    header = screen._confusion_table.horizontalHeaderItem(0).text()
    assert "bob" in header and "carol" in header


def test_reselecting_the_same_kappa_row_redraws_the_same_confusion(
        screen, two_db):
    screen.set_database(two_db.src)
    screen.compute()
    before = screen.confusion_rows()
    screen._on_pair_row_changed(0)      # combo already on 0 — the else branch
    assert screen.confusion_rows() == before


def test_confusion_and_pair_selection_are_inert_before_a_report(screen):
    screen._on_pair_row_changed(0)
    screen._show_confusion(3)
    assert screen.confusion_rows() == []
    assert screen.last_error == ""


def test_a_screen_full_of_negatives_reports_undefined_not_perfect(
        screen, tmp_path):
    """Both annotators called every crop class 1 — κ has no denominator."""
    db = build_db(tmp_path, ["alice", "bob"], [(1, 1)] * 12, name="allneg")
    screen.set_database(db.src)
    assert screen.compute() is True
    assert screen.kappa_rows()[0][5] == "undefined"
    assert screen.kappa_rows()[0][4] == "100.0%"
    assert math.isnan(screen.report().overall_kappa)
    assert "undefined" in screen.summary_text()
    assert screen.disagreement_rows() == []


def test_prevalence_paradox_is_visible_on_screen(screen, tmp_path):
    rows = [(1, 1)] * 95 + [(1, 2)] * 3 + [(2, 1)] * 2
    db = build_db(tmp_path, ["alice", "bob"], rows,
                  write_crops=False, name="lopsided")
    screen.set_database(db.src)
    assert screen.compute() is True
    assert screen.kappa_rows()[0][4] == "95.0%"
    assert screen.kappa_rows()[0][5] == "-0.025"
    summary = screen.summary_text()
    assert "95.0%" in summary and "-0.025" in summary
    assert "prevalence paradox" in summary.lower()


# ---------------------------------------------------------------------------
# Disagreement review
# ---------------------------------------------------------------------------

def test_review_list_shows_exactly_the_disagreeing_rows(screen, two_db):
    screen.set_database(two_db.src)
    screen.compute()
    rows = screen.disagreement_rows()
    assert len(rows) == 2, "the (1, NULL) row is an abstention, not a dispute"
    assert [os.path.basename(r[0]) for r in rows] == [
        "plate1_A01_1_2.png", "plate1_A01_1_4.png"]
    assert [r[1:] for r in rows] == [["1", "2"], ["2", "1"]]
    headers = [screen._review_table.horizontalHeaderItem(i).text()
               for i in range(screen._review_table.columnCount())]
    assert headers == ["png_path", "alice", "bob"]
    assert "2" in screen._review_label.text()


def test_review_list_renders_an_abstention_as_a_dash(screen, three_db):
    screen.set_database(three_db.src)
    screen.select_columns(["alice", "bob", "carol"])
    screen.compute()
    rows = screen.disagreement_rows()
    assert len(rows) == 3
    assert rows[2][1:] == ["2", "1", "—"], \
        "carol abstained on that row — it must not read as a label"


def test_selecting_a_disagreement_shows_the_crop(screen, two_db):
    screen.set_database(two_db.src)
    screen.compute()
    # The first row is selected automatically so the reviewer sees a crop.
    assert screen.current_crop_path().endswith("plate1_A01_1_2.png")
    assert not screen._crop_label.pixmap().isNull()
    assert screen.crop_message() == ""
    assert "alice=1" in screen._crop_caption.text()
    assert "bob=2" in screen._crop_caption.text()

    assert screen.select_disagreement(1) is True
    assert screen.current_crop_path().endswith("plate1_A01_1_4.png")
    assert "alice=2" in screen._crop_caption.text()


def test_a_missing_crop_is_reported_in_the_panel_not_raised(screen, two_db):
    os.remove(two_db.crops[2])
    screen.set_database(two_db.src)
    screen.compute()
    assert screen.select_disagreement(0) is False
    assert "not found" in screen.crop_message().lower()
    assert screen.current_crop_path().endswith("plate1_A01_1_2.png")
    # The screen is still usable — the next crop still renders.
    assert screen.select_disagreement(1) is True


def test_an_unreadable_crop_is_reported_in_the_panel(screen, two_db):
    with open(two_db.crops[2], "wb") as fh:
        fh.write(b"not a png at all")
    screen.set_database(two_db.src)
    screen.compute()
    assert screen.select_disagreement(0) is False
    assert "could not read" in screen.crop_message().lower()


def test_a_relative_crop_path_resolves_against_the_run_folder(
        screen, tmp_path):
    """Datasets get copied; png_list keeps the original absolute paths.

    A relative path is tried against ``<src>``, the database's grandparent.
    """
    db = build_db(tmp_path, ["alice", "bob"], [(1, 2), (1, 1)], name="rel")
    con = sqlite3.connect(db.path)
    con.execute("UPDATE png_list SET png_path = ? WHERE png_path = ?",
                ("cell_png/plate1_A01_1_0.png", db.crops[0]))
    con.commit()
    con.close()
    screen.set_database(db.src)
    screen.compute()
    assert screen.select_disagreement(0) is True
    assert not screen._crop_label.pixmap().isNull()


def test_selecting_a_row_that_is_not_there_is_a_no_op(screen, two_db):
    screen.set_database(two_db.src)
    screen.compute()
    assert screen.select_disagreement(99) is False
    assert screen.select_disagreement(-1) is False
    assert screen.last_error == ""


def test_a_row_with_no_crop_path_at_all_is_survivable(screen, tmp_path):
    db = build_db(tmp_path, ["alice", "bob"], [(1, 2), (2, 2)],
                  write_crops=False, name="nullpath")
    con = sqlite3.connect(db.path)
    con.execute("UPDATE png_list SET png_path = NULL WHERE alice = 1")
    con.commit()
    con.close()
    screen.set_database(db.src)
    screen.compute()
    assert screen.select_disagreement(0) is False
    assert "not found" in screen.crop_message().lower()


def test_disagreement_paths_lists_the_crops_to_review(screen, two_db):
    screen.set_database(two_db.src)
    screen.compute()
    paths = screen.disagreement_paths()
    assert [os.path.basename(p) for p in paths] == [
        "plate1_A01_1_2.png", "plate1_A01_1_4.png"]


def test_review_limit_caps_the_queue_and_says_so(screen, tmp_path):
    rows = [(1, 2)] * 40 + [(1, 1)] * 5
    db = build_db(tmp_path, ["alice", "bob"], rows,
                  write_crops=False, name="many")
    screen.set_database(db.src)
    assert screen._limit_box.value() == DEFAULT_REVIEW_LIMIT
    screen._limit_box.setValue(10)
    screen.compute()
    assert len(screen.disagreement_rows()) == 10
    assert screen.report().n_disagreements == 40
    assert "showing the first 10" in screen._review_label.text()


def test_no_disagreements_says_so_instead_of_showing_an_empty_panel(
        screen, tmp_path):
    db = build_db(tmp_path, ["alice", "bob"],
                  [(1, 1), (2, 2), (1, 1), (2, 2)], name="agreeing")
    screen.set_database(db.src)
    screen.compute()
    assert screen.disagreement_rows() == []
    assert "no disagreements" in screen.crop_message().lower()


# ---------------------------------------------------------------------------
# Degenerate databases — inline, never a crash and never a dialog
# ---------------------------------------------------------------------------

def test_one_annotation_column_reports_inline_instead_of_crashing(
        screen, tmp_path):
    db = build_db(tmp_path, ["alice"], [(1,), (2,), (1,), (None,)],
                  write_crops=False, name="lonely")
    assert screen.set_database(db.src) is True     # the file opened fine…
    assert screen.available_columns() == ["alice"]
    assert screen.last_error, "one column cannot produce agreement"
    assert "at least two" in screen.status_text()
    assert not screen._btn_compute.isEnabled()
    # …and asking anyway is a message, not an exception.
    assert screen.compute() is False
    assert "at least two" in screen.status_text()
    assert screen.report() is None
    assert screen.kappa_rows() == []


def test_no_annotation_columns_at_all_reports_inline(screen, tmp_path):
    src = tmp_path / "unlabelled"
    (src / "measurements").mkdir(parents=True)
    path = src / "measurements" / "measurements.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE png_list (png_path TEXT, plateID TEXT)")
    con.execute("INSERT INTO png_list VALUES ('/x/a.png', 'plate1')")
    con.commit()
    con.close()
    assert screen.set_database(str(src)) is False
    assert "annotation column" in screen.status_text()
    assert screen.last_error


def test_a_database_without_png_list_reports_inline(screen, tmp_path):
    path = tmp_path / "nopng.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE cell (prc TEXT)")
    con.commit()
    con.close()
    assert screen.set_database(str(path)) is False
    assert "png_list" in screen.status_text()


def test_a_file_that_is_not_a_database_reports_inline(screen, tmp_path):
    junk = tmp_path / "notadb.db"
    junk.write_text("plate,well,value\n1,A01,3\n" * 50)
    assert screen.set_database(str(junk)) is False
    assert screen.last_error
    assert screen.available_columns() == []


def test_missing_path_and_empty_path_report_inline(screen, tmp_path):
    assert screen.set_database("") is False
    assert screen.last_error
    assert screen.set_database(str(tmp_path / "gone")) is False
    assert screen.last_error


def test_compute_without_a_database_reports_inline(screen):
    assert screen.compute() is False
    assert "Open a measurements database" in screen.status_text()
    assert screen.report() is None


def test_untickng_down_to_one_column_reports_inline(screen, three_db):
    screen.set_database(three_db.src)
    screen.select_columns(["alice"])
    assert screen.compute() is False
    assert "at least two" in screen.status_text()
    assert screen.report() is None


def test_results_are_cleared_when_another_database_is_opened(screen, two_db,
                                                             tmp_path):
    screen.set_database(two_db.src)
    screen.compute()
    assert screen.kappa_rows()
    other = build_db(tmp_path, ["x", "y"], [(1, 1), (1, 2)],
                     write_crops=False, name="other")
    screen.set_database(other.src)
    assert screen.kappa_rows() == []
    assert screen.disagreement_rows() == []
    assert screen.report() is None
    assert screen.summary_text() == ""


def test_a_worker_failure_lands_in_the_status_label(screen, two_db,
                                                    monkeypatch):
    """Synchronous path: an exception inside the job becomes inline text."""
    import spacr.qt.screens.agreement as mod

    def _boom(*_a, **_k):
        raise RuntimeError("synthetic failure")

    screen.set_database(two_db.src)
    monkeypatch.setattr(mod.agree, "agreement_report", _boom)
    assert screen.compute() is False
    assert "synthetic failure" in screen.status_text()
    assert screen.last_error
    assert screen.report() is None


# ---------------------------------------------------------------------------
# Read-only
# ---------------------------------------------------------------------------

def test_a_full_session_leaves_the_database_byte_identical(screen, two_db):
    before, siblings = two_db.digest(), two_db.siblings()
    screen.set_database(two_db.src)
    screen.compute()
    screen.select_disagreement(1)
    screen._pair_combo.setCurrentIndex(0)
    assert two_db.digest() == before, "the database changed on disk"
    assert two_db.siblings() == siblings, "a -wal/-journal side file appeared"


# ---------------------------------------------------------------------------
# Off-thread execution
# ---------------------------------------------------------------------------

def test_report_runs_off_the_gui_thread(qtbot, qt_theme_applied, two_db,
                                         monkeypatch):
    import threading

    import spacr.qt.screens.agreement as mod

    gui_thread = threading.get_ident()
    seen = []
    real = mod.agree.agreement_report

    def _spy(*a, **k):
        seen.append(threading.get_ident())
        return real(*a, **k)

    monkeypatch.setattr(mod.agree, "agreement_report", _spy)

    w = AgreementScreen(threaded=True)
    qtbot.addWidget(w)
    w.set_database(two_db.src)
    with qtbot.waitSignal(w.job_finished, timeout=10000) as blocker:
        w.compute()
    assert blocker.args[0] is True
    assert seen, "the report never ran"
    assert all(t != gui_thread for t in seen), \
        "agreement ran on the GUI thread — the window would freeze"
    assert w.kappa_rows()[0][5] == "+0.500"
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=10000)
    assert w._thread is None and w._worker is None
    w.close()          # must not abort on a live QThread


def test_threaded_failure_reports_inline(qtbot, qt_theme_applied, two_db,
                                          monkeypatch):
    import spacr.qt.screens.agreement as mod

    w = AgreementScreen(threaded=True)
    qtbot.addWidget(w)
    w.set_database(two_db.src)

    def _boom(*_a, **_k):
        raise RuntimeError("synthetic thread failure")

    monkeypatch.setattr(mod.agree, "agreement_report", _boom)
    with qtbot.waitSignal(w.job_finished, timeout=10000) as blocker:
        w.compute()
    assert blocker.args[0] is False
    assert w.last_error
    assert "failed" in w.status_text().lower()
    assert w.report() is None
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=10000)


def test_a_second_compute_while_busy_is_refused_inline(screen, two_db):
    screen.set_database(two_db.src)
    screen._busy = True
    assert screen.is_busy() is True
    assert screen.compute() is False
    assert "already running" in screen.status_text()
    screen._busy = False
    assert screen.is_busy() is False


def test_closing_waits_for_a_live_thread_instead_of_aborting(screen):
    """A QThread garbage-collected while running takes the process with it.

    Real threads finish too fast to close on top of deterministically, so
    the shutdown contract is checked against stand-ins: a running job must
    be asked to quit and waited for, and a job whose C++ side has already
    gone (RuntimeError) must not stop the widget closing.
    """
    class _Live:
        def __init__(self):
            self.quit_called = self.waited = False

        def isRunning(self):
            return True

        def quit(self):
            self.quit_called = True

        def wait(self, _ms):
            self.waited = True
            return True

    class _Idle(_Live):
        def isRunning(self):
            return False

    class _Dead:
        def isRunning(self):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    live, idle, dead = _Live(), _Idle(), _Dead()
    screen._jobs = [(live, None), (idle, None), (dead, None)]
    screen.close()
    assert live.quit_called and live.waited
    assert not idle.quit_called
    screen._jobs = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_format_kappa_never_invents_a_number():
    assert format_kappa(0.5) == "+0.500"
    assert format_kappa(-0.0246) == "-0.025"
    assert format_kappa(float("nan")) == "undefined"
    assert format_kappa(None) == "undefined"
    assert format_kappa("wat") == "undefined"


def test_percent_formatter_says_not_applicable_rather_than_zero():
    from spacr.qt.screens.agreement import _format_pct

    assert _format_pct(0.75) == "75.0%"
    assert _format_pct(float("nan")) == "n/a"
    assert _format_pct(None) == "n/a"
    assert _format_pct("wat") == "n/a"
