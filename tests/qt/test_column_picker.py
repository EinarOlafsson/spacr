"""The SQL column picker — the button that stops a mistyped column name.

Every one of these runs offscreen against a *real* temporary
``measurements.db`` (a ``png_list`` annotation table plus a ``cell``
feature table), because the whole value of the widget is that it tells
the truth about a database that actually exists.

The properties pinned here:

* ``attach_column_picker`` adopts a field a screen has **already laid
  out** — the form row keeps its label, the field keeps its identity, and
  the host keeps its references;
* the dialog lists the **real** tables and columns, and picking one fills
  the host field;
* a name is judged out loud — *used* if it exists, *created* if it is new,
  **refused** if SQLite could not take it;
* a near-miss (``annotaet`` next to ``annotate``) names the column it
  resembles and will not be accepted until the user confirms — this is
  the failure the whole feature exists to prevent;
* it is **read-only**: the file is byte-identical after a full
  open/browse/cancel cycle and no ``-wal`` is left behind;
* it is **cheap**: opening it runs no ``COUNT(*)``;
* every problem — missing database, not-a-database, no tables, no columns
  — lands **inline**, never in a modal that would hang a headless run.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from spacr.qt.widgets.column_picker import (
    ACTION_CONFIRM,
    ACTION_CREATE,
    ACTION_INVALID,
    ACTION_UNCHECKED,
    ACTION_USE,
    ColumnPickerButton,
    ColumnPickerDialog,
    SchemaReader,
    attach_column_picker,
    field_text,
    near_miss,
    open_reader,
    resolve_db_path,
    set_field_text,
    validate_column_name,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PNG_COLUMNS = ("png_path", "plate", "well", "field", "annotate", "test")
CELL_COLUMNS = ("plate", "well", "cell_area", "nucleus_area",
                "cell_channel_1_mean_intensity")


@pytest.fixture
def measdb(tmp_path):
    """A real run folder: ``<tmp>/run/measurements/measurements.db``."""
    root = tmp_path / "run"
    (root / "measurements").mkdir(parents=True)
    path = root / "measurements" / "measurements.db"
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE png_list (png_path TEXT, plate TEXT, well TEXT, "
        "field INTEGER, annotate INTEGER, test INTEGER)")
    con.execute(
        "CREATE TABLE cell (plate TEXT, well TEXT, cell_area REAL, "
        "nucleus_area REAL, cell_channel_1_mean_intensity REAL)")
    for i in range(20):
        con.execute("INSERT INTO png_list VALUES (?,?,?,?,?,?)",
                    (f"/data/{i}.png", "plate1", f"A{i % 12 + 1:02d}",
                     i % 3, 1 if i % 2 else None, None))
        con.execute("INSERT INTO cell VALUES (?,?,?,?,?)",
                    ("plate1", f"A{i % 12 + 1:02d}", 100.0 + i, 40.0 + i, 0.5))
    con.commit()
    con.close()
    return type("Db", (), {"path": str(path), "root": str(root)})()


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Fail loudly if anything under test opens a modal.

    The picker dialog is the one deliberate modal in this file, and it is
    injectable — tests hand ``ColumnPickerButton.set_dialog_runner`` a
    function that inspects the dialog and returns a result code, so no
    event loop is ever entered. Anything that reaches ``exec()`` is a bug.
    """
    from PySide6.QtWidgets import QFileDialog, QMessageBox

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


def _dialog(qtbot, db_path, **kw):
    d = ColumnPickerDialog(db_path=db_path, **kw)
    qtbot.addWidget(d)
    return d


def _digest(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _sidecars(path):
    d = os.path.dirname(path)
    return sorted(n for n in os.listdir(d)
                  if n.startswith(os.path.basename(path) + "-")
                  or n.endswith("-journal"))


# ---------------------------------------------------------------------------
# attach_column_picker — adopting a field a host already laid out
# ---------------------------------------------------------------------------

def test_attach_adds_a_button_beside_a_line_edit_without_breaking_the_form(
        qtbot, measdb):
    host = QWidget()
    qtbot.addWidget(host)
    form = QFormLayout(host)
    label = QLabel("Annotation column")
    field = QLineEdit("annotate")
    form.addRow(label, field)
    form.addRow("Image size", QLineEdit("200"))

    button = attach_column_picker(field, lambda: measdb.root, "png_list")

    assert isinstance(button, ColumnPickerButton)
    assert button.text() == "SQL"
    # The row still has its label, and the form still has both rows.
    assert form.rowCount() == 2
    assert form.itemAt(0, QFormLayout.LabelRole).widget() is label
    # The field kept its value and its identity — the host's reference to
    # it is still the widget the user types into.
    assert field.text() == "annotate"
    # It now lives in a wrapper that occupies the field's old slot.
    wrapper = form.itemAt(0, QFormLayout.FieldRole).widget()
    assert wrapper is not field
    assert field.parentWidget() is wrapper
    assert button.parentWidget() is wrapper
    assert wrapper.layout().indexOf(field) == 0
    assert wrapper.layout().indexOf(button) == 1


def test_attached_field_is_visible_once_the_host_is_shown(qtbot, measdb):
    """Re-parenting must not leave the field hidden."""
    host = QWidget()
    qtbot.addWidget(host)
    form = QFormLayout(host)
    field = QLineEdit("annotate")
    form.addRow("Annotation column", field)
    attach_column_picker(field, lambda: measdb.root, "png_list")

    host.show()
    qtbot.waitExposed(host)
    assert field.isVisible()


def test_attach_works_on_a_form_not_yet_installed_on_a_widget(qtbot, measdb):
    """Annotate builds its QFormLayout first and installs it last."""
    form = QFormLayout()
    field = QLineEdit("annotate")
    form.addRow("Annotation column", field)

    button = attach_column_picker(field, lambda: measdb.root, "png_list",
                                  layout=form)

    host = QWidget()
    qtbot.addWidget(host)
    outer = QVBoxLayout(host)
    outer.addLayout(form)
    wrapper = field.parentWidget()
    assert form.itemAt(0, QFormLayout.FieldRole).widget() is wrapper
    assert button.parentWidget() is wrapper
    # Installing the form adopts the wrapper — no stray top-level window.
    assert wrapper.parentWidget() is host
    assert not wrapper.isWindow()
    host.show()
    qtbot.waitExposed(host)
    assert field.isVisible() and button.isVisible()


def test_attach_on_a_field_with_no_layout_returns_an_unplaced_button(qtbot,
                                                                    measdb):
    host = QWidget()
    qtbot.addWidget(host)
    field = QLineEdit(host)
    button = attach_column_picker(field, lambda: measdb.root)
    assert isinstance(button, ColumnPickerButton)
    assert button.parentWidget() is host


def test_attach_on_a_field_outside_its_parents_layout_returns_it_unplaced(
        qtbot, measdb):
    """The parent has a layout, but this field was never added to it."""
    host = QWidget()
    qtbot.addWidget(host)
    form = QFormLayout(host)
    form.addRow("Something else", QLineEdit("x"))
    stray = QLineEdit(host)          # a child, but not in the form

    button = attach_column_picker(stray, lambda: measdb.root)
    assert button.parentWidget() is host
    assert form.rowCount() == 1


def test_attach_passes_through_a_tooltip_and_an_on_pick_callback(qtbot, measdb):
    host = QWidget()
    qtbot.addWidget(host)
    form = QFormLayout(host)
    field = QLineEdit("")
    form.addRow("Annotation column", field)
    seen = []

    button = attach_column_picker(field, lambda: measdb.path, "png_list",
                                  tooltip="Pick the annotation column",
                                  on_pick=seen.append)
    assert button.toolTip() == "Pick the annotation column"
    button.set_dialog_runner(
        lambda d: (d.select_column("annotate"), QDialog.Accepted)[1])
    button.open_picker()
    assert seen == ["annotate"]


def test_attach_finds_a_field_nested_in_an_inner_layout(qtbot, measdb):
    host = QWidget()
    qtbot.addWidget(host)
    outer = QVBoxLayout(host)
    inner = QFormLayout()
    field = QLineEdit("annotate")
    inner.addRow("Column", field)
    outer.addLayout(inner)

    attach_column_picker(field, lambda: measdb.root)
    assert inner.itemAt(0, QFormLayout.FieldRole).widget() is field.parentWidget()


# ---------------------------------------------------------------------------
# The dialog: real schema, picking, and filling the host field
# ---------------------------------------------------------------------------

def test_dialog_lists_the_real_tables_and_columns(qtbot, measdb):
    d = _dialog(qtbot, measdb.path)
    assert d.table_names() == ["cell", "png_list"]
    assert d.chosen_table() == "cell"
    assert d.column_names() == list(CELL_COLUMNS)
    assert d.select_table("png_list")
    assert d.column_names() == list(PNG_COLUMNS)
    assert d.banner_text() == ""


def test_dialog_accepts_a_run_folder_not_only_the_db_file(qtbot, measdb):
    d = _dialog(qtbot, measdb.root)
    assert "png_list" in d.table_names()


def test_dialog_preselects_the_requested_table(qtbot, measdb):
    d = _dialog(qtbot, measdb.path, table="png_list")
    assert d.chosen_table() == "png_list"
    assert "annotate" in d.column_names()


def test_dialog_shows_declared_types_and_a_labelled_row_estimate(qtbot, measdb):
    d = _dialog(qtbot, measdb.path, table="cell")
    tree = d._column_tree
    types = {tree.topLevelItem(i).text(0): tree.topLevelItem(i).text(1)
             for i in range(tree.topLevelItemCount())}
    assert types["cell_area"] == "REAL"
    assert types["plate"] == "TEXT"
    assert "estimate" in d._summary.text()
    assert "5 columns" in d._summary.text()


def test_the_column_filter_narrows_the_list_without_touching_the_database(
        qtbot, measdb):
    d = _dialog(qtbot, measdb.path, table="cell")
    before = len(d.executed_sql())
    d._filter.setText("area")
    assert d.visible_column_names() == ["cell_area", "nucleus_area"]
    assert len(d.executed_sql()) == before


def test_picking_a_column_fills_the_host_field(qtbot, measdb):
    field = QLineEdit("")
    qtbot.addWidget(field)
    button = ColumnPickerButton(lambda: measdb.path, table="png_list",
                                field=field)
    qtbot.addWidget(button)

    seen = []
    button.picked.connect(seen.append)
    button.set_dialog_runner(
        lambda d: (d.select_column("annotate"), QDialog.Accepted)[1])

    assert button.open_picker() == "annotate"
    assert field.text() == "annotate"
    assert seen == ["annotate"]


def test_cancelling_leaves_the_host_field_alone(qtbot, measdb):
    field = QLineEdit("annotate")
    qtbot.addWidget(field)
    button = ColumnPickerButton(lambda: measdb.path, table="png_list",
                                field=field)
    qtbot.addWidget(button)
    button.set_dialog_runner(
        lambda d: (d.set_name("something_else"), QDialog.Rejected)[1])
    assert button.open_picker() == ""
    assert field.text() == "annotate"


def test_clicking_the_button_opens_the_picker_and_no_modal(qtbot, measdb):
    field = QLineEdit("")
    qtbot.addWidget(field)
    button = attach_column_picker(field, lambda: measdb.path, "png_list")
    opened = []

    def _runner(dialog):
        opened.append(dialog)
        dialog.select_column("test")
        return QDialog.Accepted

    button.set_dialog_runner(_runner)
    qtbot.mouseClick(button, Qt.LeftButton)
    assert len(opened) == 1
    assert field.text() == "test"


def test_double_clicking_a_column_accepts_the_dialog(qtbot, measdb):
    d = _dialog(qtbot, measdb.path, table="png_list")
    d.select_column("annotate")
    results = []
    d.accepted.connect(lambda: results.append(d.chosen_column()))
    d._on_column_activated(d._column_tree.currentItem(), 0)
    assert results == ["annotate"]


def test_a_list_valued_field_is_appended_to_not_overwritten(qtbot, measdb):
    field = QLineEdit("['annotate']")
    qtbot.addWidget(field)
    button = ColumnPickerButton(lambda: measdb.path, table="png_list",
                                field=field)
    qtbot.addWidget(button)
    button.set_dialog_runner(
        lambda d: (d.select_column("test"), QDialog.Accepted)[1])
    button.open_picker()
    assert field.text() == "['annotate', 'test']"


def test_a_comma_separated_field_is_appended_to(qtbot, measdb):
    field = QLineEdit("cell_area")
    qtbot.addWidget(field)
    button = ColumnPickerButton(lambda: measdb.path, table="cell",
                                field=field, multi=True)
    qtbot.addWidget(button)
    button.set_dialog_runner(
        lambda d: (d.select_column("nucleus_area"), QDialog.Accepted)[1])
    button.open_picker()
    assert field.text() == "cell_area, nucleus_area"
    # Picking the same one twice does not duplicate it.
    button.open_picker()
    assert field.text() == "cell_area, nucleus_area"


def test_appending_to_a_list_field_ignores_a_duplicate(qtbot, measdb):
    field = QLineEdit("['annotate']")
    qtbot.addWidget(field)
    button = ColumnPickerButton(lambda: measdb.path, table="png_list",
                                field=field)
    qtbot.addWidget(button)
    button.set_dialog_runner(
        lambda d: (d.select_column("annotate"), QDialog.Accepted)[1])
    button.open_picker()
    assert field.text() == "['annotate']"


def test_appending_to_an_empty_multi_field_just_sets_the_name(qtbot, measdb):
    field = QLineEdit("")
    qtbot.addWidget(field)
    button = ColumnPickerButton(lambda: measdb.path, table="cell",
                                field=field, multi=True)
    qtbot.addWidget(button)
    button.set_dialog_runner(
        lambda d: (d.select_column("cell_area"), QDialog.Accepted)[1])
    button.open_picker()
    assert field.text() == "cell_area"


def test_a_combo_box_host_is_filled_too(qtbot, measdb):
    combo = QComboBox()
    combo.addItems(["annotate"])
    qtbot.addWidget(combo)
    button = ColumnPickerButton(lambda: measdb.path, table="png_list",
                                field=combo)
    qtbot.addWidget(button)
    button.set_dialog_runner(
        lambda d: (d.select_column("test"), QDialog.Accepted)[1])
    button.open_picker()
    assert combo.currentText() == "test"


# ---------------------------------------------------------------------------
# The verdict: used, created, refused
# ---------------------------------------------------------------------------

def test_an_existing_name_is_reported_as_will_be_used_not_created(qtbot, measdb):
    d = _dialog(qtbot, measdb.path, table="png_list")
    d.set_name("annotate")
    assert d.action() == ACTION_USE
    assert "will be used" in d.status_text()
    assert "will be created" not in d.status_text()
    assert d.is_accept_enabled()


def test_an_existing_name_matches_case_insensitively_like_sqlite(qtbot, measdb):
    d = _dialog(qtbot, measdb.path, table="png_list")
    d.set_name("Annotate")
    assert d.action() == ACTION_USE
    assert "'annotate'" in d.status_text()
    assert "ignores case" in d.status_text()


def test_a_genuinely_new_name_is_reported_as_will_be_created(qtbot, measdb):
    d = _dialog(qtbot, measdb.path, table="png_list")
    d.set_name("second_pass_scoring")
    assert d.action() == ACTION_CREATE
    assert "will be created" in d.status_text()
    assert d.is_accept_enabled()
    assert d.near_miss_column() == ""


def test_a_near_miss_warns_and_names_the_column_it_resembles(qtbot, measdb):
    """The load-bearing test: 'annotaet' must point at 'annotate'."""
    d = _dialog(qtbot, measdb.path, table="png_list")
    d.set_name("annotaet")

    assert d.action() == ACTION_CONFIRM
    assert d.near_miss_column() == "annotate"
    status = d.status_text()
    assert "annotate" in status
    assert "Did you mean 'annotate'?" in status
    # It is a warning, not a silent success — OK is refused until confirmed.
    assert not d.is_accept_enabled()
    assert d.confirm_offered()
    # And Enter in the name box cannot slip past the disabled button.
    d.accept()
    assert d.result() != QDialog.Accepted


def test_a_confirmed_near_miss_is_allowed_through(qtbot, measdb):
    d = _dialog(qtbot, measdb.path, table="png_list")
    d.set_name("annotaet")
    d.confirm_box().setChecked(True)
    assert d.action() == ACTION_CREATE
    assert "will be created" in d.status_text()
    assert "annotate" in d.status_text()
    assert d.is_accept_enabled()


def test_a_long_extension_of_an_existing_name_is_still_a_near_miss(qtbot,
                                                                  measdb):
    d = _dialog(qtbot, measdb.path, table="png_list")
    d.set_name("annotate_by_the_second_reviewer_in_january")
    assert d.action() == ACTION_CONFIRM
    assert d.near_miss_column() == "annotate"


def test_the_confirm_box_disappears_once_the_name_is_unambiguous(qtbot, measdb):
    d = _dialog(qtbot, measdb.path, table="png_list")
    d.set_name("annotaet")
    assert d.confirm_offered()
    d.set_name("cytokine_response")
    assert not d.confirm_offered()
    assert d.action() == ACTION_CREATE


@pytest.mark.parametrize("name, fragment", [
    ("my column", "space"),
    ("2nd_pass", "starts with a digit"),
    ("annotate-2", "quotes"),
    ("index", "reserved SQLite keyword"),
    ("GROUP", "reserved SQLite keyword"),
    ("sqlite_stat9", "'sqlite_' prefix"),
    (" annotate2 ", "leading or trailing spaces"),
    ("x" * 200, "128"),
    ("", "Type a column name"),
])
def test_a_name_sqlite_could_not_take_is_refused_with_a_reason(name, fragment):
    message = validate_column_name(name)
    assert message, f"{name!r} should have been refused"
    assert fragment in message


def test_a_valid_identifier_passes_validation():
    for name in ("annotate", "_private", "pass2", "cell_area_v2"):
        assert validate_column_name(name) == ""


def test_the_dialog_refuses_an_invalid_name_inline_and_disables_ok(qtbot,
                                                                  measdb):
    d = _dialog(qtbot, measdb.path, table="png_list")
    d.set_name("second pass")
    assert d.action() == ACTION_INVALID
    assert "space" in d.status_text()
    assert not d.is_accept_enabled()
    assert not d.confirm_offered()
    d.accept()
    assert d.result() != QDialog.Accepted


def test_a_reserved_keyword_is_refused_by_the_dialog(qtbot, measdb):
    d = _dialog(qtbot, measdb.path, table="png_list")
    d.set_name("select")
    assert d.action() == ACTION_INVALID
    assert "reserved SQLite keyword" in d.status_text()


def test_allow_new_false_refuses_a_column_that_does_not_exist(qtbot, measdb):
    d = _dialog(qtbot, measdb.path, table="png_list", allow_new=False)
    d.set_name("brand_new")
    assert d.action() == ACTION_INVALID
    assert "only accepts a column that already exists" in d.status_text()
    d.set_name("annotate")
    assert d.action() == ACTION_USE


def test_the_current_value_is_prefilled_and_judged_on_open(qtbot, measdb):
    d = _dialog(qtbot, measdb.path, table="png_list", current="annotate")
    assert d.chosen_column() == "annotate"
    assert d.action() == ACTION_USE


# ---------------------------------------------------------------------------
# Read-only, and cheap
# ---------------------------------------------------------------------------

def test_the_database_is_byte_identical_after_open_browse_cancel(qtbot, measdb):
    before = _digest(measdb.path)
    assert _sidecars(measdb.path) == []

    d = _dialog(qtbot, measdb.path)
    d.select_table("png_list")
    d.select_column("annotate")
    d._filter.setText("ann")
    d.set_name("annotaet")
    d.select_table("cell")
    d.set_name("cell_area")
    d.reject()
    d.close()

    assert _digest(measdb.path) == before
    assert _sidecars(measdb.path) == [], "a -wal/-journal file was left behind"


def test_the_reader_refuses_a_write_outright(measdb):
    reader = SchemaReader(measdb.path)
    con = reader._connect()
    try:
        with pytest.raises(sqlite3.OperationalError):
            con.execute('ALTER TABLE png_list ADD COLUMN sneaky INTEGER')
    finally:
        con.close()
    assert _digest(measdb.path) == _digest(measdb.path)


def test_opening_the_dialog_runs_no_count(qtbot, measdb):
    d = _dialog(qtbot, measdb.path, table="png_list")
    d.select_table("cell")
    d.select_column("cell_area")
    sql = " ".join(d.executed_sql()).upper()
    assert "COUNT(" not in sql
    assert "SELECT *" not in sql
    # The row figure came from max(rowid), and says so.
    assert any("MAX(ROWID)" in s.upper() for s in d.executed_sql())
    assert "estimate" in d._summary.text()


def test_the_non_null_count_happens_only_when_asked_for(qtbot, measdb):
    d = _dialog(qtbot, measdb.path, table="png_list")
    d.select_column("annotate")
    assert not any("COUNT(" in s.upper() for s in d.executed_sql())

    qtbot.mouseClick(d._count_btn, Qt.LeftButton)

    assert any("COUNT(" in s.upper() for s in d.executed_sql())
    item = d._column_tree.currentItem()
    assert item.text(2) == "10"          # 10 of 20 rows carry a label


def test_the_count_button_is_disabled_until_a_column_is_selected(qtbot, measdb):
    d = _dialog(qtbot, measdb.path, table="png_list")
    assert not d._count_btn.isEnabled()
    d.select_column("annotate")
    assert d._count_btn.isEnabled()


def test_the_row_estimate_is_none_for_a_view(qtbot, tmp_path):
    path = tmp_path / "v.db"
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE t (a INTEGER)")
    con.execute("CREATE VIEW v AS SELECT a FROM t")
    con.commit()
    con.close()
    d = _dialog(qtbot, str(path), table="v")
    assert d.column_names() == ["a"]
    assert "row count unknown" in d._summary.text()


def test_the_row_estimate_is_none_for_an_empty_table(qtbot, tmp_path):
    """max(rowid) answers NULL rather than raising — still no count."""
    path = tmp_path / "e.db"
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE empty_table (a INTEGER)")
    con.commit()
    con.close()
    d = _dialog(qtbot, str(path), table="empty_table")
    assert "row count unknown" in d._summary.text()
    assert SchemaReader(str(path)).estimate_rows("empty_table") is None


# ---------------------------------------------------------------------------
# Every failure lands inline
# ---------------------------------------------------------------------------

def test_no_database_selected_is_reported_inline(qtbot):
    d = _dialog(qtbot, "")
    assert "No database selected" in d.banner_text()
    assert d.table_names() == []
    assert "No database open" in d._source.text()
    # And asking for a table's columns with no database is a quiet no-op.
    d._load_columns("png_list")
    assert d.column_names() == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_a_database_that_cannot_be_opened_says_so_rather_than_blaming_sqlite(
        qtbot, measdb):
    os.chmod(measdb.path, 0o000)
    try:
        d = _dialog(qtbot, measdb.path)
        assert "Cannot open" in d.banner_text()
        assert "not a SQLite database" not in d.banner_text()
    finally:
        os.chmod(measdb.path, 0o644)


def test_a_missing_database_is_reported_inline(qtbot, tmp_path):
    d = _dialog(qtbot, str(tmp_path / "nope" / "measurements.db"))
    assert "No database at" in d.banner_text()


def test_a_folder_with_no_database_is_reported_inline(qtbot, tmp_path):
    (tmp_path / "empty_run").mkdir()
    d = _dialog(qtbot, str(tmp_path / "empty_run"))
    assert "No database at" in d.banner_text()


def test_a_file_that_is_not_a_database_is_reported_inline(qtbot, tmp_path):
    junk = tmp_path / "measurements.db"
    junk.write_bytes(b"this is a CSV, not a database\n" * 100)
    d = _dialog(qtbot, str(junk))
    assert "not a SQLite database" in d.banner_text()
    assert d.table_names() == []


def test_a_database_with_no_tables_is_reported_inline(qtbot, tmp_path):
    path = tmp_path / "empty.db"
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE gone (a INTEGER)")
    con.execute("DROP TABLE gone")
    con.commit()
    con.close()
    d = _dialog(qtbot, str(path))
    assert "has no tables" in d.banner_text()
    assert d.table_names() == []


def test_a_table_whose_columns_cannot_be_read_is_reported_inline(qtbot,
                                                                tmp_path):
    """A view over a dropped table — SQLite resolves it at PRAGMA time."""
    path = tmp_path / "broken.db"
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE t (a INTEGER)")
    con.execute("CREATE VIEW orphan AS SELECT a FROM t")
    con.commit()
    con.execute("DROP TABLE t")
    con.commit()
    con.close()
    d = _dialog(qtbot, str(path), table="orphan")
    assert "Cannot list the columns of 'orphan'" in d.banner_text()
    assert d.column_names() == []


def test_a_table_reporting_no_columns_is_reported_inline(qtbot, measdb):
    class _NoColumns(SchemaReader):
        def column_info(self, table):
            return []

    d = ColumnPickerDialog(reader=_NoColumns(measdb.path), table="png_list")
    qtbot.addWidget(d)
    assert "reports no columns" in d.banner_text()
    assert d.column_names() == []


def test_a_schema_that_cannot_be_listed_is_reported_inline(qtbot, measdb):
    class _Broken(SchemaReader):
        def tables(self):
            raise sqlite3.OperationalError("database disk image is malformed")

    d = ColumnPickerDialog(reader=_Broken(measdb.path))
    qtbot.addWidget(d)
    assert "Cannot read the schema" in d.banner_text()


def test_a_failed_count_is_reported_inline(qtbot, measdb):
    class _CountFails(SchemaReader):
        def count_non_null(self, table, column):
            raise sqlite3.OperationalError("disk I/O error")

    d = ColumnPickerDialog(reader=_CountFails(measdb.path), table="png_list")
    qtbot.addWidget(d)
    d.select_column("annotate")
    d._count_selected()
    assert "Could not count 'annotate'" in d.banner_text()


def test_a_name_cannot_be_checked_without_a_database(qtbot):
    d = _dialog(qtbot, "")
    d.set_name("annotate")
    assert d.action() == ACTION_UNCHECKED
    assert "cannot be checked" in d.status_text()
    assert d.is_accept_enabled()
    d.set_name("")
    assert d.action() == ACTION_INVALID
    assert not d.is_accept_enabled()


def test_counting_without_a_selection_is_a_no_op(qtbot, measdb):
    d = _dialog(qtbot, measdb.path, table="png_list")
    d._count_selected()
    assert d.banner_text() == ""


# ---------------------------------------------------------------------------
# The small pure helpers
# ---------------------------------------------------------------------------

def test_near_miss_finds_the_typo_and_ignores_the_unrelated():
    columns = ["png_path", "annotate", "cell_area"]
    assert near_miss("annotaet", columns) == "annotate"
    assert near_miss("Annotate", columns) == ""       # that IS the column
    assert near_miss("annotate", columns) == ""
    assert near_miss("recruitment", columns) == ""
    assert near_miss("", columns) == ""
    assert near_miss("annotate", []) == ""


def test_resolve_db_path_takes_a_file_a_run_folder_or_a_db_folder(measdb,
                                                                  tmp_path):
    assert resolve_db_path(measdb.path) == os.path.abspath(measdb.path)
    assert resolve_db_path(measdb.root) == os.path.abspath(measdb.path)
    assert resolve_db_path(os.path.join(measdb.root, "measurements")) == \
        os.path.abspath(measdb.path)
    with pytest.raises(ValueError):
        resolve_db_path("   ")
    # A folder with nothing in it still resolves to where the db *would* be.
    empty = tmp_path / "nothing"
    empty.mkdir()
    assert resolve_db_path(str(empty)).endswith(
        os.path.join("measurements", "measurements.db"))


def test_open_reader_reports_a_path_that_is_not_a_file(tmp_path):
    """A folder called measurements.db resolves to the db that would be
    inside it, and the message names that path rather than crashing."""
    (tmp_path / "measurements.db").mkdir()
    reader, message = open_reader(str(tmp_path / "measurements.db"))
    assert reader is None
    assert "No database at" in message


def test_open_reader_returns_a_reader_for_a_real_database(measdb):
    reader, message = open_reader(measdb.path)
    assert message == ""
    assert "png_list" in reader.tables()


def test_field_text_and_set_field_text_cover_the_widget_kinds(qtbot):
    line = QLineEdit("a")
    qtbot.addWidget(line)
    assert field_text(line) == "a"
    assert set_field_text(line, "b") is True
    assert line.text() == "b"

    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.setEditable(True)
    combo.setEditText("x")
    assert field_text(combo) == "x"
    assert set_field_text(combo, "y") is True
    assert combo.currentText() == "y"

    label = QLabel("not a field")
    qtbot.addWidget(label)
    assert field_text(label) == ""
    assert set_field_text(label, "z") is False
    assert set_field_text(None, "z") is False


def test_a_broken_path_getter_yields_an_empty_path_not_a_crash(qtbot):
    def _explode():
        raise RuntimeError("src not set")

    button = ColumnPickerButton(_explode)
    qtbot.addWidget(button)
    assert button.db_path() == ""
    button.set_dialog_runner(lambda d: QDialog.Rejected)
    assert button.open_picker() == ""


def test_a_plain_string_path_is_accepted_instead_of_a_getter(qtbot, measdb):
    button = ColumnPickerButton(measdb.path)
    qtbot.addWidget(button)
    assert button.db_path() == measdb.path
    button.set_dialog_runner(lambda d: QDialog.Rejected)
    assert button.open_picker() == ""


def test_selecting_a_table_or_column_that_is_absent_returns_false(qtbot,
                                                                 measdb):
    d = _dialog(qtbot, measdb.path, table="png_list")
    assert d.select_table("no_such_table") is False
    assert d.select_column("no_such_column") is False
    assert d.chosen_table() == "png_list"


def test_accepting_with_an_empty_name_is_refused(qtbot, measdb):
    field = QLineEdit("annotate")
    qtbot.addWidget(field)
    button = ColumnPickerButton(lambda: measdb.path, table="png_list",
                                field=field)
    qtbot.addWidget(button)
    button.set_dialog_runner(
        lambda d: (d.set_name(""), QDialog.Accepted)[1])
    assert button.open_picker() == ""
    assert field.text() == "annotate"


def test_the_name_edit_is_exposed_for_hosts_that_want_to_focus_it(qtbot,
                                                                 measdb):
    d = _dialog(qtbot, measdb.path, table="png_list")
    assert isinstance(d.name_edit(), QLineEdit)
    d.name_edit().setText("annotate")
    assert d.chosen_column() == "annotate"
