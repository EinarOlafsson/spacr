"""The model zoo picker at its seams: odd clicks, folders, failures, teardown.

Pinned here, each as what the user sees or gets:

* the download folder chosen last time is where the next download goes, and
  choosing a folder in the dialog is remembered; cancelling the folder chooser
  changes nothing;
* a left click on a source heading asks to flip it, any other button does not;
  an unknown heading cannot be switched on;
* sizes, rates and times read as a person reads them, up to gigabytes and
  hours;
* an install dialog opened through :func:`install_backend` is handed to the
  caller's watcher first, and what it reports comes back;
* a backend whose state cannot be read is "unknown" rather than an error;
* a diameter popup that cannot be built says so on the status line;
* a folder that cannot be created is reported, not downloaded into;
* the progress line is throttled and a finished download joins its thread;
* the dialog closes cleanly when a thread or popup it holds is already gone;
* a community upload carries its warning on the card;
* a heading ignores a change event it cannot read, an install whose row
  has gone selects nothing, and a catalogue refresh that cannot start or
  fetch leaves the table as it was.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt, QThread
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QDialog, QTableWidgetItem

from spacr import model_zoo as mz
from spacr.qt.widgets import model_zoo_picker as mzp

_REAL_WARM_COMMUNITY = mzp.ModelZooPicker._warm_the_community_catalogue
_REAL_WARM_BIOIMAGEIO = mzp.ModelZooPicker._warm_bioimageio


@pytest.fixture
def picker(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(mzp.ModelZooPicker, "_warm_the_community_catalogue",
                        lambda self: None)
    monkeypatch.setattr(mzp.ModelZooPicker, "_warm_bioimageio",
                        lambda self: None, raising=False)
    monkeypatch.setattr(mzp, "DEFAULT_MODEL_DIR", str(tmp_path))
    dialog = mzp.ModelZooPicker(kinds=("cellpose",))
    dialog.folder_edit.setText(str(tmp_path))
    yield dialog
    dialog.reject()
    dialog.deleteLater()


def _first_downloadable_row(picker):
    for row, (stem, pairs) in enumerate(picker._groups):
        entry = pairs[picker._chosen[stem]][1]
        if picker.table.isRowHidden(row):
            continue
        if entry.kind != "backend" and picker._local_path(entry) is None:
            return row
    pytest.skip("every offered model is already present")


# ---------------------------------------------------------------------------
# The download folder
# ---------------------------------------------------------------------------

def test_the_folder_used_last_time_is_offered_again(qapp, tmp_path):
    mzp._store().setValue(mzp._DIR_SETTING, str(tmp_path / "models"))
    assert mzp.remembered_model_dir() == str(tmp_path / "models")


def test_choosing_a_folder_shows_and_remembers_it(picker, tmp_path,
                                                  monkeypatch):
    chosen = str(tmp_path / "elsewhere")
    monkeypatch.setattr(mzp.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: chosen))
    picker._browse()
    assert picker.folder_edit.text() == chosen
    assert mzp.remembered_model_dir() == chosen


def test_cancelling_the_folder_chooser_changes_nothing(picker, tmp_path,
                                                       monkeypatch):
    monkeypatch.setattr(mzp.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    picker._browse()
    assert picker.folder_edit.text() == str(tmp_path)
    assert mzp._store().value(mzp._DIR_SETTING, "") in ("", None)


def test_a_folder_that_cannot_be_made_is_reported(picker, tmp_path,
                                                  monkeypatch):
    blocker = tmp_path / "a_file"
    blocker.write_text("not a folder")
    picker.folder_edit.setText(str(blocker / "models"))
    picker.table.selectRow(_first_downloadable_row(picker))
    warned = []
    monkeypatch.setattr(mzp.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.append(a[2])))

    picker._download_selected()

    assert len(warned) == 1 and "Cannot write to" in warned[0]
    assert not picker.progress.isVisible()
    assert not os.path.exists(blocker / "models")


def test_download_with_nothing_selected_does_nothing(picker):
    picker.table.clearSelection()
    picker._download_selected()
    assert picker.selected_entry() is None
    assert not picker.progress.isVisible()


# ---------------------------------------------------------------------------
# Source headings
# ---------------------------------------------------------------------------

def _click(widget, button):
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(2, 2),
                       QPointF(2, 2), button, button, Qt.NoModifier)


def test_only_a_left_click_flips_a_heading(qtbot):
    heading = mzp._SourceHeading("bioimage.io")
    qtbot.addWidget(heading)
    assert heading.source_name == "bioimage.io"
    seen = []
    heading.clicked.connect(lambda: seen.append(True))

    heading.mousePressEvent(_click(heading, Qt.RightButton))
    assert seen == []
    heading.mousePressEvent(_click(heading, Qt.LeftButton))
    assert seen == [True]


def test_a_heading_shrugs_off_an_event_it_cannot_read(qtbot, monkeypatch):
    heading = mzp._SourceHeading("bioimage.io")
    qtbot.addWidget(heading)
    restyled = []
    monkeypatch.setattr(heading, "_restyle", lambda: restyled.append(True))

    class Unreadable(QEvent):
        def type(self):
            raise RuntimeError("event already deleted")

    heading.changeEvent(Unreadable(QEvent.Type.StyleChange))
    assert restyled == []
    assert heading.text() == "bioimage.io"


def test_an_unknown_heading_cannot_be_switched_on(picker):
    assert picker.sources.set_on("nowhere", True) is False
    assert "nowhere" not in picker.sources.enabled()


# ---------------------------------------------------------------------------
# Numbers a person reads
# ---------------------------------------------------------------------------

def test_sizes_rates_and_times_read_up_to_gigabytes_and_hours():
    assert mzp._human_bytes_local(512) == "512 B"
    assert mzp._human_bytes_local(1536) == "1.5 kB"
    assert mzp._human_bytes_local(5 * 1024 ** 2) == "5.0 MB"
    assert mzp._human_rate(100) == "100.0 B/s"
    assert mzp._human_rate(3 * 1024 ** 2) == "3.0 MB/s"
    assert mzp._human_bytes_local(3 * 1024 ** 3) == "3.0 GB"
    assert mzp._human_bytes_local(5000 * 1024 ** 3) == "5000.0 GB"
    assert mzp._human_rate(2 * 1024 ** 3) == "2.0 GB/s"
    assert mzp._human_eta(2 * 3600 + 5 * 60) == "2h 5m left"


# ---------------------------------------------------------------------------
# Installing a backend
# ---------------------------------------------------------------------------

def test_the_watcher_sees_the_install_dialog_before_it_opens(qapp,
                                                             monkeypatch):
    events = []

    class FakeDialog:
        def __init__(self, name, parent, reinstall=False, why=""):
            self.name = name
            self.installed = True

        def exec(self):
            events.append("exec")

    monkeypatch.setattr(mzp, "BackendInstallDialog", FakeDialog)
    ok = mzp.install_backend(None, "spotnet",
                             watch=lambda d: events.append(("watch", d.name)))
    assert ok is True
    assert events == [("watch", "spotnet"), "exec"]


def test_a_backend_whose_state_cannot_be_read_is_unknown(monkeypatch):
    from spacr import _segmentation_backends as backends

    def unreadable(name):
        raise OSError("disk went away")

    monkeypatch.setattr(backends, "_backend_state", unreadable)
    assert mzp._disk_state("spotnet") is None


def test_an_install_line_that_is_not_a_step_is_shown_as_printed(qtbot):
    def forbidden_job(progress=None, cancel=None):
        raise AssertionError("the job must not run")

    dialog = mzp.BackendInstallDialog("spotnet", job=forbidden_job)
    qtbot.addWidget(dialog)
    dialog._on_progress(0, 1, "Collecting numpy")
    assert dialog.status.text() == "Collecting numpy"


def test_an_install_that_names_no_listed_model_selects_nothing(picker,
                                                               monkeypatch):
    monkeypatch.setattr(mzp, "install_backend_package",
                        lambda parent, entry: True)
    picker.table.clearSelection()
    picker._install_backend(types.SimpleNamespace(name="no such backend"))
    assert picker.selected_entry() is None


def test_an_install_whose_row_has_gone_selects_nothing(picker, monkeypatch):
    monkeypatch.setattr(mzp, "install_backend_package",
                        lambda parent, entry: True)
    monkeypatch.setattr(picker, "refresh", lambda: None)
    last = len(picker._groups) - 1
    _stem, pairs = picker._groups[last]
    name = pairs[0][1].name
    picker.table.removeRow(picker.table.rowCount() - 1)
    picker.table.clearSelection()
    picker._install_backend(types.SimpleNamespace(name=name))
    assert picker.table.selectedItems() == []
    assert picker.status.text() == f"Installing {name}…"


def test_a_catalogue_refresh_that_cannot_start_leaves_the_table(picker,
                                                                monkeypatch):
    import sys

    monkeypatch.setattr(mz, "shared_catalogue_is_stale", lambda: True)
    monkeypatch.setitem(sys.modules, "spacr.qt.job_runner", None)
    rows = picker.table.rowCount()
    _REAL_WARM_COMMUNITY(picker)
    assert getattr(picker, "_catalogue_job", None) is None
    assert picker.table.rowCount() == rows


def test_a_bioimageio_fetch_that_fails_keeps_the_cached_rows(picker,
                                                             monkeypatch):
    class InlineThread:
        def __init__(self, target, daemon=False):
            self._target = target

        def start(self):
            self._target()

    def offline(allow_network=False):
        raise OSError("offline")

    monkeypatch.setattr(mzp.threading, "Thread", InlineThread)
    monkeypatch.setattr(mz, "bioimageio_entries", offline)
    redraws = []
    picker._bioimageio_warmed.connect(lambda: redraws.append(True))
    rows = picker.table.rowCount()
    _REAL_WARM_BIOIMAGEIO(picker)
    assert redraws == []
    assert picker.table.rowCount() == rows


# ---------------------------------------------------------------------------
# The table's bookkeeping
# ---------------------------------------------------------------------------

def test_a_row_the_table_no_longer_has_is_skipped(picker):
    rows = picker.table.rowCount()
    picker.table.removeRow(rows - 1)
    lost = len(picker._groups) - 1
    assert picker._row_of_group(lost) is None
    picker._fill_row(lost)
    picker._apply_source_filter()
    assert picker.table.rowCount() == rows - 1


def test_a_row_with_no_model_behind_it_selects_nothing(picker):
    picker.table.item(0, 0).setData(Qt.UserRole, None)
    assert picker._group_of_row(0) is None
    picker.table.setRowHidden(0, False)
    picker.table.selectRow(0)
    assert picker.selected_entry() is None


def test_a_version_for_a_model_that_is_not_listed_is_ignored(picker):
    before = [picker.table.item(r, 3).text()
              for r in range(picker.table.rowCount())]
    picker._version_picked(len(picker._groups) + 5, 0)
    after = [picker.table.item(r, 3).text()
             for r in range(picker.table.rowCount())]
    assert after == before


def test_a_row_without_a_first_cell_has_no_model(picker):
    picker.table.setItem(0, 0, QTableWidgetItem("x"))
    picker.table.takeItem(0, 0)
    assert picker._group_of_row(0) is None


# ---------------------------------------------------------------------------
# Status, progress, teardown
# ---------------------------------------------------------------------------

def test_a_diameter_popup_that_cannot_open_says_so(picker, monkeypatch):
    from spacr.qt import prerun

    monkeypatch.setattr(prerun, "diameter_dialog", lambda *a, **k: None)
    picker._measure_diameters()
    assert picker.status.text() == "Could not open the diameter estimate."


def test_the_progress_line_is_not_redrawn_faster_than_it_can_be_read(picker):
    import time

    picker._started_at = time.monotonic() - 1
    picker._last_emit = 0.0
    picker._on_progress(10, 100)
    first = picker.status.text()
    assert "of" in first
    picker._on_progress(20, 100)
    assert picker.status.text() == first


def test_a_finished_download_joins_its_thread(picker, monkeypatch):
    monkeypatch.setattr(picker, "refresh", lambda: None)
    picker._thread = QThread()
    picker._worker = object()
    picker.status.setText("Downloading…")
    picker._finish_download("")
    assert picker._thread is None
    assert picker._worker is None
    assert picker.status.text() == "Downloading…"


def test_a_thread_that_is_already_gone_does_not_block_closing(picker):
    class Gone:
        def isRunning(self):  # noqa: N802 - Qt naming
            raise RuntimeError("Internal C++ object already deleted.")

    picker._thread = Gone()
    picker._stop_any_download()
    assert picker._thread is None


def test_a_diameter_popup_that_is_already_gone_does_not_block_closing(picker):
    class Gone:
        def close(self):
            raise RuntimeError("Internal C++ object already deleted.")

    picker._diameter_dialog = Gone()
    picker.done(QDialog.Rejected)
    assert picker.result() == QDialog.Rejected


def test_a_community_heading_whose_fetch_fails_still_lists(picker,
                                                           monkeypatch):
    def offline(allow_network=False, **_k):
        raise OSError("offline")

    monkeypatch.setattr(mz, "community_entries", offline)
    monkeypatch.setattr(picker, "refresh", lambda: None)
    picker.sources._headings["spaCR community"].set_on(True)
    picker._sources_changed()
    assert picker.status.text() == ""


# ---------------------------------------------------------------------------
# The card and the helpers behind it
# ---------------------------------------------------------------------------

def test_a_community_upload_carries_its_warning(picker):
    entry = mz.ModelEntry(key="someone_v1", name="someone_v1.CP_model",
                          source="community", sha256="ab" * 32)
    picker._show_card(entry)
    assert "NOT vetted" in picker.card.toPlainText()


def test_a_backend_explanation_survives_an_unreadable_backend_list(
        monkeypatch):
    from spacr import _segmentation_backends as backends

    class Broken:
        def get(self, _key):
            raise RuntimeError("catalogue half-built")

    monkeypatch.setattr(backends, "_SPECS", Broken())
    entry = types.SimpleNamespace(name="Something", uri="backend:something",
                                  source="installable")
    text = mzp._where_a_backend_is_chosen(entry)
    assert "Something" in text


def test_a_dino_card_without_a_licence_does_not_invent_one():
    card = mzp._cellpose_dino_card(types.SimpleNamespace(licence=""))
    with_licence = mzp._cellpose_dino_card(
        types.SimpleNamespace(licence="MIT"))
    assert "Licence: MIT" in with_licence
    assert "Licence: MIT" not in card
    assert with_licence.count("<p>") == card.count("<p>") + 1
