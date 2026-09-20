"""Press FEATURES, drop files on the table, press Measure, get a database.

Instruction 421, driven the way a user hits it rather than through the model
underneath -- HANDOFF section 0b: instruction 52 was closed on 97 passing
tests and the controls were unreachable, because no test pressed one. So
every assertion here follows something that was pressed, dropped or
double-clicked.

The model these widgets edit is tested Qt-free in
``tests/test_features_button_measures_hand_drawn_masks.py``.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from spacr.measure import FIELD_TABLE_DECIDED_KEYS
from spacr.qt.screens.measure_inputs import (
    SETTINGS_APP_KEY,
    MeasureInputsScreen,
)
from spacr.qt.widgets.measure_input_table import (
    DEFAULT_REGEX,
    MeasureInputTable,
    role_caption,
)

LEAN = {
    "n_jobs": 1, "save_png": False, "plot": False, "verbose": False,
    "cell_min_size": 0, "nucleus_min_size": 0, "pathogen_min_size": 0,
    "cytoplasm_min_size": 0, "homogeneity": False, "radial_dist": False,
    "spatial_measurements": False, "object_distances": False,
    "calculate_correlation": False,
}


def _write(path, array):
    """Write one 16-bit TIFF, making its folder first."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, np.asarray(array), photometric="minisblack")
    return path


@pytest.fixture
def drawn(tmp_path):
    """Two fields of two channels and two masks, as loose files."""
    folder = tmp_path / "drawn"
    yy, xx = np.indices((32, 32))
    for field in (1, 2):
        _write(str(folder / f"fov00{field}_C1.tif"),
               ((yy * 31 + xx * field) % 4096).astype(np.uint16))
        _write(str(folder / f"fov00{field}_C2.tif"),
               ((xx * 17 + yy * field) % 4096).astype(np.uint16))
        cell = np.zeros((32, 32), np.uint16)
        cell[3:29, 3:29] = 1
        nucleus = np.zeros((32, 32), np.uint16)
        nucleus[10:20, 11:21] = 1
        _write(str(folder / f"fov00{field}_cell_mask.tif"), cell)
        _write(str(folder / f"fov00{field}_nucleus_mask.tif"), nucleus)
    return folder


#: Every QMimeData a drop in this file was built from, kept alive for the
#: run. A QDropEvent does NOT own the QMimeData it is given: build the two in
#: a helper and the mime is collected the moment the helper returns, leaving
#: the event holding a dangling pointer and the process segfaulting inside
#: ``event.mimeData()``. Nothing about the crash names the helper.
_MIME_KEPT_ALIVE = []


def _drop_event(paths):
    """A drop carrying local file URLs, as a file manager sends one."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    _MIME_KEPT_ALIVE.append(mime)
    return QDropEvent(QPointF(4, 4), Qt.CopyAction, mime,
                      Qt.LeftButton, Qt.NoModifier)


def _files(folder):
    """Every file in ``folder``, sorted."""
    return [str(folder / name) for name in sorted(os.listdir(folder))]


def _widget_text(widget):
    """Whatever a settings control is showing, however it spells it."""
    for name in ("text", "currentText", "value", "get_value"):
        getter = getattr(widget, name, None)
        if callable(getter):
            try:
                return str(getter())
            except TypeError:
                continue
    return ""


def test_dropping_the_folder_fills_the_table(qtbot, drawn):
    """A real drop event, not a helper call, and the table is populated."""
    widget = MeasureInputTable(threaded=False)
    qtbot.addWidget(widget)

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(drawn))])
    enter = QDragEnterEvent(QPointF(4, 4).toPoint(), Qt.CopyAction, mime,
                            Qt.LeftButton, Qt.NoModifier)
    widget.dragEnterEvent(enter)
    assert enter.isAccepted()

    widget.dropEvent(_drop_event([drawn]))

    table = widget.table()
    assert [row.label for row in table.rows] == ["fov001", "fov002"]
    assert table.n_channels == 2
    assert table.ordered_roles() == ("cell", "nucleus")
    assert widget.problems() == []
    assert widget.unassigned() == []
    assert widget._grid.rowCount() == 2
    assert widget._grid.columnCount() == 3 + 2 + 2


def test_the_grid_shows_which_file_landed_in_which_cell(qtbot, drawn):
    """What the user checks before pressing Measure has to be on screen."""
    widget = MeasureInputTable(threaded=False)
    qtbot.addWidget(widget)
    widget.dropEvent(_drop_event(_files(drawn)))

    headings = [widget._grid.horizontalHeaderItem(column).text()
                for column in range(widget._grid.columnCount())]
    assert headings[:3] == ["Field", "Well", "Field #"]
    assert headings[3:5] == ["Channel 1", "Channel 2"]
    assert headings[5:] == [role_caption("cell"), role_caption("nucleus")]

    assert widget._grid.item(0, 3).text() == "fov001_C1.tif"
    assert widget._grid.item(0, 5).text() == "fov001_cell_mask.tif"
    assert widget._grid.item(0, 5).toolTip().endswith("fov001_cell_mask.tif")
    assert widget._grid.item(0, 0).toolTip().endswith("drawn_A01_1")


def test_a_regex_that_places_nothing_lists_every_file_it_could_not(qtbot,
                                                                   drawn):
    """A file that silently vanishes is the one failure this cannot afford."""
    widget = MeasureInputTable(threaded=False)
    qtbot.addWidget(widget)
    widget.dropEvent(_drop_event(_files(drawn)))
    assert widget.unassigned() == []

    widget.set_regex(r'(?P<field>nothing)_(?P<channel>\d+)')

    assert widget.table().rows == []
    assert len(widget.unassigned()) == 8
    assert widget._left_over.count() == 8
    assert "went nowhere" in widget._left_over_caption.text()

    widget.set_regex(DEFAULT_REGEX)
    assert len(widget.table().rows) == 2
    assert widget.unassigned() == []


def test_a_regex_that_will_not_compile_says_so_and_changes_nothing(qtbot,
                                                                   drawn):
    """The box is a place to make mistakes in, not a way to break a run."""
    widget = MeasureInputTable(threaded=False)
    qtbot.addWidget(widget)
    widget.dropEvent(_drop_event(_files(drawn)))

    widget._regex.setText(r'(?P<field>')

    assert "will not compile" in widget._regex_status.text()
    assert len(widget.table().rows) == 2


def test_double_clicking_a_cell_browses_for_that_one_file(qtbot, drawn,
                                                          tmp_path):
    """The second way to fill a cell. Driven through the widget's own signal.

    Headless Qt refuses a static modal -- HANDOFF 3b -- so the chooser is
    replaced rather than patched onto QFileDialog.
    """
    widget = MeasureInputTable(threaded=False)
    qtbot.addWidget(widget)
    widget.dropEvent(_drop_event(_files(drawn)))
    replacement = _write(str(tmp_path / "redrawn_cell.tif"),
                         np.ones((32, 32), np.uint16))
    asked = []
    widget.set_file_picker(lambda caption: asked.append(caption) or
                           replacement)

    widget._grid.cellDoubleClicked.emit(0, 5)

    assert asked and "fov001" in asked[0]
    assert widget.table().rows[0].masks["cell"] == replacement
    assert widget._grid.item(0, 5).text() == "redrawn_cell.tif"


def test_the_channel_count_and_the_mask_boxes_reshape_the_table(qtbot, drawn):
    """The columns are controls, so changing one has to change the model."""
    widget = MeasureInputTable(threaded=False)
    qtbot.addWidget(widget)
    widget.dropEvent(_drop_event(_files(drawn)))

    widget._role_boxes["pathogen"].setChecked(True)

    assert widget.table().ordered_roles() == ("cell", "nucleus", "pathogen")
    assert widget.table().mask_dims()["pathogen"] == 4
    assert any("pathogen mask" in text for text in widget.problems())

    widget._role_boxes["pathogen"].setChecked(False)
    assert widget.problems() == []

    widget._channels.setValue(1)
    assert widget.table().n_channels == 1
    assert widget.table().mask_dims() == {"cell": 1, "nucleus": 2}
    assert widget.problems() == []


def test_the_window_reuses_measures_own_settings_form(qtbot, drawn):
    """Not a copy of Measure's settings and not a chosen subset of them."""
    from spacr.qt.screens.settings_model import SettingsWidgets

    screen = MeasureInputsScreen(threaded=False)
    qtbot.addWidget(screen)

    assert screen.settings.app_key == SETTINGS_APP_KEY
    measure_keys = set(SettingsWidgets(SETTINGS_APP_KEY)._widgets)
    assert measure_keys <= set(screen.settings._widgets)
    for key in ("save_png", "use_bounding_box", "png_size", "normalize",
                "crop_mode", "cell_min_size"):
        assert key in screen.settings._widgets


def test_the_settings_the_table_decides_are_shown_filled_and_disabled(qtbot,
                                                                      drawn):
    """Shown says the table answered it; missing would say spaCR cannot."""
    screen = MeasureInputsScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.inputs.dropEvent(_drop_event(_files(drawn)))

    assert set(screen._decided_widgets) <= set(FIELD_TABLE_DECIDED_KEYS)
    for key in ("cell_mask_dim", "nucleus_mask_dim", "channels"):
        assert key in screen._decided_widgets
        assert not screen._decided_widgets[key].isEnabled()

    derived = screen.derived_settings()
    assert derived["cell_mask_dim"] == 2
    assert derived["nucleus_mask_dim"] == 3
    assert derived["pathogen_mask_dim"] is None
    assert derived["channels"] == [0, 1]


def test_run_is_refused_until_the_table_is_complete(qtbot, drawn):
    """The button is dead while a cell is empty, and says which one."""
    screen = MeasureInputsScreen(threaded=False)
    qtbot.addWidget(screen)

    assert not screen.run_button.isEnabled()
    assert screen.run() is False

    screen.inputs.dropEvent(_drop_event(_files(drawn)))
    assert screen.run_button.isEnabled()

    del screen.inputs.table().rows[0].channels[1]
    screen.inputs._rebuild()
    assert not screen.run_button.isEnabled()
    assert "channel 2" in screen._status.text()


def test_pressing_measure_writes_a_measure_project(qtbot, drawn, tmp_path):
    """The whole path, from a dropped folder to a measurements database."""
    screen = MeasureInputsScreen(threaded=False)
    qtbot.addWidget(screen)
    destination = tmp_path / "features"
    screen.set_destination(str(destination))
    screen.inputs.dropEvent(_drop_event(_files(drawn)))
    assert screen.apply_settings_dict(LEAN) == len(LEAN)
    assert screen.settings.collect()["cell_min_size"] == 0

    with qtbot.waitSignal(screen.run_finished, timeout=120000):
        screen.run_button.click()

    result = screen.result()
    assert result is not None
    assert result["stems"] == ["drawn_A01_1", "drawn_A01_2"]
    db_path = result["db_path"]
    assert os.path.isfile(db_path)
    assert os.path.isdir(destination / "merged")
    assert os.path.isdir(destination / "masks" / "cell_mask_stack")

    with sqlite3.connect(db_path) as connection:
        tables = {row[0] for row in connection.execute(
            "select name from sqlite_master where type='table'")}
        assert {"cell", "nucleus", "cytoplasm"} <= tables
        assert connection.execute(
            "select count(*) from cell").fetchone()[0] == 2
    assert "Database:" in screen._log.toPlainText()


def test_make_masks_has_a_features_button_that_opens_the_table(qtbot,
                                                               drawn):
    """The button exists on the screen it was asked for, and it opens this."""
    from spacr.qt.screens.make_masks import MakeMasksScreen

    screen = MakeMasksScreen()
    qtbot.addWidget(screen)

    assert screen._btn_features.text() == "Features"
    assert screen._btn_features.toolTip()

    screen._folder = str(drawn)
    window = screen._on_open_features()
    qtbot.addWidget(window)

    assert isinstance(window, MeasureInputsScreen)
    assert window.destination() == os.path.join(str(drawn), "features")

    assert window.inputs.table().rows == [], (
        "the folder must not have been read on the GUI thread; pressing "
        "FEATURES on a folder living on a sleeping automount is the freeze "
        "this walk was moved to a worker to remove")
    qtbot.waitUntil(lambda: not window.inputs.is_scanning(), timeout=5000)
    assert [row.label for row in window.inputs.table().rows] == [
        "fov001", "fov002"]
    window.close()


@pytest.fixture
def float_drawn(tmp_path):
    """One field whose intensities are floats, which Measure refuses.

    The refusal passes :meth:`FieldTable.problems` -- nothing about the
    filenames is wrong -- and only fires once the run has started, which is
    what makes it the right way to reach the failure path.
    """
    folder = tmp_path / "floaty"
    yy, xx = np.indices((24, 24))
    _write(str(folder / "fov001_C1.tif"),
           ((yy + xx) / 7.0).astype(np.float32))
    _write(str(folder / "fov001_C2.tif"),
           ((yy - xx) / 7.0).astype(np.float32))
    cell = np.zeros((24, 24), np.uint16)
    cell[3:21, 3:21] = 1
    _write(str(folder / "fov001_cell_mask.tif"), cell)
    return folder


def test_a_failed_run_says_why_and_gives_the_window_back(qtbot, float_drawn):
    """Press Measure on files Measure refuses: the reason is shown, unthreaded.

    THE WINDOW USED TO DIE SILENTLY HERE. ``run`` disabled the Measure
    button and only ``_on_done`` re-enabled it, and ``JobRunner`` calls
    ``on_done`` only for a job that succeeded -- so a refusal left the button
    disabled for the rest of the session with nothing in the log but
    "Writing...". The explanation went to ``job_failed``, which nothing was
    listening to.
    """
    screen = MeasureInputsScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.inputs.dropEvent(_drop_event(_files(float_drawn)))
    screen.apply_settings_dict(dict(LEAN))

    assert screen.inputs.problems() == []
    assert screen.run_button.isEnabled()

    with qtbot.waitSignal(screen.run_finished, timeout=5000) as caught:
        screen.run_button.click()

    assert caught.args == [None]
    assert screen.run_button.isEnabled(), (
        "a refusal must not leave the window dead for the rest of the session")
    assert screen.inputs.isEnabled()
    log = screen._log.toPlainText()
    assert "The run stopped:" in log
    assert "floating-point" in log or "8- or 16-bit" in log
    assert screen.result() is None


def test_a_failed_run_on_the_real_worker_thread_also_reports(qtbot,
                                                             float_drawn):
    """The same, through the threaded runner -- the path a user actually uses.

    The unthreaded and threaded failure routes are different code in
    ``JobRunner`` (``submit``'s own ``except`` against ``_on_settled``'s
    ``if ok``), and only the threaded one runs in the application. Neither
    was exercised before, which is why the dead button went unnoticed.
    """
    screen = MeasureInputsScreen(threaded=True)
    qtbot.addWidget(screen)
    screen.inputs.add_paths(_files(float_drawn))
    screen.apply_settings_dict(dict(LEAN))

    assert screen.run_button.isEnabled()
    with qtbot.waitSignal(screen.run_finished, timeout=20000) as caught:
        screen.run_button.click()

    assert caught.args == [None]
    assert screen.run_button.isEnabled()
    assert "The run stopped:" in screen._log.toPlainText()
    screen.close()


def test_the_table_is_locked_while_a_run_reads_it(qtbot, drawn, tmp_path):
    """The grid cannot be edited under the worker that is reading it.

    ``run`` hands the measurement a SNAPSHOT, and the table is disabled for
    the duration as well. Before both, the worker iterated the live model
    while the user was free to drop the next batch on it -- a ``KeyError``
    inside the write, or a field nobody asked for, on a run that takes
    minutes and during which preparing the next batch is the natural thing
    to do.
    """
    screen = MeasureInputsScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.inputs.dropEvent(_drop_event(_files(drawn)))
    screen.set_destination(str(tmp_path / "out"))
    screen.apply_settings_dict(dict(LEAN))

    seen = {}

    def peek(_message):
        """Look at the window while the worker is between its two stages."""
        seen.setdefault('inputs_enabled', screen.inputs.isEnabled())
        seen.setdefault('run_enabled', screen.run_button.isEnabled())
        screen.inputs.add_field()

    screen.progress.connect(peek)
    with qtbot.waitSignal(screen.run_finished, timeout=60000):
        screen.run_button.click()

    assert seen['inputs_enabled'] is False, "the grid was editable mid-run"
    assert seen['run_enabled'] is False
    assert screen.inputs.isEnabled()
    assert screen.result() is not None
    assert len(screen.result()['stems']) == 2, (
        "the row added mid-run must not have reached the snapshot")


def test_dropping_a_folder_never_stats_it_on_the_gui_thread(qtbot, drawn,
                                                            monkeypatch):
    """The walk runs on a worker, which is the whole of the fix.

    A drop is a path the user chose; on a microscope rig that is the share
    the images live on, where one stat under a sleeping automount was
    measured at over twenty seconds. Done inline it froze the application
    with no traceback. Replacing the walk and recording the thread it ran on
    is how ``import_workbench`` reproduces the same freeze.
    """
    import threading

    from spacr.qt.widgets import measure_input_table as module

    gui_thread = threading.current_thread()
    ran_on = {}
    real = module.files_under

    def watched(paths):
        ran_on['thread'] = threading.current_thread()
        return real(paths)

    monkeypatch.setattr(module, 'files_under', watched)

    widget = MeasureInputTable(threaded=True)
    qtbot.addWidget(widget)
    widget.dropEvent(_drop_event([drawn]))

    assert widget.table().rows == []
    qtbot.waitUntil(lambda: not widget.is_scanning(), timeout=5000)

    assert ran_on['thread'] is not gui_thread
    assert [row.label for row in widget.table().rows] == ["fov001", "fov002"]


def test_a_drag_over_the_table_asks_the_filesystem_nothing(qtbot, drawn,
                                                           monkeypatch):
    """Accepting a drag is decided from the mime data alone.

    ``dragEnterEvent`` used to expand the folder to decide whether to accept,
    and ``dragMoveEvent`` calls it -- so the stats ran again for every event
    the pointer produced while merely hovering over the table.
    """
    from spacr.qt.widgets import measure_input_table as module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a drag-over expanded the dropped folder")

    monkeypatch.setattr(module, 'files_under', forbidden)

    widget = MeasureInputTable(threaded=True)
    qtbot.addWidget(widget)

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(drawn))])
    _MIME_KEPT_ALIVE.append(mime)
    enter = QDragEnterEvent(QPointF(4, 4).toPoint(), Qt.CopyAction, mime,
                            Qt.LeftButton, Qt.NoModifier)
    for _ in range(5):
        widget.dragEnterEvent(enter)
        assert enter.isAccepted()


def test_src_shows_where_the_run_will_write_without_a_destination(qtbot,
                                                                  drawn):
    """The disabled ``src`` box names the folder the results land in.

    It is captioned "the file table decides this", and it showed the settings
    spec's ``path`` placeholder whenever the window was opened without a
    folder -- while the run wrote to ``<the files>/features``.
    """
    screen = MeasureInputsScreen(threaded=False)
    qtbot.addWidget(screen)
    assert screen.destination() is None

    screen.inputs.dropEvent(_drop_event(_files(drawn)))

    expected = os.path.join(str(drawn), "features", "merged")
    assert screen.derived_settings()['src'] == expected
    widget = screen._decided_widgets['src']
    assert not widget.isEnabled()
    assert expected in _widget_text(widget)
