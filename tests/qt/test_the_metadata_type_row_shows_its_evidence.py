"""Mask's microscope-convention row: grouped, exampled, and testable.

A user whose instrument is not a Yokogawa used to meet a four-entry dropdown
that did not name it, and the only feedback on a wrong choice was a warning
printed per unreadable filename during a run that then produced a wrong
measurements database rather than an empty one.

What this file holds down:

* the menu carries a DISABLED heading per vendor, so it can be scanned by
  the name on the microscope rather than by the key a settings file stores;
* the stored value is still the key -- the caption is grouped and indented
  and a settings CSV must not learn about that;
* the example filename is under the menu, and a provisional convention says
  so on the same line;
* 'Test on my folder' reports a count AND the first name that did not
  parse, and 'Which convention fits?' ranks without assigning;
* the scan is off the GUI thread, and the thread outlives the widget that
  started it rather than crashing with it.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt


@pytest.fixture
def plate(tmp_path):
    """A small Opera Phenix folder, plus one name nothing parses."""
    folder = tmp_path / "raw"
    folder.mkdir()
    for name in ("r01c01f01p01-ch1sk1fk1fl1.tiff",
                 "r01c01f01p01-ch2sk1fk1fl1.tiff",
                 "r02c02f01p01-ch1sk1fk1fl1.tiff",
                 "an_unparsable_name.tiff",
                 "notes.txt"):
        (folder / name).write_bytes(b"0")
    return folder


def _field(qapp, default, folder, threaded=False, custom=None):
    """One row, wired to ``folder``, scanning inline unless asked otherwise."""
    from spacr.qt.screens.settings_model import _MetadataTypeField

    widget = _MetadataTypeField(default=default,
                                source_folder=lambda: str(folder),
                                custom_regex=lambda: custom)
    widget.set_threaded(threaded)
    return widget


def test_the_menu_has_a_heading_for_every_vendor_and_they_are_not_choosable(
        qapp, plate):
    """A heading a user can select would store a vendor name as a setting."""
    from spacr.regex_infer import _metadata_convention_menu

    field = _field(qapp, "cellvoyager", plate)
    model = field.combo.model()
    headings = [field.combo.itemText(i) for i in range(field.combo.count())
                if not bool(model.item(i).flags() & Qt.ItemIsEnabled)]
    assert headings == [vendor for vendor, _rows
                        in _metadata_convention_menu()]
    for index in range(field.combo.count()):
        enabled = bool(model.item(index).flags() & Qt.ItemIsEnabled)
        stored = field.combo.itemData(index)
        assert enabled == (stored is not None), (
            "a heading must carry no stored value, and a convention must")


def test_the_row_stores_the_key_however_the_caption_is_dressed(qapp, plate):
    """The caption is indented and may say 'provisional'; the value may not."""
    field = _field(qapp, "cellvoyager", plate)
    assert field.get_value() == "cellvoyager"
    assert field.text() == "cellvoyager"

    field.set_value("opera_phenix")
    assert field.get_value() == "opera_phenix"
    assert "opera_phenix" not in field.combo.currentText()

    field.setText("cq1")
    assert field.get_value() == "cq1"


def test_an_unknown_stored_value_is_left_alone_rather_than_corrected(
        qapp, plate):
    """A settings CSV with a typo must not silently run a different parser."""
    field = _field(qapp, "cellvoyager", plate)
    field.set_value("CellVoyager")
    assert field.get_value() == "cellvoyager", (
        "an unrecognised name leaves the selection where it was; "
        "_get_regex is what refuses it, by name, at run time")


def test_the_example_filename_is_shown_and_provisional_says_so(qapp, plate):
    """The one thing a user can check in a second."""
    field = _field(qapp, "cellvoyager", plate)
    assert "AssayPlate_Greiner" in field.example.text()
    assert "reconstructed" not in field.example.text()

    field.set_value("evos")
    assert "scan_R_p2_z1_0_B03f01d0.tif" in field.example.text()
    assert "reconstructed" in field.example.text(), (
        "a provisional convention must say so where the choice is made")


def test_testing_the_folder_reports_the_count_and_the_first_failure(
        qapp, plate):
    """The readout that turns a wrong guess into a question, before the run."""
    field = _field(qapp, "opera_phenix", plate)
    field.test_on_the_folder()
    text = field.report.text()
    assert "3 of 4" in text
    assert "an_unparsable_name.tiff" in text
    assert "notes.txt" not in text, "only image files are counted"

    field.set_value("cq1")
    assert not field.report.isVisible(), (
        "a stale readout beside a new choice is worse than no readout")
    field.test_on_the_folder()
    assert "0 of 4" in field.report.text()


def test_the_first_failure_is_the_same_one_on_every_machine(qapp, tmp_path):
    """os.scandir returns directory order; a bug report needs one answer."""
    folder = tmp_path / "raw"
    folder.mkdir()
    for name in ("zzz_last.tif", "aaa_first.tif"):
        (folder / name).write_bytes(b"0")
    field = _field(qapp, "cellvoyager", folder)
    field.test_on_the_folder()
    assert "aaa_first.tif" in field.report.text()


def test_the_ranking_is_offered_and_changes_nothing(qapp, plate):
    """Applying a convention the user did not choose is the same defect."""
    field = _field(qapp, "cq1", plate)
    field.rank_the_conventions()
    text = field.report.text()
    assert "Opera Phenix" in text
    assert "opera_phenix" in text
    assert "3 of 4" in text
    assert field.get_value() == "cq1", "the offer must not apply itself"


def test_a_folder_with_nothing_in_it_says_so_rather_than_reporting_zero(
        qapp, tmp_path):
    """Zero of zero reads as a broken convention; it is a broken path."""
    empty = tmp_path / "empty"
    empty.mkdir()
    field = _field(qapp, "cellvoyager", empty)
    field.test_on_the_folder()
    assert "No image files" in field.report.text()

    from spacr.qt.screens.settings_model import _MetadataTypeField
    unset = _MetadataTypeField(default="cellvoyager",
                               source_folder=lambda: "",
                               custom_regex=lambda: None)
    unset.set_threaded(False)
    unset.test_on_the_folder()
    assert "source folder" in unset.report.text()


def test_the_originals_are_tested_when_spacr_has_already_set_them_aside(
        qapp, tmp_path):
    """A plate run once has spaCR's own names on top of the microscope's."""
    folder = tmp_path / "plate"
    (folder / "orig").mkdir(parents=True)
    (folder / "plate1_A01_T0001F001L01C01.tif").write_bytes(b"0")
    for name in ("r01c01f01p01-ch1sk1fk1fl1.tiff",
                 "r01c01f01p01-ch2sk1fk1fl1.tiff"):
        (folder / "orig" / name).write_bytes(b"0")
    field = _field(qapp, "opera_phenix", folder)
    field.test_on_the_folder()
    assert "All 2" in field.report.text()


def test_the_scan_runs_off_the_gui_thread_and_lets_go_of_it(qapp, plate):
    """A directory listing over a 70,000-file plate cannot block the panel."""
    import time

    from spacr.qt.screens import settings_model as sm

    before = len(sm._LIVE_FOLDER_SCANS)
    field = _field(qapp, "opera_phenix", plate, threaded=True)
    field.test_on_the_folder()
    assert not field.test_button.isEnabled(), (
        "a second scan must not be startable while one runs")
    for _ in range(600):
        qapp.processEvents()
        if field._thread is None and "3 of 4" in field.report.text():
            break
        time.sleep(0.005)
    assert "3 of 4" in field.report.text()
    assert field.test_button.isEnabled()
    assert len(sm._LIVE_FOLDER_SCANS) == before, (
        "a finished scan must be released; a leaked QThread is a crash later")


def test_a_scan_survives_the_widget_that_started_it(qapp, plate):
    """Switching away from Mask mid-scan must not take the process down."""
    import gc
    import time

    from spacr.qt.screens import settings_model as sm

    before = len(sm._LIVE_FOLDER_SCANS)
    field = _field(qapp, "opera_phenix", plate, threaded=True)
    field.rank_the_conventions()
    del field
    gc.collect()
    for _ in range(600):
        qapp.processEvents()
        if len(sm._LIVE_FOLDER_SCANS) == before:
            break
        time.sleep(0.005)
    assert len(sm._LIVE_FOLDER_SCANS) == before


def test_the_row_announces_a_change_so_custom_regex_can_grey_itself(
        qapp, plate):
    """`custom_regex` is gated on this setting and follows `value_changed`."""
    from spacr.qt.screens.settings_model import _VALUE_CHANGED_SIGNALS

    field = _field(qapp, "cellvoyager", plate)
    assert _VALUE_CHANGED_SIGNALS[0] == "value_changed"
    seen = []
    field.value_changed.connect(lambda: seen.append(field.get_value()))
    field.set_value("custom")
    assert seen == ["custom"]
