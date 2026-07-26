"""Tests for :class:`spacr.qt.regex_editor.RegexEditorDialog`.

The dialog is the last line of defence when spaCR cannot parse a dropped
folder, so these drive the real build → preview → validate → save loop
against real Yokogawa/CellVoyager filenames and against malformed ones,
asserting on the text the user actually sees.

Nothing here calls ``exec()`` — the dialog is built, poked and read
synchronously, offscreen.
"""
from __future__ import annotations

import re

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox

from spacr.qt import regex_detect as rd
from spacr.qt.regex_editor import RegexEditorDialog


# Real Yokogawa CV7000/CV8000 exports.
YOKO = [
    "plate1_A01_T0001F001L01A01Z01C01.tif",
    "plate1_A01_T0001F001L01A01Z01C02.tif",
    "plate1_A01_T0001F002L01A01Z01C01.tif",
    "plate1_B03_T0001F002L01A01Z01C02.tif",
]

# Things that turn up in the same folder and must NOT be parsed.
MALFORMED = [
    "plate1_A1_T0001F001L01A01Z01C01.tif",     # well "A1", not "A01"
    "plate1_A01_T0001F001L01A01Z01C01.txt",    # not an image
    "MeasurementData.mlf",                      # Yokogawa sidecar
    "plate1_A01.tif",                           # truncated name
]


@pytest.fixture
def make_dialog(qtbot):
    """Build a dialog, register it with qtbot, return it."""
    def _make(samples=YOKO, initial_regex="", multichannel=True):
        dlg = RegexEditorDialog(list(samples), initial_regex=initial_regex,
                                multichannel=multichannel)
        qtbot.addWidget(dlg)
        return dlg
    return _make


def _preview(dlg):
    return dlg._preview.toPlainText()


def _save_button(dlg):
    box = dlg.findChild(QDialogButtonBox)
    return box, box.button(QDialogButtonBox.Save)


# ---------------------------------------------------------------------------
# Construction + auto-detect on open
# ---------------------------------------------------------------------------

def test_opening_without_a_regex_autodetects_and_previews(make_dialog):
    dlg = make_dialog()
    assert dlg._regex_input.text() == rd.CELLVOYAGER
    assert dlg._preset_combo.currentText() == "cellvoyager"
    assert dlg.regex == "", "regex is only set once Save is pressed"

    body = _preview(dlg)
    # Header naming every group the pattern captures...
    header = body.splitlines()[0].split()
    assert header == ["plateID", "wellID", "fieldID", "chanID", "timeID",
                      "sliceID", "laserID", "AID", "filename"]
    # ...and the real parsed values for every sampled file.
    assert "plate1" in body and "A01" in body and "B03" in body
    for name in YOKO:
        assert name in body
    assert "[auto] chose regex `cellvoyager` — matched 4/4" in body
    assert "did NOT match" not in body


def test_opening_without_a_regex_reports_all_fields_captured(make_dialog):
    dlg = make_dialog()
    assert "All required fields" in dlg._warnings_lbl.text()
    assert "Warnings" not in dlg._warnings_lbl.text()


def test_opening_with_an_initial_regex_skips_auto_detect(make_dialog):
    dlg = make_dialog(initial_regex=rd.YOKOGAWA)
    assert dlg._regex_input.text() == rd.YOKOGAWA
    assert dlg._preset_combo.currentText() == "yokogawa"
    assert "[auto]" not in _preview(dlg)
    # Preview was still built from the initial pattern.
    assert "plate1_A01_T0001F001L01A01Z01C01.tif" in _preview(dlg)


def test_only_the_first_twenty_filenames_are_sampled(make_dialog):
    names = [f"plate1_A01_T0001F{i:03d}L01A01Z01C01.tif" for i in range(50)]
    dlg = make_dialog(samples=names)
    assert len(dlg._samples) == 20
    assert dlg._samples == names[:20]
    assert "matched 20/20" in _preview(dlg)


def test_empty_folder_reports_that_nothing_could_be_inferred(make_dialog):
    dlg = make_dialog(samples=[])
    assert dlg._regex_input.text() == ""
    body = _preview(dlg)
    assert "[auto] no regex could be inferred from the sample." in body
    assert "(no records" in body
    # BUG (fixed): this branch used to leave the status label blank, so a
    # user with an unreadable folder saw no diagnosis at all.
    assert dlg._warnings_lbl.text() != ""
    assert "No filenames matched" in dlg._warnings_lbl.text()


# ---------------------------------------------------------------------------
# Typing a regex
# ---------------------------------------------------------------------------

def test_typing_a_custom_regex_reparses_and_switches_the_preset_to_custom(
        make_dialog):
    dlg = make_dialog()
    dlg._regex_input.setText(
        r"(?P<plateID>[^_]+)_(?P<wellID>[A-Z]\d{2})_.*C(?P<chanID>\d{2})\.tif$"
    )
    assert dlg._preset_combo.currentText() == "(custom)"
    assert dlg._preset_combo.currentData() is None

    body = _preview(dlg)
    header = body.splitlines()[0].split()
    assert header == ["plateID", "wellID", "chanID", "filename"]
    assert "timeID" not in body
    assert "All required fields" in dlg._warnings_lbl.text()


def test_typing_a_builtin_pattern_snaps_the_dropdown_onto_it(make_dialog):
    dlg = make_dialog()
    dlg._regex_input.setText(rd.CQ1)
    assert dlg._preset_combo.currentText() == "cq1"
    assert dlg._preset_combo.currentData() == "cq1"


def test_a_regex_without_a_channel_group_warns_for_multichannel_data(
        make_dialog):
    dlg = make_dialog()
    dlg._regex_input.setText(r"(?P<plateID>[^_]+)_(?P<wellID>[A-Z]\d{2})_.*")
    warn = dlg._warnings_lbl.text()
    assert "Warnings" in warn
    assert "chanID" in warn
    assert "location" not in warn.lower()   # wellID was captured


def test_the_same_regex_is_accepted_for_single_channel_data(make_dialog):
    pattern = r"(?P<plateID>[^_]+)_(?P<wellID>[A-Z]\d{2})_T\d+F(?P<fieldID>\d+).*"
    multi = make_dialog(initial_regex=pattern, multichannel=True)
    single = make_dialog(initial_regex=pattern, multichannel=False)
    assert "chanID" in multi._warnings_lbl.text()
    assert "chanID" not in single._warnings_lbl.text()
    assert "All required fields" in single._warnings_lbl.text()


def test_a_regex_with_no_location_group_warns_about_wells_and_fields(
        make_dialog):
    dlg = make_dialog()
    dlg._regex_input.setText(r".*C(?P<chanID>\d{2})\.tif$")
    warn = dlg._warnings_lbl.text()
    assert "wellID or" in warn and "fieldID" in warn
    assert "plateID" in warn        # soft warning fires too
    assert "chanID" not in warn


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------

def test_an_uncompilable_regex_is_reported_not_raised(make_dialog):
    dlg = make_dialog()
    dlg._regex_input.setText(r"(?P<plateID>")      # unterminated group
    assert "Warnings" in dlg._warnings_lbl.text()
    assert "No filenames matched the regex." in dlg._warnings_lbl.text()
    body = _preview(dlg)
    assert "(no records — regex did not match any files)" in body
    assert f"[4 filenames did NOT match — first: {YOKO[0]}]" in body


def test_unmatched_filenames_are_counted_and_the_first_is_named(make_dialog):
    dlg = make_dialog(samples=MALFORMED + YOKO,
                      initial_regex=rd.YOKOGAWA)
    body = _preview(dlg)
    # Exactly the four YOKO names parse; all four MALFORMED ones do not.
    assert f"[4 filenames did NOT match — first: {MALFORMED[0]}]" in body
    for name in YOKO:
        assert name in body
    for name in MALFORMED:
        # the malformed names appear only in the trailing note, never as
        # a parsed row
        assert body.count(name) == (1 if name == MALFORMED[0] else 0)
    assert "All required fields" in dlg._warnings_lbl.text()


def test_a_regex_that_matches_nothing_says_so_in_both_places(make_dialog):
    # CQ1 needs a leading "W"; Yokogawa plate exports never have one.
    dlg = make_dialog(initial_regex=rd.CQ1)
    assert "No filenames matched the regex." in dlg._warnings_lbl.text()
    body = _preview(dlg)
    assert "(no records" in body
    assert f"[4 filenames did NOT match — first: {YOKO[0]}]" in body


# ---------------------------------------------------------------------------
# Preset dropdown
# ---------------------------------------------------------------------------

def _preset_index(dlg, label):
    return dlg._preset_combo.findData(label)


def test_every_builtin_plus_custom_is_offered(make_dialog):
    dlg = make_dialog()
    labels = [dlg._preset_combo.itemText(i)
              for i in range(dlg._preset_combo.count())]
    assert labels == list(rd.BUILTIN_REGEXES) + ["(custom)"]
    assert dlg._preset_combo.itemData(dlg._preset_combo.count() - 1) is None


def test_picking_a_preset_fills_the_box_and_rebuilds_the_preview(make_dialog):
    dlg = make_dialog()
    dlg._preset_combo.setCurrentIndex(_preset_index(dlg, "yokogawa"))
    assert dlg._regex_input.text() == rd.YOKOGAWA
    body = _preview(dlg)
    header = body.splitlines()[0].split()
    assert header == ["plateID", "wellID", "fieldID", "chanID", "timeID",
                      "sliceID", "laserID", "AID", "filename"]
    assert "did NOT match" not in body
    assert "[auto]" not in body, "picking a preset must not re-run auto-detect"


def test_picking_a_preset_that_does_not_fit_shows_the_failure(make_dialog):
    dlg = make_dialog()
    dlg._preset_combo.setCurrentIndex(_preset_index(dlg, "canonical"))
    assert dlg._regex_input.text() == rd.CANONICAL
    assert "No filenames matched the regex." in dlg._warnings_lbl.text()


def test_selecting_the_custom_entry_leaves_the_regex_alone(make_dialog):
    dlg = make_dialog()
    before = dlg._regex_input.text()
    dlg._preset_combo.setCurrentIndex(dlg._preset_combo.count() - 1)
    assert dlg._preset_combo.currentData() is None
    assert dlg._regex_input.text() == before


# ---------------------------------------------------------------------------
# Auto-detect button
# ---------------------------------------------------------------------------

def test_auto_detect_button_rescues_a_hand_broken_regex(make_dialog, qtbot):
    dlg = make_dialog(initial_regex=r"(?P<nope>zzz)")
    assert "No filenames matched" in dlg._warnings_lbl.text()

    qtbot.mouseClick(dlg._auto_btn, Qt.LeftButton)

    assert dlg._regex_input.text() == rd.CELLVOYAGER
    assert "All required fields" in dlg._warnings_lbl.text()
    body = _preview(dlg)
    assert "[auto] chose regex `cellvoyager` — matched 4/4" in body
    assert YOKO[0] in body


def test_auto_detect_is_idempotent(make_dialog, qtbot):
    """BUG (fixed): QLineEdit stays silent when setText is a no-op, so a
    second click appended another status line to a never-rebuilt table."""
    dlg = make_dialog(initial_regex=rd.CELLVOYAGER)
    qtbot.mouseClick(dlg._auto_btn, Qt.LeftButton)
    once = _preview(dlg)
    qtbot.mouseClick(dlg._auto_btn, Qt.LeftButton)
    qtbot.mouseClick(dlg._auto_btn, Qt.LeftButton)
    assert _preview(dlg) == once
    assert once.count("[auto]") == 1


def test_auto_detect_on_a_folder_it_cannot_read_reports_honestly(make_dialog,
                                                                 qtbot):
    dlg = make_dialog(samples=[])
    qtbot.mouseClick(dlg._auto_btn, Qt.LeftButton)
    assert dlg._regex_input.text() == ""
    assert _preview(dlg).count("[auto] no regex could be inferred") == 1


def test_auto_detect_reports_a_partial_match_truthfully(make_dialog, qtbot):
    dlg = make_dialog(samples=[YOKO[0], "notes.txt"])
    body = _preview(dlg)
    assert "[auto] chose regex `cellvoyager` — matched 1/2" in body
    assert "[1 filenames did NOT match — first: notes.txt]" in body


def test_auto_detect_synthesises_for_an_unknown_naming_scheme(make_dialog):
    names = ["IMG_0001.tif", "IMG_0002.tif", "IMG_0003.tif"]
    dlg = make_dialog(samples=names)
    pattern = dlg._regex_input.text()
    rx = re.compile(pattern)
    assert [n for n in names if rx.match(n)] == names
    assert "matched 3/3" in _preview(dlg)
    assert dlg._preset_combo.currentText() == "(custom)"


# ---------------------------------------------------------------------------
# Save / Cancel
# ---------------------------------------------------------------------------

def test_save_publishes_the_regex_and_accepts(make_dialog, qtbot):
    dlg = make_dialog()
    dlg._regex_input.setText("   " + rd.YOKOGAWA + "  ")
    box, save = _save_button(dlg)
    assert save is not None
    with qtbot.waitSignal(dlg.accepted, timeout=1000):
        qtbot.mouseClick(save, Qt.LeftButton)
    assert dlg.regex == rd.YOKOGAWA      # surrounding whitespace stripped
    assert dlg.result() == QDialog.Accepted
    # The saved pattern is the one the preview was built from.
    assert re.compile(dlg.regex).match(YOKO[0]).group("wellID") == "A01"


def test_cancel_leaves_the_regex_unset(make_dialog, qtbot):
    dlg = make_dialog()
    box, _ = _save_button(dlg)
    cancel = box.button(QDialogButtonBox.Cancel)
    with qtbot.waitSignal(dlg.rejected, timeout=1000):
        qtbot.mouseClick(cancel, Qt.LeftButton)
    assert dlg.regex == ""
    assert dlg.result() == QDialog.Rejected


def test_saving_an_empty_box_yields_an_empty_regex(make_dialog, qtbot):
    """The drop handler treats a falsy regex as "user saved nothing" and
    must not push it into the settings widget."""
    dlg = make_dialog()
    dlg._regex_input.setText("     ")
    _, save = _save_button(dlg)
    qtbot.mouseClick(save, Qt.LeftButton)
    assert dlg.regex == ""
