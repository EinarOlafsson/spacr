"""Where a rejected drop goes when the usual place to say it is missing.

``_report_drop_problem`` has three surfaces in decreasing order of
preference -- a console panel, a standalone tool's read-only log pane, and
the status line -- and the screens that have none of the first two are
exactly the ones a drop is most likely to be rejected on. The CSV router
has the matching shape: the input a file belongs in may not be the widget
type the router can fill, and the file-list widget may not import at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QPlainTextEdit, QWidget       # noqa: E402

from spacr.qt import dnd                                    # noqa: E402

pytestmark = pytest.mark.qt


class _MuteConsole:
    """A console panel with no text-appending surface at all."""

    def __init__(self):
        self.ai = []

    def _current_provider(self):
        return None

    def open_error_flow(self, text, **kwargs):
        self.ai.append(text)


class _RecordingModel:
    """The settings model, reduced to the widget map the router reads."""

    def __init__(self, widgets):
        self._widgets = widgets


class _PlainScreen:
    """A screen with a settings model and nothing else -- no console."""

    def __init__(self, model):
        self._settings_model = model
        self.applied = []

    def apply_settings_dict(self, values):
        self.applied.append(dict(values))
        return len(values)


# --- where the rejection is printed ---------------------------------------

def test_a_console_that_cannot_append_falls_through_to_the_log_pane(
        qtbot, tmp_path):
    """A console with no append method is not a place to print, so skip it."""
    screen = QWidget()
    qtbot.addWidget(screen)
    screen._console = _MuteConsole()
    pane = QPlainTextEdit()
    qtbot.addWidget(pane)
    screen._summary = pane

    dnd._report_drop_problem(screen, tmp_path / "plate1", "not a plate",
                             "choose the plate folder")

    assert "[drop rejected]" in pane.toPlainText()
    assert "not a plate" in pane.toPlainText()
    assert screen._console.ai == []


def test_a_screen_with_nowhere_to_print_still_says_it_on_the_status_line(
        qtbot, tmp_path):
    """No console and no log pane leaves the status line as the last resort."""
    said = []

    class _Screen(QWidget):
        def _set_status(self, text):
            said.append(text)

    screen = _Screen()
    qtbot.addWidget(screen)

    message = dnd._report_drop_problem(
        screen, tmp_path / "plate1", "not a plate",
        "choose the plate folder")

    assert said == ["Drop rejected: not a plate Suggestion: "
                    "choose the plate folder"]
    assert "Reason: not a plate" in message


# --- what the CSV router can and cannot fill ------------------------------

def _score_csv(tmp_path):
    path = tmp_path / "plate1_dv.csv"
    path.write_text("path,pred,plate,row,col\na,0.5,plate1,r1,c1\n")
    return path


def _annotation_csv(tmp_path):
    path = tmp_path / "grna_barcodes.csv"
    path.write_text("name,sequence\nTGGT1_225160_2,ACGT\n")
    return path


def test_a_router_without_the_file_list_widget_takes_nothing(
        qtbot, monkeypatch, tmp_path):
    """The router cannot recognise an input it cannot import the class for."""
    from spacr.qt.widgets.file_list import FilePathListWidget

    widget = FilePathListWidget(kind="table")
    qtbot.addWidget(widget)
    screen = _PlainScreen(_RecordingModel({"score_data": widget}))

    monkeypatch.setitem(sys.modules, "spacr.qt.widgets.file_list", None)

    assert dnd._route_data_csv_to_inputs(_score_csv(tmp_path), screen) is None
    assert widget.get_value() == []


def test_an_annotation_csv_skips_a_metadata_input_it_cannot_fill(
        qtbot, tmp_path):
    """A metadata slot of the wrong widget type is passed over, not used."""
    from spacr.qt.widgets.file_list import FilePathListWidget

    wrong_kind = QPlainTextEdit()
    qtbot.addWidget(wrong_kind)
    score = FilePathListWidget(kind="table")
    qtbot.addWidget(score)
    screen = _PlainScreen(_RecordingModel(
        {"metadata_files": wrong_kind, "score_data": score}))

    annotation = _annotation_csv(tmp_path)

    assert dnd._route_data_csv_to_inputs(annotation, screen) == "score_data"
    assert score.get_value() == [str(annotation)]
    assert wrong_kind.toPlainText() == ""


def test_a_routed_csv_needs_no_console_to_be_accepted(qtbot, tmp_path):
    """A screen without a console still keeps the file it was given."""
    from spacr.qt.widgets.file_list import FilePathListWidget

    widget = FilePathListWidget(kind="table")
    qtbot.addWidget(widget)
    screen = _PlainScreen(_RecordingModel({"score_data": widget}))
    assert not hasattr(screen, "_console")

    score = _score_csv(tmp_path)
    dnd._apply_settings_csv(score, screen)

    assert widget.get_value() == [str(score)]
    assert screen.applied == []


def test_a_settings_csv_that_does_not_parse_to_a_mapping_is_not_applied(
        qtbot, monkeypatch, tmp_path):
    """Only a mapping can become settings; anything else is left alone."""
    import spacr.utils

    settings_csv = tmp_path / "settings.csv"
    settings_csv.write_text("Key,Value\nsrc,/tmp\n")

    monkeypatch.setattr(spacr.utils, "load_settings",
                        lambda *args, **kwargs: ["src", "/tmp"])

    screen = _PlainScreen(_RecordingModel({}))
    dnd._apply_settings_csv(settings_csv, screen)

    assert screen.applied == []
