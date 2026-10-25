"""Mask-role labels translate while organizer decisions retain canonical IDs."""
import sys
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from spacr.qt.i18n import tr
from spacr.qt.widgets import organize_for_measure as ofm


pytestmark = pytest.mark.qt


@pytest.fixture(params=[(language, role) for language in ("sv", "fr")
                        for role in ("cell", "nucleus", "pathogen", "organelle")])
def organizer(request, qtbot, qt_theme_applied, monkeypatch):
    language, role = request.param
    monkeypatch.setenv("SPACR_LANGUAGE", language)
    dialog = ofm._dialog_for("import")()
    qtbot.addWidget(dialog)
    dialog.add_column("channel")
    dialog.add_column("mask", role=role, of_channel=1)
    yield dialog, language, role
    dialog._stop_thumbs()


def test_real_table_header_translates_the_role_and_keeps_its_id(organizer):
    dialog, language, role = organizer
    expected = tr("{role} mask (channel {n})", language,
                  role=tr(role, language), n=1)
    assert dialog.table.model().headerData(
        1, Qt.Horizontal, Qt.DisplayRole) == expected
    assert dialog.columns[1].role == role


def test_real_marker_question_translates_the_role_but_returns_its_id(
        organizer, monkeypatch):
    dialog, language, role = organizer
    seen = {}

    def inspect_question(box):
        # dialogs._set_question: on macOS the question is the text and what
        # the box says about it leads the informative text.
        seen["text"] = (box.informativeText().split("\n\n", 1)[0]
                        if sys.platform == "darwin" else box.text())
        seen["keep"] = next(button for button in box.buttons()
                            if box.buttonRole(button) == QMessageBox.AcceptRole)
        seen["caption"] = seen["keep"].text()
        return 0

    monkeypatch.setattr(ofm, "_headless", lambda: False)
    monkeypatch.setattr(QMessageBox, "exec", inspect_question)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: seen["keep"])
    label = ("mask", 1, role)
    assert dialog._ask_marker("seg", label) == label
    kind = tr("the {role} mask of channel {n}", language,
              role=tr(role, language), n=1)
    assert seen["text"] == tr(
        "Images whose names have “{marker}” here look like {kind}. "
        "Does this part of the name mark a channel or a mask, or is it "
        "only part of the field's name?", language, marker="seg", kind=kind)
    assert seen["caption"] == tr("It marks {kind}", language, kind=kind)


def test_invalid_channel_problem_translates_role_without_changing_plan_ids(
        organizer, monkeypatch, tmp_path):
    dialog, language, role = organizer
    image = str(tmp_path / "field.tif")
    mask = str(tmp_path / "masks" / "field.tif")
    dialog.rows = [[image, mask]]
    dialog.row_keys = [None]
    submitted = []

    def capture_plan(folder, sets, **kwargs):
        submitted.append(kwargs)
        return SimpleNamespace(problems=[])

    monkeypatch.setattr(ofm.cs, "build_plan", capture_plan)
    assert dialog.prepare_plan().problems == []
    assert submitted[-1]["mask_roles"] == {1: role}
    assert submitted[-1]["masks"] == {image: mask}
    dialog.columns[1].of_channel = 99
    plan = dialog.prepare_plan()
    assert plan.problems == [tr(
        "A {role} mask column belongs to channel {n}, which there is not.",
        language, role=tr(role, language), n=99)]
    assert submitted[-1]["mask_roles"] == {}
    assert dialog.columns[1].role == role
