"""The sharing form submits the user's scorecard and optional data folder."""
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPushButton

from spacr.qt.widgets import model_share_dialog


def test_scorecard_defaults_and_edited_values_are_returned(qtbot):
    dialog = model_share_dialog.ShareDialog("weights.bin")
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Share weights.bin"
    values = dialog.values()
    assert values["kind"] == "cellpose"
    assert values["cv"] == "no"
    assert values["f1"] == ""
    assert values["train_data_dir"] == ""
    dialog._edits["display_name"].setText("  Plate A  ")
    dialog._edits["f1"].setText(" 0.81 ")
    assert dialog.values()["display_name"] == "Plate A"
    assert dialog.values()["f1"] == "0.81"


def test_folder_picker_keeps_previous_attachment_on_cancel(qtbot, monkeypatch, tmp_path):
    dialog = model_share_dialog.ShareDialog("weights.bin")
    qtbot.addWidget(dialog)
    answers = iter([str(tmp_path), ""])
    calls = []
    def choose(parent, title, initial):
        calls.append(initial)
        return next(answers)
    monkeypatch.setattr(model_share_dialog.QFileDialog, "getExistingDirectory", choose)
    button = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Train data")
    assert dialog.train_data_dir() == ""
    qtbot.mouseClick(button, Qt.LeftButton)
    assert dialog.train_data_dir() == str(tmp_path)
    assert dialog.values()["train_data_dir"] == str(tmp_path)
    assert str(tmp_path) in dialog._train_label.text()
    qtbot.mouseClick(button, Qt.LeftButton)
    assert dialog.train_data_dir() == str(tmp_path)
    assert calls == ["", str(tmp_path)]


@pytest.mark.parametrize("button,result", [
    (QDialogButtonBox.Ok, QDialog.Accepted),
    (QDialogButtonBox.Cancel, QDialog.Rejected),
])
def test_scorecard_buttons_finish_the_dialog(qtbot, button, result):
    dialog = model_share_dialog.ShareDialog("weights.bin")
    qtbot.addWidget(dialog)
    dialog.show()
    buttons = dialog.findChild(QDialogButtonBox)
    with qtbot.waitSignal(dialog.finished):
        qtbot.mouseClick(buttons.button(button), Qt.LeftButton)
    assert dialog.result() == result
