"""Organizer confirmation uses localized standard buttons and Qt decisions."""
import pytest
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from spacr.qt import i18n
from spacr.qt.widgets import organize_for_measure as ofm


pytestmark = pytest.mark.qt
_REAL_QUESTION = QMessageBox.question
_REAL_MESSAGE_EXEC = QMessageBox.exec


@pytest.fixture
def question_host(qtbot, qapp):
    host = QWidget()
    qtbot.addWidget(host)
    previous = getattr(qapp, "_spacr_dialog_i18n_filter", None)
    i18n.install_dialog_translation(qapp)
    yield host
    if previous is None:
        installed = getattr(qapp, "_spacr_dialog_i18n_filter", None)
        if installed is not None:
            from spacr.qt.gil_priority import _stop_watching_application_events

            _stop_watching_application_events(qapp, installed)
            delattr(qapp, "_spacr_dialog_i18n_filter")
            installed.deleteLater()


def _answer_real_question(host, action, monkeypatch):
    monkeypatch.setattr(ofm, "_headless", lambda: False)
    monkeypatch.setattr(QMessageBox, "question", _REAL_QUESTION)
    monkeypatch.setattr(QMessageBox, "exec", _REAL_MESSAGE_EXEC)
    captured = {}

    def answer():
        box = QApplication.activeModalWidget()
        if (not isinstance(box, QMessageBox) or not box.isVisible()
                or box.parentWidget() is not host
                or box.windowTitle() != "Question title"):
            timer.start(10)
            return
        captured["box"] = box
        captured.update(
            title=box.windowTitle(), text=box.text(), icon=box.icon(),
            standard=box.standardButtons(),
            yes=box.button(QMessageBox.Yes).text(),
            no=box.button(QMessageBox.No).text(),
            default=box.standardButton(box.defaultButton()))
        if action in ("yes", "no"):
            QTest.mouseClick(box.button(
                QMessageBox.Yes if action == "yes" else QMessageBox.No),
                Qt.LeftButton)
        elif action == "enter":
            QTest.keyClick(box, Qt.Key_Return)
        elif action == "escape":
            QTest.keyClick(box, Qt.Key_Escape)
        else:
            box.close()

    timer = QTimer(host)
    timer.setSingleShot(True)
    timer.timeout.connect(answer)
    expired = []

    def stop_unanswered():
        expired.append(True)
        modal = QApplication.activeModalWidget()
        if modal is not None:
            modal.reject()

    deadline = QTimer(host)
    deadline.setSingleShot(True)
    deadline.timeout.connect(stop_unanswered)
    deadline.start(5000)
    timer.start(0)
    try:
        result = ofm.OrganizeForMeasureDialog._ask(
            host, "Question title", "Question body")
    finally:
        deadline.stop()
        timer.stop()
    assert not expired
    assert isinstance(captured.get("box"), QMessageBox)
    assert captured["title"] == "Question title"
    assert captured["text"] == "Question body"
    assert captured["icon"] == QMessageBox.Question
    assert captured["standard"] == QMessageBox.Yes | QMessageBox.No
    return result, captured


@pytest.mark.parametrize("language,yes,no", [
    ("hi", "हाँ", "नहीं"), ("is", "Já", "Nei")])
@pytest.mark.parametrize("action", ["yes", "no"])
def test_real_localized_standard_buttons_keep_yes_no_results(
        question_host, monkeypatch, qapp, language, yes, no, action):
    monkeypatch.setenv("SPACR_LANGUAGE", language)
    assert not i18n.install_qt_translations(qapp, language)
    result, captured = _answer_real_question(question_host, action, monkeypatch)
    assert result is (action == "yes")
    assert captured["yes"].replace("&", "") == yes
    assert captured["no"].replace("&", "") == no


@pytest.mark.parametrize("action,expected", [
    ("enter", True), ("escape", False), ("close", False)])
def test_existing_default_and_cancel_decisions_are_preserved(
        question_host, monkeypatch, action, expected):
    result, captured = _answer_real_question(question_host, action, monkeypatch)
    assert captured["default"] == QMessageBox.NoButton
    assert result is expected


def test_headless_question_still_refuses_without_creating_a_message_box(
        question_host, monkeypatch):
    monkeypatch.setattr(ofm, "_headless", lambda: True)
    before = set(QApplication.topLevelWidgets())
    assert ofm.OrganizeForMeasureDialog._ask(question_host, "Title", "Body") is False
    assert set(QApplication.topLevelWidgets()) == before


def test_question_click_waits_for_modal_after_nested_construction_events(
        question_host, monkeypatch):
    original_init = QMessageBox.__init__
    construction_events = []

    def initialize_with_events(box, *args, **kwargs):
        original_init(box, *args, **kwargs)
        QApplication.processEvents()
        construction_events.append(box.isVisible())

    monkeypatch.setattr(QMessageBox, "__init__", initialize_with_events)
    result, captured = _answer_real_question(question_host, "yes", monkeypatch)
    assert construction_events == [False]
    assert result is True
    assert captured["box"].parentWidget() is question_host
