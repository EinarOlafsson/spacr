"""PDF drops open the chosen Live mode with pixels from that document."""
from pathlib import Path
import threading

import numpy as np
import pytest
from PIL import Image
from pypdf import PdfReader

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QMessageBox

from spacr.qt import dnd_handlers as dh
from spacr.qt.widgets import plaque_preview as ppv


@pytest.fixture
def screen(qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt.screens.app_screen import AppScreen

    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a, **k: [])
    made = AppScreen("analyze_plaques")
    qtbot.addWidget(made)
    return made


def _answer(monkeypatch, wanted):
    questions = []

    def choose(box):
        questions.append(box.text())
        if wanted is None:
            box.reject()
        else:
            button = next(button for button in box.buttons()
                          if button.property("plaque_mode") == wanted)
            button.click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", choose)
    return questions


@pytest.mark.parametrize("mode", (ppv.FIGURE_MODE, ppv.PLAQUE_MODE))
def test_a_pdf_mode_choice_opens_live_and_keeps_the_new_document(
        screen, qtbot, monkeypatch, tmp_path, mode):
    from spacr.plaque_papers import PAPER_FILE

    previous = tmp_path / "previous.png"
    Image.new("RGB", (13, 11), (180, 20, 10)).save(previous)
    panel = screen._live_preview
    screen._settings_model.set_value_for_key("src", str(previous))
    assert panel.load_source_async(str(previous))
    qtbot.waitUntil(lambda: panel.current_path() == previous
                    and panel._view.array() is not None, timeout=10000)
    assert panel._view.array().shape[:2] == (11, 13)
    assert not screen._preview_switch.isChecked()

    pdf = tmp_path / "new-document.pdf"
    Image.new("RGB", (48, 32), (20, 60, 200)).save(pdf, format="PDF")
    expected = np.array(PdfReader(str(pdf)).pages[0].images[0].image.convert("RGB"))
    entered, release = threading.Event(), threading.Event()
    references = []
    original_fetch = panel.fetch_paper

    def extract(reference, destination):
        references.append(Path(reference))
        document = PdfReader(str(reference))
        image = document.pages[0].images[0].image.convert("RGB")
        entered.set()
        if not release.wait(timeout=10):
            raise RuntimeError("the PDF test did not release its reader")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        image.save(destination / "figure_1.png")
        (destination / PAPER_FILE).write_text("{}", encoding="utf-8")
        return {"folder": str(destination), "paper": Path(reference).name,
                "figures": 1, "with_legend": 0}

    def fetch(reference, parent, **kwargs):
        kwargs.update(fetch=extract, offer_install=False)
        return original_fetch(reference, parent, **kwargs)

    monkeypatch.setattr(panel, "fetch_paper", fetch)
    questions = _answer(monkeypatch, mode)
    try:
        assert dh.PlaqueDropHandler().apply_all([pdf], screen)
        qtbot.waitUntil(entered.is_set, timeout=10000)
        assert screen._preview_switch.isChecked()
        assert screen._live_preview_card.isVisibleTo(screen)
        assert panel.mode() == screen._plaque_mode_switch.mode() == mode
        assert panel.current_path() is None
        assert panel._view.array() is None
    finally:
        release.set()
    expected_path = tmp_path / ppv.paper_folder_name(pdf) / "figure_1.png"
    qtbot.waitUntil(lambda: panel.current_path() == expected_path
                    and panel._view.array() is not None
                    and panel._view.array().shape == expected.shape, timeout=10000)
    assert references == [pdf]
    assert len(questions) == 1
    assert np.array_equal(panel._view.array(), expected)
    assert panel.mode() == screen._plaque_mode_switch.mode() == mode
    assert Path(screen._settings_model._widgets["src"].text()) == expected_path.parent


def test_a_pdf_batch_opens_live_in_plaque_mode_with_both_documents(
        screen, qtbot, monkeypatch, tmp_path):
    from spacr.plaque_papers import PAPER_FILE

    pdfs = [tmp_path / "first.pdf", tmp_path / "second.pdf"]
    for pdf, colour in zip(pdfs, ((20, 160, 40), (180, 40, 80))):
        Image.new("RGB", (36, 24), colour).save(pdf, format="PDF")
    panel = screen._live_preview
    original_fetch = panel.fetch_papers
    expected = {}

    def extract(reference, destination):
        image = PdfReader(str(reference)).pages[0].images[0].image.convert("RGB")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "figure_1.png"
        image.save(output)
        expected[output] = np.array(image)
        (destination / PAPER_FILE).write_text("{}", encoding="utf-8")
        return {"folder": str(destination), "paper": Path(reference).name,
                "figures": 1, "with_legend": 0}

    def fetch(paths, parent, **kwargs):
        kwargs.update(fetch=extract, offer_install=False)
        return original_fetch(paths, parent, **kwargs)

    monkeypatch.setattr(panel, "fetch_papers", fetch)
    questions = _answer(monkeypatch, ppv.PLAQUE_MODE)
    assert dh.PlaqueDropHandler().apply_all(pdfs, screen)
    qtbot.waitUntil(lambda: len(expected) == 2 and len(panel._paths) == 2
                    and panel._view.array() is not None, timeout=10000)
    assert screen._preview_switch.isChecked()
    assert screen._live_preview_card.isVisibleTo(screen)
    assert len(questions) == 1
    assert panel.mode() == screen._plaque_mode_switch.mode() == ppv.PLAQUE_MODE
    assert set(panel._paths) == set(expected)
    for index, path in enumerate(panel._paths):
        panel._picker.setCurrentIndex(index)
        qtbot.waitUntil(lambda path=path: panel.current_path() == path
                        and panel._view.array() is not None
                        and np.array_equal(panel._view.array(), expected[path]),
                        timeout=10000)


@pytest.mark.parametrize("dismiss", ("window_close", "escape"))
def test_cancelling_the_pdf_mode_question_does_not_open_live(
        screen, qtbot, monkeypatch, tmp_path, dismiss):
    from PySide6.QtCore import Qt

    pdf = tmp_path / "cancelled.pdf"
    Image.new("RGB", (24, 16), (30, 40, 50)).save(pdf, format="PDF")
    panel = screen._live_preview
    fetched = []
    monkeypatch.setattr(panel, "fetch_paper", lambda *a, **k: fetched.append(a))
    dismissed = []

    def cancel(box):
        box.show()
        qtbot.waitExposed(box)
        if dismiss == "window_close":
            assert box.close()
        else:
            qtbot.keyClick(box, Qt.Key_Escape)
        assert not box.isVisible()
        assert box.clickedButton() is box.escapeButton()
        assert not box.clickedButton().property("plaque_mode")
        dismissed.append(dismiss)
        return box.result()

    monkeypatch.setattr(QMessageBox, "exec", cancel)
    source = screen._settings_model._widgets["src"].text()
    assert dh.PlaqueDropHandler().apply_all([pdf], screen)
    assert dismissed == [dismiss]
    assert fetched == []
    assert not screen._preview_switch.isChecked()
    assert not screen._live_preview_card.isVisibleTo(screen)
    assert screen._settings_model._widgets["src"].text() == source
