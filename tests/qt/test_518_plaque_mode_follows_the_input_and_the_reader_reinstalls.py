"""Item 518: Plaque Assay's mode follows what it is given, and the figure
reader is installed, or installed again, from Plaque Assay itself.

* A PDF, or a folder of PDFs, given in Plaque mode asks to switch to Figure
  mode; images given in Figure mode ask to switch to Plaque mode; both
  together say "PDFs are read in Figure mode, images in Plaque mode" and ask
  which to read. For a drop and for a new ``src`` alike.
* A reader installed before spaCR pinned pdfplumber for it is found from its
  install record, and Plaque Assay offers the reinstall in place, saying
  what the install is doing and how it ended.

Offscreen; every dialog is answered by the test.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from spacr import _segmentation_backends as SB
from spacr import plaque_papers as PP
from spacr.qt import dnd_handlers as dh
from spacr.qt.widgets import plaque_preview as ppv


def _png(path: Path, value: int = 90) -> Path:
    """A small real PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.full((24, 24, 3), value, dtype=np.uint8))
    return path


def _pdf(path: Path) -> Path:
    """A PDF-shaped file; nothing here opens it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    return path


class _Answers:
    """Answers each mode question with the next mode, recording the box."""

    def __init__(self, *modes):
        """Hold the answers, in order."""
        self.modes = list(modes)
        self.boxes = []

    def __call__(self, box):
        """Stand in for QMessageBox.exec: click the button for the next mode."""
        self.boxes.append({
            "title": box.windowTitle(), "text": box.text(),
            "info": box.informativeText(),
            "buttons": {b.text(): b.property("plaque_mode")
                        for b in box.buttons()}})
        wanted = self.modes.pop(0)
        for button in box.buttons():
            if button.property("plaque_mode") == wanted:
                button.click()
                break
        return 0


@pytest.fixture
def answer(monkeypatch):
    """``answer('figure', ...)`` sets the answers the questions get."""
    def install(*modes):
        answers = _Answers(*modes)
        monkeypatch.setattr(QMessageBox, "exec", lambda box: answers(box))
        return answers
    return install


@pytest.fixture
def plaque_screen(qtbot, qt_theme_applied, monkeypatch):
    """A real Plaque Assay screen whose PDF reading is recorded, not run."""
    from spacr.qt.screens.app_screen import AppScreen

    made = AppScreen("analyze_plaques")
    qtbot.addWidget(made)
    fetched = []
    monkeypatch.setattr(made._live_preview, "fetch_paper",
                        lambda ref, parent: fetched.append((ref, parent))
                        or True)
    made.fetched = fetched
    return made


def _mode(screen) -> str:
    """The mode Plaque Assay is in."""
    return dh._plaque_mode_of(screen)


def _src(screen) -> str:
    """The form's ``src``."""
    return screen._settings_model._widgets["src"].text()


def _drop(screen, paths):
    """What the drop zone does once the paths are accepted."""
    handler = dh.PlaqueDropHandler()
    accepted = [Path(p) for p in paths if handler.can_accept(Path(p))]
    assert accepted, "the drop was refused before it reached the handler"
    handler.apply_all(accepted, screen)


def _settle(ms: int = 700) -> None:
    """Let the src debounce fire."""
    deadline = time.monotonic() + ms / 1000.0
    while time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)


def test_the_questions_follow_the_input():
    q = ppv.input_mode_question
    assert q(["a.pdf"], [], [], "plaque") == ppv.TO_FIGURE
    assert q([], [], ["paper"], "plaque") == ppv.TO_FIGURE
    assert q([], ["a.png"], [], "figure") == ppv.TO_PLAQUE
    assert q(["a.pdf"], ["a.png"], [], "plaque") == ppv.MIXED
    assert q(["a.pdf"], ["a.png"], [], "figure") == ppv.MIXED
    assert q(["a.pdf"], [], [], "figure") == ""
    assert q([], ["a.png"], [], "plaque") == ""
    assert q([], ["a.png"], ["paper"], "figure") == ppv.MIXED


def test_the_inputs_are_read_from_the_top_of_a_folder(tmp_path):
    _pdf(tmp_path / "mixed" / "b.pdf")
    _pdf(tmp_path / "mixed" / "a.pdf")
    _png(tmp_path / "mixed" / "p1.png")
    _png(tmp_path / "mixed" / "deeper" / "p2.png")
    paper = tmp_path / "paper"
    _png(paper / "fig1.png")
    (paper / PP.LEGENDS_FILE).write_text("file,legend\n")
    pdfs, images, papers = dh.plaque_inputs([tmp_path / "mixed", paper])
    assert [p.name for p in pdfs] == ["a.pdf", "b.pdf"]
    assert [p.name for p in images] == ["p1.png"]
    assert papers == [paper]


def test_a_folder_of_pdfs_is_accepted_by_the_drop(tmp_path):
    folder = tmp_path / "papers"
    _pdf(folder / "one.pdf")
    assert dh.PlaqueDropHandler().can_accept(folder)


def test_a_pdf_dropped_in_plaque_mode_asks_and_switches(plaque_screen,
                                                        tmp_path, answer):
    asked = answer("figure")
    pdf = _pdf(tmp_path / "paper.pdf")
    assert _mode(plaque_screen) == "plaque"
    _drop(plaque_screen, [pdf])
    assert asked.boxes[0]["title"] == "Switch to Figure mode?"
    assert "PDFs are read in Figure mode" in asked.boxes[0]["text"]
    assert _mode(plaque_screen) == "figure"
    assert plaque_screen._plaque_mode_switch.mode() == "figure"
    assert plaque_screen.fetched == [(str(pdf), str(tmp_path))]


def test_a_pdf_dropped_in_plaque_mode_stays_unread_when_told_to_stay(
        plaque_screen, tmp_path, answer):
    asked = answer("plaque")
    _drop(plaque_screen, [_pdf(tmp_path / "paper.pdf")])
    assert len(asked.boxes) == 1
    assert _mode(plaque_screen) == "plaque"
    assert plaque_screen.fetched == []


def test_a_folder_of_pdfs_dropped_in_plaque_mode_asks_and_reads_the_first(
        plaque_screen, tmp_path, answer):
    answer("figure")
    folder = tmp_path / "papers"
    _pdf(folder / "b.pdf")
    first = _pdf(folder / "a.pdf")
    _drop(plaque_screen, [folder])
    assert _mode(plaque_screen) == "figure"
    assert plaque_screen.fetched[0] == (str(first), str(folder))


def test_images_dropped_in_figure_mode_ask_to_switch_to_plaque(
        plaque_screen, tmp_path, answer):
    ppv.choose_plaque_mode(plaque_screen, "figure")
    asked = answer("plaque")
    folder = tmp_path / "plaques"
    _png(folder / "p1.png")
    _drop(plaque_screen, [folder])
    assert asked.boxes[0]["title"] == "Switch to Plaque mode?"
    assert _mode(plaque_screen) == "plaque"
    assert _src(plaque_screen) == str(folder)


def test_images_dropped_in_figure_mode_are_read_as_figures_when_told_to_stay(
        plaque_screen, tmp_path, answer):
    ppv.choose_plaque_mode(plaque_screen, "figure")
    answer("figure")
    folder = tmp_path / "figures"
    _png(folder / "fig1.png")
    _drop(plaque_screen, [folder])
    assert _mode(plaque_screen) == "figure"
    assert _src(plaque_screen) == str(folder)


@pytest.mark.parametrize("choice", ["plaque", "figure", ""])
def test_pdfs_and_images_dropped_together_ask_which_to_read(
        plaque_screen, tmp_path, answer, choice):
    asked = answer(choice)
    folder = tmp_path / "both"
    pdf = _pdf(folder / "paper.pdf")
    _png(folder / "p1.png")
    _drop(plaque_screen, [folder])
    box = asked.boxes[0]
    assert box["text"] == "PDFs are read in Figure mode, images in Plaque mode."
    assert "1 PDF(s)" in box["info"] and "1 image(s)" in box["info"]
    assert sorted(v for v in box["buttons"].values()) == ["", "figure", "plaque"]
    if choice == "plaque":
        assert _mode(plaque_screen) == "plaque"
        assert _src(plaque_screen) == str(folder)
        assert plaque_screen.fetched == []
    elif choice == "figure":
        assert _mode(plaque_screen) == "figure"
        assert plaque_screen.fetched == [(str(pdf), str(folder))]
    else:
        assert _mode(plaque_screen) == "plaque"
        assert _src(plaque_screen) != str(folder)
        assert plaque_screen.fetched == []


def test_a_pdf_file_and_an_image_file_dropped_together_ask_which(
        plaque_screen, tmp_path, answer):
    ppv.choose_plaque_mode(plaque_screen, "figure")
    asked = answer("plaque")
    pdf = _pdf(tmp_path / "a" / "paper.pdf")
    image = _png(tmp_path / "b" / "p1.png")
    _drop(plaque_screen, [pdf, image])
    assert asked.boxes[0]["title"] == "PDFs or images?"
    assert _mode(plaque_screen) == "plaque"
    assert _src(plaque_screen) == str(image.parent)


def test_src_with_pdfs_in_plaque_mode_asks_and_reads_the_pdf(
        plaque_screen, tmp_path, answer):
    plaque_screen.show()
    asked = answer("figure")
    folder = tmp_path / "papers"
    pdf = _pdf(folder / "paper.pdf")
    plaque_screen._settings_model._widgets["src"].setText(str(folder))
    _settle()
    assert [b["title"] for b in asked.boxes] == ["Switch to Figure mode?"]
    assert _mode(plaque_screen) == "figure"
    assert plaque_screen.fetched == [(str(pdf), str(folder))]


def test_src_with_images_in_figure_mode_asks_to_switch(plaque_screen,
                                                       tmp_path, answer):
    plaque_screen.show()
    ppv.choose_plaque_mode(plaque_screen, "figure")
    asked = answer("plaque")
    folder = tmp_path / "plaques"
    _png(folder / "p1.png")
    plaque_screen._settings_model._widgets["src"].setText(str(folder))
    _settle()
    assert [b["title"] for b in asked.boxes] == ["Switch to Plaque mode?"]
    assert _mode(plaque_screen) == "plaque"


def test_src_with_both_asks_which_and_asks_once(plaque_screen, tmp_path,
                                               answer):
    plaque_screen.show()
    asked = answer("plaque")
    folder = tmp_path / "both"
    _pdf(folder / "paper.pdf")
    _png(folder / "p1.png")
    field = plaque_screen._settings_model._widgets["src"]
    field.setText(str(folder))
    _settle()
    assert [b["text"] for b in asked.boxes] == [
        "PDFs are read in Figure mode, images in Plaque mode."]
    assert _mode(plaque_screen) == "plaque"
    field.setText("")
    _settle()
    field.setText(str(folder))
    _settle()
    assert len(asked.boxes) == 1, "the same src and mode are asked about once"


def test_src_written_by_a_drop_is_not_asked_about_again(plaque_screen,
                                                         tmp_path, answer):
    plaque_screen.show()
    ppv.choose_plaque_mode(plaque_screen, "figure")
    asked = answer("figure")
    folder = tmp_path / "figures"
    _png(folder / "fig1.png")
    _drop(plaque_screen, [folder])
    _settle()
    assert len(asked.boxes) == 1


def test_a_paper_folder_in_figure_mode_asks_nothing(plaque_screen, tmp_path,
                                                     answer):
    plaque_screen.show()
    ppv.choose_plaque_mode(plaque_screen, "figure")
    asked = answer()
    paper = tmp_path / "paper"
    _png(paper / "fig1.png")
    (paper / PP.PAPER_FILE).write_text("{}")
    plaque_screen._settings_model._widgets["src"].setText(str(paper))
    _settle()
    assert asked.boxes == []


def test_a_record_without_pdfplumber_is_stale():
    old = {"requirements": ["ultralytics==8.4.157",
                            "rapidocr_onnxruntime==1.4.4"]}
    assert SB._stale_requirements("papers", old) == ["pdfplumber==0.11.10"]
    new = dict(old, requirements=old["requirements"] + ["pdfplumber==0.11.10"])
    assert SB._stale_requirements("papers", new) == []
    assert SB._stale_requirements("papers", {}) == []


def _state(ready=True, record=None, reason="in its own environment"):
    """A stand-in for the reader's on-disk state."""
    return SimpleNamespace(ready=ready, in_process=False, reason=reason,
                           env="/env/papers", record=record or {},
                           state="installed" if ready else "installable")


def test_the_readiness_check_reads_the_install_record(monkeypatch):
    monkeypatch.setattr(PP, "_importable", lambda module: False)
    full = {"requirements": list(SB._spec("papers").requirements)}
    monkeypatch.setattr(SB, "_backend_state", lambda name: _state(record=full))
    assert PP.reader_problem(pdf=True) == ("", "")
    old = {"requirements": list(SB._spec("papers").requirements[:2])}
    monkeypatch.setattr(SB, "_backend_state", lambda name: _state(record=old))
    kind, why = PP.reader_problem(pdf=True)
    assert kind == "reinstall" and "pdfplumber" in why
    assert PP.reader_problem() == ("", ""), "detection does not need pdfplumber"
    monkeypatch.setattr(SB, "_backend_state",
                        lambda name: _state(ready=False, reason="not here"))
    assert PP.reader_problem(pdf=True) == ("install", "not here")


def test_the_reader_says_a_missing_pdfplumber_is_an_old_install(monkeypatch,
                                                               tmp_path):
    monkeypatch.setitem(sys.modules, "pdfplumber", None)
    with pytest.raises(ModuleNotFoundError) as caught:
        SB._worker_read_pdf({"pdf": "x.pdf", "dest": str(tmp_path)}, {})
    assert "Model Zoo" not in str(caught.value)
    assert caught.value.name == "pdfplumber"


def test_a_reader_without_pdfplumber_asks_for_the_reinstall(monkeypatch,
                                                            tmp_path):
    monkeypatch.setattr(PP, "reader_problem", lambda pdf=False: ("", ""))
    monkeypatch.setattr(PP, "reader_environment", lambda: "/env/papers")

    class Worker:
        def request(self, op, **payload):
            raise SB._BackendError(
                "Plaque figure reader raised ModuleNotFoundError: no "
                "pdfplumber", "ModuleNotFoundError")

    monkeypatch.setattr(SB, "_worker_for", lambda name, env: Worker())
    with pytest.raises(PP.ReaderNeedsInstall) as caught:
        PP._pdf_pages_in_reader(tmp_path / "x.pdf", tmp_path, 100)
    assert caught.value.reinstall


def _fake_runner(record):
    """Pretend each install step ran and succeeded."""
    def run(argv, env=None, cwd=None, on_line=None, cancel=None):
        record.append(tuple(argv))
        if argv[1:3] == ("-m", "venv"):
            python = Path(SB._env_python(argv[3]))
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("")
        if "--selftest" in argv:
            return 0, [json.dumps({"ok": True, "python": "3.12.0",
                                   "packages": {}, "device": "cpu"})]
        return 0, ["ok"]
    return run


def test_reinstall_builds_an_installed_reader_again(tmp_path, monkeypatch):
    root = tmp_path / "backends"
    monkeypatch.setenv(SB._ROOT_ENV, str(root))
    monkeypatch.setattr(SB, "_PROBED", {})
    env = root / "papers"
    python = Path(SB._env_python(str(env)))
    python.parent.mkdir(parents=True)
    python.write_text("")
    SB._write_marker(str(env), {"backend": "papers", "requirements": [
        "ultralytics==8.4.157", "rapidocr-onnxruntime==1.4.4"]})
    assert SB._stale_requirements("papers") == ["pdfplumber==0.11.10"]
    record = []
    same = SB._install_backend("papers", runner=_fake_runner(record),
                               preflight=lambda spec, r: (sys.executable,),
                               torch_index="")
    assert record == [] and same.ready, "without reinstall nothing is rebuilt"
    state = SB._install_backend("papers", runner=_fake_runner(record),
                                preflight=lambda spec, r: (sys.executable,),
                                torch_index="", reinstall=True)
    assert state.ready and record
    assert SB._stale_requirements("papers") == []


class _FakeDialog(QObject):
    """The install dialog's signals, without the dialog."""

    job_started = Signal()
    job_progressed = Signal(str)
    job_failed = Signal(str)
    job_cancelled = Signal()


def _installer(outcome, calls):
    """A fake :func:`install_backend` that ends with ``outcome``."""
    def install(parent, name, *, watch=None, reinstall=False, why=""):
        calls.append({"name": name, "reinstall": reinstall, "why": why})
        dialog = _FakeDialog()
        watch(dialog)
        dialog.job_started.emit()
        dialog.job_progressed.emit("Install Plaque figure reader: pip line")
        if outcome == "failed":
            dialog.job_failed.emit("pip said: no space left on device")
            return False
        if outcome == "cancelled":
            dialog.job_cancelled.emit()
            return False
        return True
    return install


@pytest.fixture
def panel(qtbot, monkeypatch):
    """A preview panel whose status line keeps every message."""
    widget = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    said = []
    real = widget.set_preview_status
    monkeypatch.setattr(widget, "set_preview_status",
                        lambda text: said.append(text) or real(text))
    widget.said = said
    return widget


def test_the_reinstall_runs_in_place_and_says_it_worked(panel, monkeypatch):
    monkeypatch.setattr(PP, "reader_problem",
                        lambda pdf=False: ("reinstall", "installed before PDFs"))
    calls = []
    assert panel._offer_install(installer=_installer("installed", calls))
    assert calls == [{"name": "papers", "reinstall": True,
                      "why": "installed before PDFs"}]
    assert any("Installing Plaque figure reader…: Install Plaque figure "
               "reader: pip line" == text for text in panel.said)
    assert panel.said[-1] == "Plaque figure reader is installed"


def test_a_failed_reinstall_says_why(panel, monkeypatch):
    monkeypatch.setattr(PP, "reader_problem",
                        lambda pdf=False: ("reinstall", "old"))
    assert not panel._offer_install(installer=_installer("failed", []))
    assert any("failed" in text and "no space left on device" in text
               for text in panel.said)
    assert not any("was not installed" in text for text in panel.said)


def test_figure_mode_offers_the_reinstall_on_its_banner(panel, monkeypatch):
    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a: [])
    monkeypatch.setattr(PP, "reader_problem",
                        lambda pdf=False: ("reinstall", "installed before PDFs"))
    panel.set_mode("figure")
    assert not panel._deps_banner.isHidden()
    assert panel._install_btn.text() == "Reinstall"
    assert "installed before PDFs" in panel._deps_text.text()
    assert "Model Zoo" not in panel._deps_text.text()


def test_a_pdf_that_needs_the_reader_offers_it_then_reads(panel, monkeypatch,
                                                          tmp_path):
    problem = {"kind": "reinstall"}
    monkeypatch.setattr(PP, "reader_problem",
                        lambda pdf=False: (problem["kind"], "old reader"))
    fetched = []
    monkeypatch.setattr(PP, "fetch_paper_to_folder",
                        lambda ref, dest: fetched.append(ref) or {"folder": ""})
    calls = []

    def install(parent, name, **kw):
        calls.append(kw["reinstall"])
        problem["kind"] = ""
        return _installer("installed", [])(parent, name, **kw)

    monkeypatch.setattr("spacr.qt.widgets.model_zoo_picker.install_backend",
                        install)
    pdf = _pdf(tmp_path / "paper.pdf")
    assert panel.fetch_paper(str(pdf), str(tmp_path))
    assert calls == [True]
    assert fetched == [str(pdf)]


def test_a_pdf_is_left_unread_when_the_reinstall_fails(panel, monkeypatch,
                                                       tmp_path):
    monkeypatch.setattr(PP, "reader_problem",
                        lambda pdf=False: ("install", "not installed"))
    fetched = []
    monkeypatch.setattr(PP, "fetch_paper_to_folder",
                        lambda ref, dest: fetched.append(ref) or {})
    monkeypatch.setattr("spacr.qt.widgets.model_zoo_picker.install_backend",
                        _installer("failed", []))
    assert panel.fetch_paper(str(_pdf(tmp_path / "paper.pdf")), str(tmp_path))
    assert fetched == []
    assert any("no space left on device" in text for text in panel.said)


def test_a_reader_that_fails_while_reading_offers_the_reinstall(
        panel, monkeypatch, tmp_path):
    monkeypatch.setattr(PP, "reader_problem", lambda pdf=False: ("", ""))
    attempts = []

    def fetch(ref, dest):
        attempts.append(ref)
        if len(attempts) == 1:
            raise PP.ReaderNeedsInstall("no pdfplumber", reinstall=True)
        return {"folder": ""}

    monkeypatch.setattr(PP, "fetch_paper_to_folder", fetch)
    calls = []
    monkeypatch.setattr("spacr.qt.widgets.model_zoo_picker.install_backend",
                        _installer("installed", calls))
    pdf = _pdf(tmp_path / "paper.pdf")
    assert panel.fetch_paper(str(pdf), str(tmp_path))
    assert [c["reinstall"] for c in calls] == [True]
    assert attempts == [str(pdf), str(pdf)]


def test_the_install_dialog_can_reinstall(qtbot, monkeypatch):
    from spacr.qt.widgets import model_zoo_picker as mzp

    dialog = mzp.BackendInstallDialog("papers", reinstall=True,
                                      why="It was installed before PDFs.")
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Reinstall Plaque figure reader"
    assert dialog.start_button.text() == "Reinstall"
    assert dialog.blurb.text().startswith("It was installed before PDFs.")
    seen = []
    monkeypatch.setattr(SB, "_install_backend",
                        lambda name, **kw: seen.append((name, kw["reinstall"])))
    dialog._job(progress=None, cancel=None)
    assert seen == [("papers", True)]


@pytest.mark.parametrize("as_folder", [False, True])
def test_src_naming_a_pdf_in_figure_mode_reads_it_without_asking(
        plaque_screen, tmp_path, answer, as_folder):
    """The maintainer could not load a PDF through src: the preview said
    "No images found in paper.pdf". In Figure mode a PDF in src is read."""
    plaque_screen.show()
    ppv.choose_plaque_mode(plaque_screen, "figure")
    asked = answer()
    pdf = _pdf(tmp_path / "papers" / "paper.pdf")
    target = pdf.parent if as_folder else pdf
    plaque_screen._settings_model._widgets["src"].setText(str(target))
    _settle()
    assert asked.boxes == []
    assert plaque_screen.fetched == [(str(pdf), str(pdf.parent))]
