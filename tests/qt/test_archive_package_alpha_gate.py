"""The Report screen's Archive package button is an alpha feature.

``ReportArchivePackage`` is registered in ``spacr.settings.ALPHA_FEATURES``,
so it is hidden until Preferences -> "Show alpha features" is on. Hiding is
display only: the package a form asks for is still written and checked
while the button is hidden.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.settings import ALPHA_FEATURES  # noqa: E402


@pytest.fixture
def alpha(monkeypatch):
    from spacr.qt import preferences

    state = {"on": False}
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: state["on"])
    return state


@pytest.fixture
def screen_folder(tmp_path) -> Path:
    src = tmp_path / "plate1"
    (src / "settings").mkdir(parents=True)
    for well in ("A01", "A02"):
        (src / f"plate1_{well}_T0001F001L01A01Z01C01.tif").write_bytes(
            well.encode() * 8)
    pd.DataFrame({"Key": ["experiment"], "Value": ["gate test"]}).to_csv(
        src / "settings" / "measure_settings.csv", index=False)
    return src


def _screen(qtbot, src=None):
    from spacr.qt.screens.report import ReportScreen

    screen = ReportScreen(threaded=False)
    qtbot.addWidget(screen)
    if src is not None:
        screen.set_source(str(src))
    return screen


def test_registered_as_alpha():
    assert ALPHA_FEATURES[574] == {
        "widgets": ("ReportArchivePackage", "ArchiveScreenSources")}


def test_hidden_until_alpha_features_are_shown(qtbot, alpha):
    from PySide6.QtWidgets import QPushButton

    from spacr.qt.preferences import _apply_alpha_widgets

    screen = _screen(qtbot)
    button = screen.findChild(QPushButton, "ReportArchivePackage")
    assert button is not None and button.isHidden()
    alpha["on"] = True
    _apply_alpha_widgets(screen)
    assert not button.isHidden()
    alpha["on"] = False
    _apply_alpha_widgets(screen)
    assert button.isHidden()


def test_the_form_is_filled_from_the_run_folder(qtbot, alpha, screen_folder):
    from PySide6.QtWidgets import QLineEdit

    alpha["on"] = True
    screen = _screen(qtbot, screen_folder)
    assert screen._btn_archive.isEnabled()
    dialog = screen._archive_dialog()
    qtbot.addWidget(dialog)
    title = dialog.findChild(QLineEdit, "ArchiveField_title")
    assert title.text() == "gate test"
    assert dialog.findChild(QLineEdit, "ArchiveOutput").text() == str(
        screen_folder.parent)


def test_a_package_asked_for_while_hidden_is_still_written(qtbot, alpha,
                                                           screen_folder,
                                                           tmp_path):
    screen = _screen(qtbot, screen_folder)
    assert screen._btn_archive.isHidden()
    form = {"description": "Two wells.", "authors": "Doe Jane",
            "email": "jane@example.org", "microscope": "Nikon Ti2"}
    assert screen._write_archive(str(screen_folder), str(tmp_path / "out"),
                                form)
    pkg = tmp_path / "out" / "gate-test"
    assert (pkg / "idr" / "gate-test-study.txt").is_file()
    assert screen._archive_problems == []
    assert "passes the IDR, BioStudies and MIHCSME checks" in \
        screen.status_text()

    screen._write_archive(str(screen_folder), str(tmp_path / "bad"),
                         {"authors": "Doe Jane"})
    assert "does not pass" in screen.status_text()
    assert screen._archive_problems


def test_without_a_run_folder_the_form_is_not_opened(qtbot, alpha, tmp_path):
    """Blank, or a path that is not a folder, asks for a folder first."""
    for src in ("", str(tmp_path / "missing")):
        screen = _screen(qtbot, src=src)
        assert screen._archive_dialog() is None
        screen._on_archive_package()
        assert "Choose a run folder first" in screen._status.text()


def test_the_archive_button_opens_its_form(qtbot, alpha, screen_folder):
    from PySide6.QtWidgets import QDialog

    screen = _screen(qtbot, src=str(screen_folder))
    screen._on_archive_package()
    dialog = screen.findChild(QDialog, "ReportArchiveDialog")
    assert dialog is not None and dialog.isVisible()
    dialog.reject()


def test_further_screens_in_the_form_make_one_study(qtbot, alpha,
                                                    screen_folder, tmp_path):
    """Run folders typed under Further screens become screens B, C, ...

    The field hides with the alpha switch off, and a study asked for while
    the button is hidden is still written and passes its checks.
    """
    import shutil

    from PySide6.QtWidgets import QPlainTextEdit

    from spacr.qt.preferences import _apply_alpha_widgets

    second = tmp_path / "plate2"
    shutil.copytree(screen_folder, second)
    screen = _screen(qtbot, screen_folder)
    dialog = screen._archive_dialog()
    qtbot.addWidget(dialog)
    field = dialog.findChild(QPlainTextEdit, "ArchiveScreenSources")
    assert field is not None
    _apply_alpha_widgets(dialog)
    assert field.isHidden()
    alpha["on"] = True
    _apply_alpha_widgets(dialog)
    assert not field.isHidden()
    alpha["on"] = False
    assert screen._archive_screen_sources(f"\n {second} \n\n") == [
        str(second)]
    form = {"description": "Two plates.", "authors": "Doe Jane",
            "email": "jane@example.org", "microscope": "Nikon Ti2"}
    assert screen._write_archive(str(screen_folder), str(tmp_path / "out"),
                                 form, screens=[str(second)])
    pkg = tmp_path / "out" / "gate-test"
    assert (pkg / "study_manifest.json").is_file()
    assert (pkg / "biostudies" / "gate-test.pagetab.tsv").is_file()
    assert screen._archive_problems == []
    assert "passes the IDR, BioStudies and MIHCSME checks" in \
        screen.status_text()
