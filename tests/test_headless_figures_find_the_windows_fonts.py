"""Headless figures choose offscreen Qt, and on Windows they keep real fonts.

Windows never sets DISPLAY, so every headless figure there starts Qt on the
offscreen platform, whose font database is empty unless QT_QPA_FONTDIR names
a folder: all text fell back to one built-in face.
"""
import os

import pytest

from spacr.figures import scene


@pytest.fixture
def headless(monkeypatch):
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM", "QT_QPA_FONTDIR"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_no_display_chooses_offscreen(headless):
    scene._choose_offscreen_without_a_display()
    assert os.environ["QT_QPA_PLATFORM"] == "offscreen"


def test_a_named_display_is_left_to_qt(headless):
    headless.setenv("DISPLAY", ":0")
    scene._choose_offscreen_without_a_display()
    assert "QT_QPA_PLATFORM" not in os.environ
    assert "QT_QPA_FONTDIR" not in os.environ


def test_a_chosen_platform_is_kept_and_gets_no_font_folder(headless):
    headless.setenv("QT_QPA_PLATFORM", "minimal")
    scene._choose_offscreen_without_a_display()
    assert os.environ["QT_QPA_PLATFORM"] == "minimal"
    assert "QT_QPA_FONTDIR" not in os.environ


@pytest.mark.skipif(os.name != "nt", reason="the Windows font folder exists only on Windows")
def test_windows_offscreen_names_the_system_font_folder(headless):
    scene._choose_offscreen_without_a_display()
    fonts = os.environ["QT_QPA_FONTDIR"]
    assert os.path.normcase(fonts) == os.path.normcase(
        os.path.join(os.environ.get("WINDIR") or os.environ["SystemRoot"], "Fonts"))
    assert os.path.isdir(fonts)


@pytest.mark.skipif(os.name != "nt", reason="the Windows font folder exists only on Windows")
def test_a_font_folder_the_caller_chose_is_kept(headless, tmp_path):
    headless.setenv("QT_QPA_FONTDIR", str(tmp_path))
    scene._choose_offscreen_without_a_display()
    assert os.environ["QT_QPA_FONTDIR"] == str(tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX offscreen Qt finds fonts through fontconfig")
def test_posix_offscreen_needs_no_font_folder(headless):
    scene._choose_offscreen_without_a_display()
    assert "QT_QPA_FONTDIR" not in os.environ
