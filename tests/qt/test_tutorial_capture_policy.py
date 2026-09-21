"""The recording policy checks live widgets before a frame is saved."""
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QLabel, QListWidget, QTableWidget, QTableWidgetItem, QWidget

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "tutorials"))
from capture_policy import configure_appearance, verify_appearance, verify_visible_paths


@pytest.fixture
def recording(qtbot, monkeypatch, tmp_path):
    from spacr.qt import preferences as prefs
    from spacr.qt.widgets.ambient import AmbientWidget

    store = QSettings(str(tmp_path / "capture.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    monkeypatch.delenv("SPACR_NO_BACKDROP", raising=False)
    configure_appearance()
    window = QWidget()
    qtbot.addWidget(window)
    window.resize(400, 300)
    palette = window.palette()
    palette.setColor(QPalette.Window, QColor("#121212"))
    window.setPalette(palette)
    backdrop = AmbientWidget(window, theme="blobs")
    backdrop.setGeometry(window.rect())
    window.show()
    backdrop.show()
    qtbot.waitUntil(lambda: backdrop.frames_painted > 0)
    yield window, backdrop, prefs
    backdrop.hide()


def test_a_real_dark_window_with_painted_blobs_is_accepted(recording):
    window, backdrop, _ = recording
    receipt = verify_appearance(window)
    assert receipt == {"theme": "dark", "backdrop": "blobs",
                       "painted_frames": backdrop.frames_painted}


@pytest.mark.parametrize("drift", ["saved_theme", "painted_palette", "backdrop", "hidden", "disabled"])
def test_drift_is_refused_even_when_the_capture_command_requested_blobs(recording, drift):
    window, backdrop, prefs = recording
    if drift == "saved_theme":
        prefs.set_theme("light")
    elif drift == "painted_palette":
        palette = window.palette()
        palette.setColor(QPalette.Window, QColor("white"))
        window.setPalette(palette)
    elif drift == "backdrop":
        backdrop.set_theme("aurora")
    elif drift == "hidden":
        backdrop.hide()
    else:
        prefs.set_ambient_enabled(False)
    with pytest.raises(RuntimeError, match="Capture refused"):
        verify_appearance(window)


def test_the_capture_does_not_silently_accept_another_named_appearance():
    with pytest.raises(ValueError, match="dark mode and the Blobs"):
        configure_appearance("light", "blobs")


def test_a_personal_capture_root_is_not_an_exception_to_the_path_rule():
    with pytest.raises(ValueError, match="neutral prepared capture directory"):
        verify_visible_paths([], "/mnt/firecuda2/alice/tutorials")


@pytest.mark.parametrize("path", ["/home/alice/data/plate.tif", "/Users/alice/data", r"C:\Users\Alice\data", "/mnt/firecuda2/private/plate.tif", "/nas_mnt/plate"])
def test_visible_personal_paths_are_refused_but_hidden_text_is_not(qtbot, path):
    window = QWidget()
    qtbot.addWidget(window)
    label = QLabel(path, window)
    window.show()
    with pytest.raises(RuntimeError, match="visible text"):
        verify_visible_paths([window], "/tmp/tutorial")
    label.hide()
    verify_visible_paths([window], "/tmp/tutorial")


def test_neutral_prepared_paths_are_accepted_and_table_paths_are_checked(qtbot):
    table = QTableWidget(1, 1)
    qtbot.addWidget(table)
    table.setItem(0, 0, QTableWidgetItem("/tmp/tutorial/example/plate.tif"))
    table.show()
    verify_visible_paths([table], "/tmp/tutorial")
    table.item(0, 0).setText("/home/alice/private/plate.tif")
    with pytest.raises(RuntimeError, match="visible text"):
        verify_visible_paths([table], "/tmp/tutorial")


def test_search_results_use_the_real_list_model_without_private_virtual_calls(qtbot):
    results = QListWidget()
    qtbot.addWidget(results)
    results.addItem("Performance — Preferences")
    results.show()
    verify_visible_paths([results], "/tmp/tutorial")
    results.item(0).setText("/home/alice/private/result")
    with pytest.raises(RuntimeError, match="visible text"):
        verify_visible_paths([results], "/tmp/tutorial")


@pytest.mark.parametrize("text", [
    "spaCR 0.0.0.1 release notes",
    "<b>spaCR</b> <span>0.0.0.1</span> release notes",
    "spaCR-0.0.0.1-Linux-x86_64-Online.run",
])
def test_visible_other_releases_are_refused_but_hidden_history_is_not(qtbot, text):
    window = QWidget()
    qtbot.addWidget(window)
    label = QLabel(text, window)
    window.show()
    with pytest.raises(RuntimeError, match="another spaCR release"):
        verify_visible_paths([window], "/tmp/tutorial")
    label.hide()
    verify_visible_paths([window], "/tmp/tutorial")


def test_current_release_and_other_software_versions_are_accepted(qtbot):
    from spacr import __version__

    label = QLabel(f"spaCR {__version__}; Python 3.12; Qt 6.11.2")
    qtbot.addWidget(label)
    label.show()
    verify_visible_paths([label], "/tmp/tutorial")
