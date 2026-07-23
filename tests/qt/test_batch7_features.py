"""Tests for the Batch 7 UX additions:

* Preferences menu entry on the &spaCR menu.
* Cellpose-SAM as the default model in the Live Preview panel.
* "LP" toggle label next to the AI toggle on the Mask app screen.
* End-to-end demo entry on the &Demos menu (confirm popup →
  folder picker → HF download → chained mask/measure/annotate).

The download itself is monkey-patched: we don't want tests to hit
huggingface.co, and the point of these tests is the *wiring*, not
the network layer (the HF module has its own tests where relevant).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from spacr.qt.app import MainWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def _isolated_qsettings(monkeypatch, tmp_path):
    """Redirect QSettings so we don't clobber user prefs during tests.

    Also re-marks the first-launch tour as seen so the overlay never
    intercepts events in a MainWindow-constructing test.
    """
    from PySide6.QtCore import QSettings
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope,
                        str(tmp_path))
    try:
        from spacr.qt.first_run import mark_tour_seen
        mark_tour_seen()
    except Exception:
        pass
    yield


@pytest.fixture
def mw(qtbot, _isolated_qsettings):
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def _menu_labels(win, name: str):
    """Return the visible-text of every non-separator action under the
    top-level menu with visible ``name`` on ``win``'s menubar.

    Everything happens inside a single expression so we never hand a
    QMenu reference back to Python — Qt keeps ownership and there's no
    "already deleted" race across function boundaries.
    """
    mb = win.menuBar()
    labels: list = []
    for top_act in mb.actions():
        if top_act.text().replace("&", "") != name:
            continue
        m = top_act.menu()
        if m is None:
            continue
        for a in m.actions():
            if not a.isSeparator():
                labels.append(a.text())
        break
    return labels


def _menu_actions(win, name: str):
    """Same shape as :func:`_menu_labels` but returns per-action
    ``(text, shortcut_str)`` tuples so callers can assert on the
    shortcut without holding a QAction ref."""
    mb = win.menuBar()
    out: list = []
    for top_act in mb.actions():
        if top_act.text().replace("&", "") != name:
            continue
        m = top_act.menu()
        if m is None:
            continue
        for a in m.actions():
            if not a.isSeparator():
                out.append((a.text(), a.shortcut().toString()))
        break
    return out


# ---------------------------------------------------------------------------
# Preferences on the spaCR menu
# ---------------------------------------------------------------------------

class TestPreferencesMenuEntry:
    def test_preferences_action_present(self, mw):
        labels = _menu_labels(mw, "spaCR")
        assert any("Preferences" in lbl for lbl in labels)

    def test_preferences_action_has_ctrl_comma_shortcut(self, mw):
        actions = _menu_actions(mw, "spaCR")
        for text, shortcut in actions:
            if "Preferences" in text:
                # Ctrl+, sometimes normalises to "Ctrl+" on Qt
                assert shortcut in ("Ctrl+,", "Ctrl+")
                return
        pytest.fail("no Preferences action found")

    def test_open_preferences_opens_dialog(self, mw, monkeypatch):
        """Stub the dialog's exec so we don't block on modal input."""
        called = {"opened": False}
        class _StubDialog:
            def __init__(self, parent=None):
                pass
            def exec(self):
                called["opened"] = True
        monkeypatch.setattr("spacr.qt.preferences.PreferencesDialog",
                             _StubDialog)
        mw._open_preferences()
        assert called["opened"] is True


# ---------------------------------------------------------------------------
# Cellpose-SAM as default model
# ---------------------------------------------------------------------------

class TestLivePreviewModelDefault:
    def test_default_model_is_cpsam(self, qtbot):
        from spacr.qt.widgets.live_preview import LivePreviewPanel
        panel = LivePreviewPanel()
        qtbot.addWidget(panel)
        assert panel.current_params()["model"] == "cpsam"

    def test_legacy_models_still_available(self, qtbot):
        from spacr.qt.widgets.live_preview import LivePreviewPanel
        panel = LivePreviewPanel()
        qtbot.addWidget(panel)
        items = [panel._model_box.itemText(i)
                  for i in range(panel._model_box.count())]
        assert "cpsam" in items
        for legacy in ("cyto3", "cyto2", "nuclei"):
            assert legacy in items
        # SAM should be first (default)
        assert items[0] == "cpsam"


# ---------------------------------------------------------------------------
# LP toggle label on Mask app
# ---------------------------------------------------------------------------

class TestLpToggle:
    def test_mask_screen_has_lp_switch(self, qtbot):
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        assert getattr(scr, "_lp_switch", None) is not None
        assert scr._lp_switch.text() == "LP"

    def test_other_screens_have_no_lp_switch(self, qtbot):
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen("measure")
        qtbot.addWidget(scr)
        assert getattr(scr, "_lp_switch", None) is None

    def test_lp_starts_off_and_hides_card(self, qtbot):
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        scr.show()
        assert scr._lp_switch.isChecked() is False
        assert scr._live_preview_card.isVisible() is False

    def test_toggling_lp_shows_card(self, qtbot):
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        scr.show()
        scr._lp_switch.setChecked(True)
        assert scr._live_preview_card.isVisible() is True
        scr._lp_switch.setChecked(False)
        assert scr._live_preview_card.isVisible() is False


# ---------------------------------------------------------------------------
# End-to-end HF demo entry
# ---------------------------------------------------------------------------

class TestE2EDemoMenu:
    def test_e2e_action_present(self, mw):
        labels = _menu_labels(mw, "Demos")
        assert any("End-to-end" in lbl and "Annotate" in lbl
                    for lbl in labels)

    def test_e2e_asks_for_confirmation_no_downloads_on_cancel(
            self, mw, monkeypatch, tmp_path):
        """User clicks "No" -> we should not call the downloader."""
        monkeypatch.setattr(QMessageBox, "question",
                             lambda *a, **k: QMessageBox.No)
        called = {"downloaded": False}
        def _stub_download(parent, dest, on_done):
            called["downloaded"] = True
        monkeypatch.setattr("spacr.qt.hf_download.download_toxo_mito_demo",
                             _stub_download)
        # Also stub the folder picker so if we WERE to reach it, nothing
        # opens on the test box.
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getExistingDirectory",
            lambda *a, **k: "")
        mw._on_e2e_demo()
        assert called["downloaded"] is False

    def test_e2e_yes_then_no_folder_still_no_download(
            self, mw, monkeypatch):
        """Yes -> folder picker returns empty -> no download either."""
        monkeypatch.setattr(QMessageBox, "question",
                             lambda *a, **k: QMessageBox.Yes)
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getExistingDirectory",
            lambda *a, **k: "")
        called = {"downloaded": False}
        def _stub_download(parent, dest, on_done):
            called["downloaded"] = True
        monkeypatch.setattr("spacr.qt.hf_download.download_toxo_mito_demo",
                             _stub_download)
        mw._on_e2e_demo()
        assert called["downloaded"] is False

    def test_e2e_yes_and_folder_kicks_download(
            self, mw, monkeypatch, tmp_path):
        """Yes + folder picked -> downloader called with that folder."""
        monkeypatch.setattr(QMessageBox, "question",
                             lambda *a, **k: QMessageBox.Yes)
        monkeypatch.setattr(
            "PySide6.QtWidgets.QFileDialog.getExistingDirectory",
            lambda *a, **k: str(tmp_path))
        captured = {}
        def _stub_download(parent, dest, on_done):
            captured["dest"] = str(dest)
            captured["parent"] = parent
        monkeypatch.setattr("spacr.qt.hf_download.download_toxo_mito_demo",
                             _stub_download)
        mw._on_e2e_demo()
        assert captured["dest"] == str(tmp_path)
        assert captured["parent"] is mw

    def test_e2e_chain_prompts_and_navigates(
            self, mw, monkeypatch, tmp_path):
        """After a fake successful download, the chain should prompt
        the user before each stage; if they answer Yes to all three,
        we should end up having navigated to mask, measure, and
        annotate in turn."""
        settings_dir = tmp_path / "settings"
        settings_dir.mkdir()
        for stage in ("mask", "measure", "annotate"):
            (settings_dir / f"{stage}_settings.csv").write_text("plot,false\n")
        dataset_dir = tmp_path / "plate1"
        dataset_dir.mkdir()

        prompts = []
        def _yes(*a, **k):
            prompts.append(a)
            return QMessageBox.Yes
        monkeypatch.setattr(QMessageBox, "question", _yes)
        # Stub the pipeline run so we don't actually start Cellpose
        monkeypatch.setattr(
            "spacr.qt.screens.app_screen.AppScreen._on_run",
            lambda self: None)

        mw._run_e2e_chain(dataset_dir, settings_dir)

        # Three prompts — one per stage
        assert len(prompts) == 3
        assert "annotate" in mw._screens
        assert "mask" in mw._screens
        assert "measure" in mw._screens

    def test_e2e_chain_stops_when_user_says_no(
            self, mw, monkeypatch, tmp_path):
        """Answering No at the first prompt should abort the chain
        without touching downstream screens."""
        settings_dir = tmp_path / "settings"
        settings_dir.mkdir()
        (settings_dir / "mask_settings.csv").write_text("plot,false\n")
        dataset_dir = tmp_path / "plate1"
        dataset_dir.mkdir()

        monkeypatch.setattr(QMessageBox, "question",
                             lambda *a, **k: QMessageBox.No)
        called = {"run": 0}
        def _bump(self):
            called["run"] += 1
        monkeypatch.setattr(
            "spacr.qt.screens.app_screen.AppScreen._on_run", _bump)

        mw._run_e2e_chain(dataset_dir, settings_dir)
        assert called["run"] == 0
        # Also — measure/annotate should not have been navigated to.
        assert "measure" not in mw._screens
        assert "annotate" not in mw._screens
