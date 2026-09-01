"""Tests for the Batch 7 UX additions:

* Preferences menu entry on the &spaCR menu.
* Cellpose-SAM as the default model in the Live Preview panel.
* "Live" toggle label next to the AI toggle on the Mask app screen.
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
    stack = [act for act in mb.actions()]
    while stack:
        top_act = stack.pop(0)
        m = top_act.menu()
        if m is None:
            continue
        if top_act.text().replace("&", "") != name:
            # Not this one -- but a menu can be nested now: Demos moved
            # under Help on 2026-08-23, so a search that only looks at
            # the bar's own actions finds nothing at all.
            stack.extend(m.actions())
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

    def test_legacy_models_are_not_offered_but_are_still_accepted(self, qtbot):
        """UPDATED 2026-09-01, and the distinction is the point.

        This used to assert the pre-SAM spellings were IN the live combo. They
        are deliberately not, at the maintainer's request: all four resolve to
        cpsam, so offering them is four labels for one model.

        The obligation they existed for is real and is kept -- a SAVED settings
        file naming cyto2 must still round-trip, or the preview quietly uses a
        different model than the settings say. That is now handled by
        accepting the value rather than by advertising it, which is the half
        that actually protected the user.
        """
        from spacr.qt.widgets.live_preview import LivePreviewPanel
        panel = LivePreviewPanel()
        qtbot.addWidget(panel)
        items = [panel._model_box.itemText(i)
                  for i in range(panel._model_box.count())]
        assert "cpsam" in items
        assert items[0] == "cpsam", "SAM is the default and comes first"
        for legacy in ("cyto3", "cyto2", "nuclei"):
            assert legacy not in items, f"{legacy} is still offered"

        # ... and a settings file naming one is still honoured.
        panel.apply_settings({"model_name": "cyto2"})
        assert panel._model_box.currentText() == "cyto2"


# ---------------------------------------------------------------------------
# Live toggle label on Mask app
# ---------------------------------------------------------------------------

class TestLpToggle:
    def test_mask_screen_has_lp_switch(self, qtbot):
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        assert getattr(scr, "_lp_switch", None) is not None
        assert scr._lp_switch.text() == "Live"

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

    def test_importing_the_demo_opens_mask_with_its_settings(
            self, mw, monkeypatch, tmp_path):
        """One screen, filled in, and nothing started.

        This asserted three prompts and three screens until 2026-08-31,
        when the import stopped being a Mask -> Measure -> Annotate chain
        that ran each pipeline itself. Asked for as "the user should be
        able to hit import and then live preview or run": a demo dataset
        exists to be looked at, and the first thing anyone wants is Live
        Preview on one field.
        """
        settings_dir = tmp_path / "settings"
        settings_dir.mkdir()
        (settings_dir / "mask_settings.csv").write_text("plot,false\n")
        dataset_dir = tmp_path / "plate1"
        dataset_dir.mkdir()

        prompts = []
        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **k: prompts.append(a))
        runs = {"n": 0}
        monkeypatch.setattr(
            "spacr.qt.screens.app_screen.AppScreen._on_run",
            lambda self: runs.__setitem__("n", runs["n"] + 1))

        mw._run_e2e_chain(dataset_dir, settings_dir)

        assert "mask" in mw._screens, "Mask Generation did not open"
        assert runs["n"] == 0, "importing the demo started a pipeline"
        assert prompts == [], (
            "the import asked a question; a Continue prompt before work "
            "nobody asked to start has No as its safe answer")
        # Measure and Annotate are reached from Mask once masks exist.
        # Opening them against a dataset with none would open two screens
        # that can only report there is nothing to do.
        assert "measure" not in mw._screens
        assert "annotate" not in mw._screens

    def test_the_dataset_folder_wins_over_the_packs_own_src(
            self, mw, monkeypatch, tmp_path):
        """The pack names a path on the machine that produced it.

        Driven through the real chain rather than the loader alone,
        because the argument that carries it is the one easy to drop when
        wiring the two together.
        """
        settings_dir = tmp_path / "settings"
        settings_dir.mkdir()
        (settings_dir / "mask_settings.csv").write_text(
            "src,/somebody/elses/disk\n")
        dataset_dir = tmp_path / "plate1"
        dataset_dir.mkdir()

        applied = {}
        monkeypatch.setattr(
            "spacr.qt.screens.app_screen.AppScreen.apply_settings_dict",
            lambda self, settings: applied.update(settings))
        monkeypatch.setattr(
            "spacr.qt.screens.app_screen.AppScreen._on_run",
            lambda self: None)

        mw._run_e2e_chain(dataset_dir, settings_dir)
        assert applied.get("src") == str(dataset_dir)
