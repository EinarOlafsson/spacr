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

    def test_preferences_action_has_ctrl_p_shortcut(self, mw):
        """Ctrl+P, asked for on 2026-09-08 in place of Ctrl+comma.

        The old key needed the "sometimes normalises to Ctrl+" allowance
        below, because a trailing comma is not a character Qt round-trips
        through a key sequence cleanly. Ctrl+P has no such problem.
        """
        actions = _menu_actions(mw, "spaCR")
        for text, shortcut in actions:
            if "Preferences" in text:
                assert shortcut == "Ctrl+P"
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

class TestRetiredDemoMenu:
    def test_e2e_action_is_absent(self, mw):
        labels = _menu_labels(mw, "Demos")
        assert labels == []
