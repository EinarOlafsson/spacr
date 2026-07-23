"""Tests for Batch B — theme accent constancy, font-scale-widens-widgets,
and the verbose logger toggle in Preferences."""
from __future__ import annotations

import logging

import pytest


# ---------------------------------------------------------------------------
# Constant accent across themes
# ---------------------------------------------------------------------------

class TestConstantAccent:
    def test_button_accent_same_in_dark_and_light(self):
        from spacr.qt.theme import palette_for
        assert (palette_for("dark")["button_accent"]
                == palette_for("light")["button_accent"])
        assert (palette_for("dark")["button_accent_hi"]
                == palette_for("light")["button_accent_hi"])
        assert (palette_for("dark")["button_accent_lo"]
                == palette_for("light")["button_accent_lo"])

    def test_ai_toggle_uses_constant_accent(self, qtbot):
        from spacr.qt.widgets.ai_toggle_label import AiToggleLabel
        from spacr.qt.theme import CONSTANT_ROLES

        label = AiToggleLabel()
        qtbot.addWidget(label)
        label.setChecked(True)
        css = label.styleSheet()
        # ON colour must be the constant accent (not the theme accent)
        assert CONSTANT_ROLES["button_accent"].lower() in css.lower()

    def test_primary_button_style_uses_constant_accent(self):
        """The PrimaryButton QSS block should reference button_accent
        in both dark and light stylesheets."""
        from spacr.qt.theme import stylesheet, CONSTANT_ROLES
        dark = stylesheet(theme="dark", font_scale=1.0)
        light = stylesheet(theme="light", font_scale=1.0)
        target = CONSTANT_ROLES["button_accent"].lower()
        # Both stylesheets should render the button_accent hex string
        # in the PrimaryButton block.
        assert target in dark.lower()
        assert target in light.lower()


# ---------------------------------------------------------------------------
# font-scale widens widgets
# ---------------------------------------------------------------------------

class TestScaledPx:
    def _fake_settings(self, monkeypatch, scale):
        from spacr.qt import preferences as prefs
        monkeypatch.setattr(prefs, "get_font_scale", lambda: scale)

    def test_scaled_px_scales_linearly(self, monkeypatch):
        self._fake_settings(monkeypatch, 1.5)
        from spacr.qt.preferences import scaled_px
        assert scaled_px(100) == 150
        assert scaled_px(220) == 330

    def test_scaled_px_never_zero(self, monkeypatch):
        self._fake_settings(monkeypatch, 0.001)
        from spacr.qt.preferences import scaled_px
        assert scaled_px(100) >= 1

    def test_sidebar_width_scales(self, qtbot, monkeypatch):
        """The main-window sidebar should widen when font scale is
        150 % so labels don't get clipped."""
        self._fake_settings(monkeypatch, 1.5)
        from spacr.qt.app import Sidebar
        bar = Sidebar()
        qtbot.addWidget(bar)
        # 220 * 1.5 = 330
        assert bar.width() == 330

    def test_htile_min_height_scales(self, qtbot, monkeypatch):
        self._fake_settings(monkeypatch, 2.0)
        from spacr.qt.widgets.tile import HTile
        tile = HTile("Test", "desc")
        qtbot.addWidget(tile)
        assert tile.minimumHeight() == 144   # 72 * 2


# ---------------------------------------------------------------------------
# Verbose logger toggle
# ---------------------------------------------------------------------------

@pytest.fixture
def _isolated_prefs(tmp_path, monkeypatch):
    """Redirect QSettings so tests don't clobber user prefs."""
    from PySide6.QtCore import QSettings
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope,
                       str(tmp_path))
    yield


class TestVerboseLoggerPref:
    def test_defaults_off(self, _isolated_prefs):
        from spacr.qt.preferences import get_verbose_logging
        assert get_verbose_logging() is False

    def test_set_and_get_true(self, _isolated_prefs):
        from spacr.qt.preferences import (
            get_verbose_logging, set_verbose_logging,
        )
        set_verbose_logging(True)
        assert get_verbose_logging() is True

    def test_apply_verbose_flips_spacr_logger_level(self,
                                                       _isolated_prefs):
        from spacr.qt.verbose_logger import apply_verbose_logging
        apply_verbose_logging(True)
        assert logging.getLogger("spacr").level == logging.DEBUG
        apply_verbose_logging(False)
        assert logging.getLogger("spacr").level == logging.INFO

    def test_console_target_receives_records(self, qtbot, _isolated_prefs):
        """When a target is registered, log records should be echoed."""
        from spacr.qt.verbose_logger import (
            apply_verbose_logging, register_console_target,
        )
        class _FakeConsole:
            def __init__(self):
                self.lines = []
            def append_stdout(self, s):
                self.lines.append(s)
        fake = _FakeConsole()
        register_console_target(fake)
        apply_verbose_logging(True)
        logging.getLogger("spacr").debug("hello world")
        # At least one line should contain our message
        assert any("hello world" in ln for ln in fake.lines)

    def test_file_handler_creates_log_and_appends_records(
            self, tmp_path, monkeypatch, _isolated_prefs):
        """Every apply_verbose_logging call attaches (once) a rotating
        file handler that writes into ~/.spacr/logs. Tests override
        the log dir via SPACR_LOG_DIR."""
        monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path))
        # Reset the module-level file handler so it re-attaches at the
        # tmp path (some earlier test may already have primed it).
        from spacr.qt import verbose_logger as vl
        if vl._file_handler is not None:
            for name in vl._ATTACHED_LOGGERS:
                logging.getLogger(name).removeHandler(vl._file_handler)
            vl._file_handler.close()
            vl._file_handler = None
        vl.apply_verbose_logging(True)
        logging.getLogger("spacr").info("hello world from a test")
        # Force flush so the assertion sees the record.
        if vl._file_handler is not None:
            vl._file_handler.flush()
        log_file = vl.current_log_file()
        assert log_file.exists()
        assert "hello world from a test" in log_file.read_text()

    def test_dialog_carries_toggle(self, qtbot, _isolated_prefs):
        """The Preferences dialog should surface the verbose toggle
        with the current pref value."""
        from spacr.qt.preferences import (
            PreferencesDialog, set_verbose_logging,
        )
        set_verbose_logging(True)
        dlg = PreferencesDialog()
        qtbot.addWidget(dlg)
        # Find the QCheckBox with "verbose" in the text
        from PySide6.QtWidgets import QCheckBox
        checks = dlg.findChildren(QCheckBox)
        assert checks, "no QCheckBox found in Preferences dialog"
        matching = [c for c in checks if "verbose" in c.text().lower()]
        assert matching, "no 'verbose' checkbox found"
        assert matching[0].isChecked() is True
