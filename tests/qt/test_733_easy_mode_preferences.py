"""Startup preserves Navigator choices; obsolete visibility controls stay out of the UI."""
from PySide6.QtWidgets import QDialogButtonBox, QMainWindow

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


class _RefreshOwner(QMainWindow):
    def __init__(self):
        super().__init__()
        from spacr.qt.app import MainWindow

        self.REBUILT_FOR = MainWindow.REBUILT_FOR
        self.refreshed = []
        self.dock_applied = []
        self.dock_refreshes = 0

    def refresh_theme(self):
        from spacr.qt import preferences
        self.refreshed.append((preferences.get_easy_mode(),
                               preferences.get_easy_mode_dock()))

    def refresh_after_preferences(self, changed):
        from spacr.qt.app import MainWindow

        return MainWindow.refresh_after_preferences(self, changed)

    def apply_dock_mode(self):
        from spacr.qt import preferences

        self.dock_applied.append((preferences.get_easy_mode(),
                                  preferences.get_easy_mode_dock()))

    def _backdrop_the_dock_column(self):
        self.dock_refreshes += 1


def test_navigator_defaults_on_and_preserves_ai_and_normal_dock(private_ai_profile):
    from spacr import crashreport

    preferences, terms, settings = private_ai_profile
    preferences._set_spacr_ai_enabled(False)
    preferences.set_dock_mode("locked")
    agreement = terms.agreement_record()
    before = preferences._window_signature()
    assert preferences.get_easy_mode() is True
    assert preferences.get_easy_mode_dock() is False
    preferences.set_easy_mode(False)
    assert preferences._window_changes(before, preferences._window_signature()) == {"easy_mode"}
    preferences.set_easy_mode(True)
    before = preferences._window_signature()
    preferences.set_easy_mode_dock(True)
    assert preferences._window_changes(before, preferences._window_signature()) == {"easy_mode_dock"}
    assert preferences._settings().value("prefs/easy_mode", type=bool) is True
    assert preferences._settings().value("prefs/easy_mode_dock", type=bool) is True
    assert preferences.get_easy_mode() is True
    assert preferences.get_easy_mode_dock() is True
    preferences._settings().setValue("prefs/easy_mode", "false")
    preferences._settings().setValue("prefs/easy_mode_dock", "false")
    assert preferences.get_easy_mode() is False
    assert preferences.get_easy_mode_dock() is False
    assert preferences.get_dock_mode() == "locked"
    assert preferences.get_preferred_provider() == "none"
    assert crashreport.error_report_mode() == "off"
    assert terms.agreement_record() == agreement


def test_startup_omits_navigator_without_changing_default_or_saved_choice(
        qtbot, private_ai_profile):
    from PySide6.QtWidgets import QLabel

    from spacr.qt.setup_screen import questions
    from spacr.qt.widgets.setup_dialog import GROUPS, SetupDialog
    from spacr.qt.widgets.setup_slides import SLIDES, SetupSlides

    preferences, terms, settings = private_ai_profile
    assert 'easy_mode' not in {question[0] for question in questions()}
    assert all('easy_mode' not in group for _, group in GROUPS)
    assert all('easy_mode' not in group for _, _, group in SLIDES)
    assert preferences.get_easy_mode()
    for saved in (False, True):
        preferences.set_easy_mode(saved)
        for screen in (SetupDialog(), SetupSlides()):
            qtbot.addWidget(screen)
            assert 'easy_mode' not in screen._editors
            if isinstance(screen, SetupSlides):
                prose = ' '.join(label.text() for label in screen.findChildren(
                    QLabel, 'StartupAiGuidance'))
                assert all(word in prose for word in (
                    'daily allowance', 'Claude', 'Codex', 'Gemini', 'CLI',
                    'module console', 'Enter', 'unmatched', 'lower description'))
            screen.reject()
            assert preferences.get_easy_mode() is saved


def test_preferences_save_and_cancel_preserve_legacy_dock_values_without_controls(
        qtbot, private_ai_profile):
    from spacr.qt import preferences
    from spacr.qt.widgets.setup_slides import SetupSlides
    from spacr.qt.widgets.toggle import Toggle

    owner = _RefreshOwner()
    qtbot.addWidget(owner)
    owner.show()
    preferences.set_dock_mode("hidden")
    preferences.set_easy_mode(True)
    dialog = preferences.PreferencesDialog(parent=owner)
    qtbot.addWidget(dialog)
    easy = dialog.findChild(Toggle, "EasyMode")
    assert easy is None and dialog.findChild(Toggle, "EasyModeDock") is None
    from PySide6.QtWidgets import QLabel, QComboBox
    assert not any(label.text() in ('App dock', 'Show app dock in Navigator')
                   for label in dialog.findChildren(QLabel))
    assert not any(combo.findData('hidden') >= 0 and combo.findData('locked') >= 0
                   for combo in dialog.findChildren(QComboBox))
    assert preferences.get_easy_mode() and not preferences.get_easy_mode_dock()
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert preferences.get_easy_mode() and not preferences.get_easy_mode_dock()
    assert owner.refreshed == [(True, False)]
    assert preferences.get_dock_mode() == "hidden"
    reopened = preferences.PreferencesDialog(parent=owner)
    qtbot.addWidget(reopened)
    assert reopened.findChild(Toggle, "EasyMode") is None
    assert reopened.findChild(Toggle, "EasyModeDock") is None
    reopened.reject()
    assert preferences.get_easy_mode() and not preferences.get_easy_mode_dock()
    startup = SetupSlides()
    qtbot.addWidget(startup)
    assert "easy_mode" not in startup._editors

    startup.close()
