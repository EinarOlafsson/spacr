"""Easy mode shares one saved choice across startup and Preferences."""
from PySide6.QtWidgets import QDialogButtonBox, QFormLayout, QMainWindow, QMessageBox

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


def test_easy_choices_default_off_and_preserve_ai_and_normal_dock(private_ai_profile):
    from spacr import crashreport

    preferences, terms, settings = private_ai_profile
    preferences._set_spacr_ai_enabled(False)
    preferences.set_dock_mode("locked")
    agreement = terms.agreement_record()
    before = preferences._window_signature()
    assert preferences.get_easy_mode() is False
    assert preferences.get_easy_mode_dock() is False
    preferences.set_easy_mode(True)
    assert preferences._window_changes(before, preferences._window_signature()) == {"easy_mode"}
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


def test_both_startup_layouts_put_easy_mode_directly_below_ai_and_share_saved_answers(
        qtbot, private_ai_profile):
    from spacr.qt.setup_screen import questions
    from spacr.qt.widgets.setup_dialog import GROUPS, SetupDialog
    from spacr.qt.widgets.setup_slides import SLIDES, SetupSlides
    from spacr.qt.widgets.toggle import Toggle

    preferences, terms, settings = private_ai_profile
    preferences._set_spacr_ai_enabled(False)
    keys = [question[0] for question in questions()]
    assert keys.index("easy_mode") == keys.index("spacr_ai") + 1
    assert next(group for title, group in GROUPS if "spacr_ai" in group)[-2:] == (
        "spacr_ai", "easy_mode")
    assert next(group for title, blurb, group in SLIDES if "spacr_ai" in group)[-2:] == (
        "spacr_ai", "easy_mode")
    slides = SetupSlides()
    qtbot.addWidget(slides)
    easy = slides._editors["easy_mode"]
    assert isinstance(easy, Toggle) and not easy.isChecked()
    form = easy.parentWidget().layout()
    assert isinstance(form, QFormLayout)
    assert form.getWidgetPosition(easy)[0] == form.getWidgetPosition(
        slides._editors["spacr_ai"])[0] + 1
    easy.setChecked(True)
    slides.reject()
    assert preferences.get_easy_mode() is True
    dialog = SetupDialog()
    qtbot.addWidget(dialog)
    easy = dialog._editors["easy_mode"]
    assert easy.isChecked()
    form = next(layout for layout in dialog.findChildren(QFormLayout)
                if layout.getWidgetPosition(easy)[0] >= 0)
    assert form.getWidgetPosition(easy)[0] == form.getWidgetPosition(
        dialog._editors["spacr_ai"])[0] + 1
    easy.setChecked(False)
    dialog.accept()
    assert preferences.get_easy_mode() is False
    assert preferences.get_preferred_provider() == "none"
    assert preferences.get_easy_mode_dock() is False


def test_preferences_save_refreshes_owner_and_cancel_preserves_both_saved_choices(
        qtbot, private_ai_profile):
    from spacr.qt import preferences
    from spacr.qt.widgets.setup_slides import SetupSlides
    from spacr.qt.widgets.toggle import Toggle

    owner = _RefreshOwner()
    qtbot.addWidget(owner)
    owner.show()
    preferences.set_dock_mode("hidden")
    dialog = preferences.PreferencesDialog(parent=owner)
    qtbot.addWidget(dialog)
    easy = dialog.findChild(Toggle, "EasyMode")
    dock = dialog.findChild(Toggle, "EasyModeDock")
    assert easy is not None and dock is not None
    assert not easy.isChecked() and not dock.isChecked()
    easy.setChecked(True)
    dock.setChecked(True)
    assert not preferences.get_easy_mode() and not preferences.get_easy_mode_dock()
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert preferences.get_easy_mode() and preferences.get_easy_mode_dock()
    assert owner.refreshed == [(True, True)]
    assert preferences.get_dock_mode() == "hidden"
    reopened = preferences.PreferencesDialog(parent=owner)
    qtbot.addWidget(reopened)
    assert reopened.findChild(Toggle, "EasyMode").isChecked()
    assert reopened.findChild(Toggle, "EasyModeDock").isChecked()
    reopened.findChild(Toggle, "EasyMode").setChecked(False)
    reopened.findChild(Toggle, "EasyModeDock").setChecked(False)
    reopened.reject()
    assert preferences.get_easy_mode() and preferences.get_easy_mode_dock()
    startup = SetupSlides()
    qtbot.addWidget(startup)
    assert startup._editors["easy_mode"].isChecked()

    startup.close()
    applied = preferences.PreferencesDialog(parent=owner)
    qtbot.addWidget(applied)
    applied.show()
    qtbot.waitExposed(applied)
    assert applied._window_signature_seen is not None
    buttons = applied.findChild(QDialogButtonBox)
    applied.findChild(Toggle, "EasyModeDock").setChecked(False)
    buttons.button(QDialogButtonBox.Apply).click()
    question = applied._apply_confirmation
    assert question is not None
    next(button for button in question.buttons()
         if question.buttonRole(button) == QMessageBox.AcceptRole).click()
    assert applied._apply_confirmation is None
    qtbot.waitUntil(lambda: owner.dock_applied == [(True, False)])
    assert owner.dock_refreshes == 1
    assert owner.refreshed == [(True, True)]
    assert preferences.get_easy_mode() and not preferences.get_easy_mode_dock()
    applied.findChild(Toggle, "EasyMode").setChecked(False)
    buttons.button(QDialogButtonBox.Apply).click()
    question = applied._apply_confirmation
    assert question is not None
    next(button for button in question.buttons()
         if question.buttonRole(button) == QMessageBox.AcceptRole).click()
    assert applied._apply_confirmation is None
    qtbot.waitUntil(lambda: owner.refreshed == [(True, True), (False, False)])
    assert not preferences.get_easy_mode() and not preferences.get_easy_mode_dock()
    assert preferences.get_dock_mode() == "hidden"
    applied.reject()
