"""Startup AI choices share provider/report state without granting consent."""
import pytest

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QFormLayout, QRadioButton


@pytest.fixture
def private_ai_profile(tmp_path, monkeypatch, qtbot):
    from spacr.qt import preferences, terms
    from spacr.qt.ai import settings
    from spacr.qt.widgets import setup_slides

    path = str(tmp_path / "profile.ini")
    store = lambda: QSettings(path, QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", store)
    monkeypatch.setattr(settings, "_settings", store)
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "spacr-home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(setup_slides, "graphics_card", lambda: (False, ""))
    from PySide6.QtCore import QCoreApplication, QEvent
    from shiboken6 import isValid
    opened = []
    original_add = qtbot.addWidget

    def add_widget(widget, *args, **kwargs):
        opened.append(widget)
        return original_add(widget, *args, **kwargs)

    monkeypatch.setattr(qtbot, "addWidget", add_widget)
    yield preferences, terms, settings
    for widget in reversed(opened):
        if isValid(widget):
            widget.close()
            widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_startup_ai_defaults_on_immediately_below_hash_in_both_layouts(qtbot, private_ai_profile):
    from spacr.qt.setup_screen import questions
    from spacr.qt.widgets.setup_dialog import GROUPS, SetupDialog
    from spacr.qt.widgets.setup_slides import SLIDES, SetupSlides
    from spacr.qt.widgets.toggle import Toggle

    preferences, terms, settings = private_ai_profile
    before = terms.agreement_record()
    keys = [question[0] for question in questions()]
    assert keys.index("spacr_ai") == keys.index("hash_inputs") + 1
    assert next(question for question in questions() if question[0] == "spacr_ai")[1] == "spaCR AI"
    assert next(keys for title, keys in GROUPS if "hash_inputs" in keys)[-2:] == ("hash_inputs", "spacr_ai")
    assert next(keys for title, blurb, keys in SLIDES if "hash_inputs" in keys)[-2:] == ("hash_inputs", "spacr_ai")
    slides = SetupSlides()
    qtbot.addWidget(slides)
    editor = slides._editors["spacr_ai"]
    assert isinstance(editor, Toggle) and editor.isChecked()
    hash_editor = slides._editors["hash_inputs"]
    form = editor.parentWidget().layout()
    assert isinstance(form, QFormLayout)
    assert form.getWidgetPosition(editor)[0] == form.getWidgetPosition(hash_editor)[0] + 1
    dialog = SetupDialog()
    qtbot.addWidget(dialog)
    assert dialog._editors["spacr_ai"].isChecked()
    assert preferences.get_preferred_provider() == ""
    assert terms.agreement_record() == before


def test_startup_off_persists_none_and_reports_off_without_accepting_terms(qtbot, private_ai_profile):
    from spacr import crashreport
    from spacr.qt.widgets.setup_slides import SetupSlides

    preferences, terms, settings = private_ai_profile
    before = terms.agreement_record()
    slides = SetupSlides()
    qtbot.addWidget(slides)
    slides._editors["spacr_ai"].setChecked(False)
    slides.reject()
    assert preferences.get_preferred_provider() == "none"
    assert crashreport.error_report_mode() == "off"
    assert terms.agreement_record() == before
    reopened = SetupSlides()
    qtbot.addWidget(reopened)
    assert not reopened._editors["spacr_ai"].isChecked()
    assert preferences._settings().value("ai/preferred_provider") == "none"


def test_enabling_startup_ai_preserves_explicit_report_off_and_existing_vendor(private_ai_profile):
    from spacr import crashreport
    from spacr.qt.ai.providers import HOSTED_PROVIDER_NAME

    preferences, terms, settings = private_ai_profile
    before = terms.agreement_record()
    preferences._set_spacr_ai_enabled(False)
    preferences._set_spacr_ai_enabled(True)
    assert preferences.get_preferred_provider() == HOSTED_PROVIDER_NAME
    assert preferences._get_spacr_ai_enabled()
    assert crashreport.error_report_mode() == "off"
    preferences.set_preferred_provider("claude")
    preferences._set_spacr_ai_enabled(True)
    assert preferences.get_preferred_provider() == "claude"
    assert crashreport.error_report_mode() == "off"
    assert terms.agreement_record() == before


def test_preferences_none_can_save_sending_mode_without_enabling_ai(qtbot, private_ai_profile, monkeypatch):
    from spacr import crashreport

    preferences, terms, settings = private_ai_profile
    terms.record_agreement()
    accepted = terms.agreement_record()
    preferences.set_issue_prompt_mode(preferences.ISSUE_PROMPT_ALWAYS)
    settings.set_auto_file_issues(True)
    preferences._set_spacr_ai_enabled(False)
    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    none = dialog.findChild(QRadioButton, "AiProvider_none")
    assert none is not None and none.isChecked()
    reports = dialog.findChild(QComboBox, "AiErrorReport")
    reports.setCurrentIndex(reports.findData("all"))
    assert crashreport.error_report_mode() == "off"
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert preferences.get_preferred_provider() == "none"
    assert not preferences._get_spacr_ai_enabled()
    assert crashreport.error_report_mode() == "all"
    assert terms.agreement_record() == accepted
    with monkeypatch.context() as policy_network:
        policy_network.setattr(crashreport, "network_allowed", lambda: True)
        policy, explain, send = crashreport._automatic_error_policy(crashreport.KIND_FAULT)
    assert policy["provider"] == "none"
    assert not explain and send
    reopened = preferences.PreferencesDialog()
    qtbot.addWidget(reopened)
    assert reopened.findChild(QRadioButton, "AiProvider_none").isChecked()
    assert reopened.findChild(QComboBox, "AiErrorReport").currentData() == "all"
