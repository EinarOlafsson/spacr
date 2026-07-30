"""Offline tests for ``spacr.qt.widgets.ai_chat_panel``.

No CLI is ever spawned and no socket is ever opened: every provider is a
scripted double, ``ai_module.configured_providers`` / ``get_provider`` are
redirected, and QSettings is pointed at a temp .ini file.

Three regressions are pinned here (all were live bugs):

* ``refresh_provider_combo`` left the Send button disabled forever after
  the empty-state branch had disabled it — so "install a CLI, hit
  Refresh" produced a chat you could not send from.
* ``clear_chat`` kept a reference to the in-flight assistant bubble it
  had just deleted, so the next streamed chunk raised
  "Internal C++ object already deleted" inside a Qt slot.
* ``_start_stream`` created the QThread with no Qt parent and
  ``_on_stream_finished`` then dropped the last Python reference to it —
  the documented ``QThread: Destroyed while thread is still running``
  abort.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QLabel, QLineEdit


# ---------------------------------------------------------------------------
# Doubles + fixtures
# ---------------------------------------------------------------------------

class _ScriptedProvider:
    """ChatProvider-shaped double; streams a fixed list of chunks."""

    install_hint = "pip install nothing"
    login_command = "fake login"

    def __init__(self, name="claude", label="Fake Claude", chunks=("hi",),
                 installed=True, raise_exc=None):
        self.name = name
        self.label = label
        self.cli_name = name
        self._chunks = list(chunks)
        self._installed = installed
        self._raise = raise_exc
        self.cancelled = 0
        self.seen = []

    # -- ChatProvider surface -----------------------------------------
    def is_installed(self):
        return self._installed

    def is_logged_in(self):
        return self._installed

    def is_configured(self):
        return self._installed

    def cancel_stream(self):
        self.cancelled += 1

    def stream_chat(self, messages, system="", model=None):
        self.seen.append({"messages": list(messages), "system": system,
                          "model": model})
        if self._raise is not None:
            raise self._raise
        for c in self._chunks:
            yield c


@pytest.fixture()
def ai_env(monkeypatch, tmp_path, qt_theme_applied):
    """Isolate QSettings and take control of the provider registry."""
    from PySide6.QtCore import QSettings
    from spacr.qt import ai as ai_module
    from spacr.qt.ai import github_auth
    from spacr.qt.ai import settings as ai_settings

    store = QSettings(str(tmp_path / "panel.ini"), QSettings.IniFormat)
    monkeypatch.setattr(ai_settings, "_settings", lambda: store)
    monkeypatch.setattr(github_auth, "_settings", lambda: store)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(github_auth, "_gh_cli_token", lambda: "")

    state = {"providers": []}

    def _configured():
        return [p for p in state["providers"] if p.is_configured()]

    def _get(name):
        for p in state["providers"]:
            if p.name == name:
                return p
        return None

    monkeypatch.setattr(ai_module, "configured_providers", _configured)
    monkeypatch.setattr(ai_module, "get_provider", _get)
    monkeypatch.setattr(ai_module, "list_providers",
                        lambda: list(state["providers"]))
    state["settings"] = ai_settings
    state["github_auth"] = github_auth
    return state


@pytest.fixture()
def panel(ai_env, qtbot):
    """AIChatPanel with one configured fake provider."""
    from spacr.qt.widgets.ai_chat_panel import AIChatPanel

    ai_env["providers"] = [_ScriptedProvider(chunks=["Hello ", "world"])]
    p = AIChatPanel()
    qtbot.addWidget(p)
    yield p
    p.shutdown()


# ---------------------------------------------------------------------------
# _current_system_prompt
# ---------------------------------------------------------------------------

def test_current_system_prompt_follows_the_user_override(ai_env):
    from spacr.qt.ai.prompts import default_system_prompt
    from spacr.qt.widgets.ai_chat_panel import _current_system_prompt

    assert _current_system_prompt() == default_system_prompt()
    ai_env["settings"].set_system_prompt("Reply in exactly one sentence.")
    assert _current_system_prompt() == "Reply in exactly one sentence."


# ---------------------------------------------------------------------------
# _MessageBubble
# ---------------------------------------------------------------------------

def test_user_bubble_is_right_aligned_with_the_user_object_name(qtbot,
                                                                qt_theme_applied):
    from spacr.qt.widgets.ai_chat_panel import _MessageBubble

    b = _MessageBubble("user", "what went wrong?")
    qtbot.addWidget(b)
    assert b.role == "user"
    assert b._text.text() == "what went wrong?"
    assert b._text.objectName() == "ChatBubbleUser"
    lay = b.layout()
    # stretch first, label second => the bubble hugs the right edge
    assert lay.itemAt(0).spacerItem() is not None
    assert lay.itemAt(1).widget() is b._text


def test_assistant_bubble_is_left_aligned(qtbot, qt_theme_applied):
    from spacr.qt.widgets.ai_chat_panel import _MessageBubble

    b = _MessageBubble("assistant", "because channels was None")
    qtbot.addWidget(b)
    assert b._text.objectName() == "ChatBubbleAssistant"
    lay = b.layout()
    assert lay.itemAt(0).widget() is b._text
    assert lay.itemAt(1).spacerItem() is not None


def test_bubble_is_selectable_wrapped_and_width_capped(qtbot, qt_theme_applied):
    from spacr.qt.widgets.ai_chat_panel import _MessageBubble

    b = _MessageBubble("assistant", "x")
    qtbot.addWidget(b)
    assert b._text.wordWrap() is True
    assert b._text.maximumWidth() == 720
    flags = b._text.textInteractionFlags()
    assert flags & Qt.TextSelectableByMouse       # users copy AI answers
    assert flags & Qt.LinksAccessibleByMouse
    assert b._text.openExternalLinks() is True


def test_bubble_set_text_replaces_the_body(qtbot, qt_theme_applied):
    from spacr.qt.widgets.ai_chat_panel import _MessageBubble

    b = _MessageBubble("assistant", "…")
    qtbot.addWidget(b)
    b.set_text("streamed so far")
    assert b._text.text() == "streamed so far"
    b.set_text("")
    assert b._text.text() == ""


def test_unknown_role_renders_as_assistant(qtbot, qt_theme_applied):
    from spacr.qt.widgets.ai_chat_panel import _MessageBubble

    b = _MessageBubble("system", "ctx")
    qtbot.addWidget(b)
    assert b._text.objectName() == "ChatBubbleAssistant"


# ---------------------------------------------------------------------------
# _ChatInput key handling
# ---------------------------------------------------------------------------

def test_plain_enter_submits_without_inserting_a_newline(qtbot,
                                                          qt_theme_applied):
    from spacr.qt.widgets.ai_chat_panel import _ChatInput

    inp = _ChatInput()
    qtbot.addWidget(inp)
    inp.setPlainText("send me")
    with qtbot.waitSignal(inp.submitted, timeout=1000):
        QTest.keyClick(inp, Qt.Key_Return)
    assert inp.toPlainText() == "send me"      # unchanged


def test_keypad_enter_also_submits(qtbot, qt_theme_applied):
    from spacr.qt.widgets.ai_chat_panel import _ChatInput

    inp = _ChatInput()
    qtbot.addWidget(inp)
    inp.setPlainText("typed on the numpad")
    with qtbot.waitSignal(inp.submitted, timeout=1000):
        QTest.keyClick(inp, Qt.Key_Enter)
    assert inp.toPlainText() == "typed on the numpad"


def test_shift_enter_inserts_a_newline_and_does_not_submit(qtbot,
                                                            qt_theme_applied):
    from spacr.qt.widgets.ai_chat_panel import _ChatInput

    inp = _ChatInput()
    qtbot.addWidget(inp)
    fired = []
    inp.submitted.connect(lambda: fired.append(1))
    inp.setPlainText("line1")
    from PySide6.QtGui import QTextCursor
    inp.moveCursor(QTextCursor.MoveOperation.End)
    QTest.keyClick(inp, Qt.Key_Return, Qt.ShiftModifier)
    QTest.keyClicks(inp, "line2")
    assert inp.toPlainText() == "line1\nline2"
    assert fired == []


def test_ordinary_keys_type_normally(qtbot, qt_theme_applied):
    from spacr.qt.widgets.ai_chat_panel import _ChatInput

    inp = _ChatInput()
    qtbot.addWidget(inp)
    fired = []
    inp.submitted.connect(lambda: fired.append(1))
    QTest.keyClicks(inp, "abc")
    assert inp.toPlainText() == "abc"
    assert fired == []


def test_chat_input_is_plain_text_only_and_height_bounded(qtbot,
                                                           qt_theme_applied):
    from spacr.qt.widgets.ai_chat_panel import _ChatInput

    inp = _ChatInput()
    qtbot.addWidget(inp)
    assert inp.acceptRichText() is False
    assert inp.minimumHeight() == 56
    assert inp.maximumHeight() == 140
    inp.insertHtml("<b>bold</b>")
    assert "<b>" not in inp.toPlainText()


# ---------------------------------------------------------------------------
# Provider combo / empty state
# ---------------------------------------------------------------------------

def test_panel_lists_every_configured_provider(ai_env, qtbot):
    from spacr.qt.widgets.ai_chat_panel import AIChatPanel

    ai_env["providers"] = [
        _ScriptedProvider("claude", "Claude (via Claude Code)"),
        _ScriptedProvider("codex", "ChatGPT (via Codex CLI)"),
        _ScriptedProvider("gemini", "Gemini", installed=False),
    ]
    p = AIChatPanel()
    qtbot.addWidget(p)
    combo = p._provider_combo
    assert combo.count() == 2                      # gemini isn't installed
    assert [combo.itemData(i) for i in range(2)] == ["claude", "codex"]
    assert combo.itemText(0) == "Claude (via Claude Code)"
    assert p._current_provider().name == "claude"


def test_refresh_reenables_send_after_a_cli_appears(ai_env, qtbot):
    """REGRESSION: install a CLI, hit Providers > Refresh — Send stayed
    permanently disabled because only the empty-state branch touched it."""
    from spacr.qt.widgets.ai_chat_panel import AIChatPanel

    ai_env["providers"] = []
    p = AIChatPanel()
    qtbot.addWidget(p)
    assert p._stack.currentWidget() is p._empty_state
    assert p._btn_send.isEnabled() is False
    assert p._input.isEnabled() is False

    ai_env["providers"] = [_ScriptedProvider()]
    p.refresh_provider_combo()

    assert p._stack.currentWidget() is p._chat_scroll
    assert p._input.isEnabled() is True
    assert p._btn_send.isEnabled() is True, \
        "Send button stayed disabled after a provider became available"
    assert p._btn_send.text() == "Send"


def test_refresh_back_to_empty_disables_input_and_send(ai_env, qtbot):
    from spacr.qt.widgets.ai_chat_panel import AIChatPanel

    ai_env["providers"] = [_ScriptedProvider()]
    p = AIChatPanel()
    qtbot.addWidget(p)
    assert p._btn_send.isEnabled()

    ai_env["providers"] = []          # user uninstalled the CLI
    p.refresh_provider_combo()
    assert p._provider_combo.count() == 0
    assert p._stack.currentWidget() is p._empty_state
    assert p._input.isEnabled() is False
    assert p._btn_send.isEnabled() is False
    assert p._current_provider() is None


def test_refresh_does_not_emit_combo_signals_while_rebuilding(ai_env, qtbot):
    from spacr.qt.widgets.ai_chat_panel import AIChatPanel

    ai_env["providers"] = [_ScriptedProvider()]
    p = AIChatPanel()
    qtbot.addWidget(p)
    fired = []
    p._provider_combo.currentIndexChanged.connect(fired.append)
    ai_env["providers"] = [_ScriptedProvider("codex", "Codex")]
    p.refresh_provider_combo()
    assert fired == []


def test_current_provider_is_none_when_registry_forgets_it(ai_env, qtbot):
    from spacr.qt.widgets.ai_chat_panel import AIChatPanel

    ai_env["providers"] = [_ScriptedProvider()]
    p = AIChatPanel()
    qtbot.addWidget(p)
    ai_env["providers"] = []          # combo still holds "claude"
    assert p._provider_combo.currentData() == "claude"
    assert p._current_provider() is None


# ---------------------------------------------------------------------------
# Providers dialog
# ---------------------------------------------------------------------------

def test_open_keys_dialog_refreshes_on_accept(ai_env, qtbot, monkeypatch):
    from spacr.qt.widgets import ai_chat_panel as mod

    ai_env["providers"] = []
    p = mod.AIChatPanel()
    qtbot.addWidget(p)
    assert p._btn_send.isEnabled() is False

    # The user installs a CLI while the (non-modal, stubbed) dialog is open.
    def _exec(self):
        ai_env["providers"] = [_ScriptedProvider()]
        return QDialog.Accepted

    monkeypatch.setattr(mod._ProvidersDialog, "exec", _exec)
    p._on_open_keys_dialog()
    assert p._provider_combo.count() == 1
    assert p._btn_send.isEnabled() is True


def test_open_keys_dialog_does_not_refresh_on_reject(ai_env, qtbot,
                                                      monkeypatch):
    from spacr.qt.widgets import ai_chat_panel as mod

    ai_env["providers"] = []
    p = mod.AIChatPanel()
    qtbot.addWidget(p)

    def _exec(self):
        ai_env["providers"] = [_ScriptedProvider()]
        return QDialog.Rejected

    monkeypatch.setattr(mod._ProvidersDialog, "exec", _exec)
    p._on_open_keys_dialog()
    assert p._provider_combo.count() == 0     # Close != Refresh


def test_dialog_speed_combo_persists_the_selection(ai_env, qtbot):
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    ai_env["providers"] = [_ScriptedProvider()]
    ai_env["settings"].set_response_speed("deep")
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    assert dlg._speed_combo.currentData() == "deep"      # preselected

    idx = [dlg._speed_combo.itemData(i)
           for i in range(dlg._speed_combo.count())].index("fast")
    dlg._speed_combo.setCurrentIndex(idx)
    assert ai_env["settings"].get_response_speed() == "fast"
    assert ai_env["settings"].provider_args("claude") == ["--model", "haiku"]


def test_dialog_speed_combo_offers_exactly_the_valid_speeds(ai_env, qtbot):
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    ai_env["providers"] = []
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    values = [dlg._speed_combo.itemData(i)
              for i in range(dlg._speed_combo.count())]
    assert values == list(ai_env["settings"].VALID_SPEEDS)


def test_dialog_tolerates_a_persisted_speed_it_cannot_offer(ai_env, qtbot,
                                                             monkeypatch):
    """Guards against SPEED_MAP and the dialog's hard-coded three rows
    drifting apart: an unknown persisted speed must leave the combo on its
    first row without silently rewriting the stored value."""
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    ai_env["providers"] = []
    monkeypatch.setattr(ai_env["settings"], "get_response_speed",
                        lambda: "turbo")
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    assert dlg._speed_combo.currentIndex() == 0
    assert dlg._speed_combo.currentData() == "fast"
    # No currentIndexChanged fired, so nothing was persisted behind the
    # user's back.
    assert ai_env["settings"]._settings().value("ai/response_speed") is None


def test_dialog_speed_handler_ignores_an_unknown_value(ai_env, qtbot):
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    ai_env["providers"] = []
    ai_env["settings"].set_response_speed("balanced")
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    dlg._speed_combo.addItem("Experimental", "turbo")
    dlg._speed_combo.setCurrentIndex(dlg._speed_combo.count() - 1)
    # set_response_speed would raise ValueError; the guard must skip it
    assert ai_env["settings"].get_response_speed() == "balanced"


def test_dialog_auto_issue_checkbox_persists(ai_env, qtbot):
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    ai_env["providers"] = []
    assert ai_env["settings"].get_auto_file_issues() is False
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    assert dlg._auto_issue_chk.isChecked() is False
    dlg._auto_issue_chk.setChecked(True)
    assert ai_env["settings"].get_auto_file_issues() is True

    dlg2 = _ProvidersDialog()
    qtbot.addWidget(dlg2)
    assert dlg2._auto_issue_chk.isChecked() is True      # reloaded


def test_dialog_route_errors_checkbox_defaults_on_and_persists(ai_env, qtbot):
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    ai_env["providers"] = []
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    assert dlg._route_errors_chk.isChecked() is True
    dlg._route_errors_chk.setChecked(False)
    assert ai_env["settings"].get_route_errors_through_ai() is False


def test_dialog_github_token_is_masked_saved_and_cleared(ai_env, qtbot):
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    ai_env["providers"] = []
    ga = ai_env["github_auth"]
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    # A PAT must never be shown in the clear.
    assert dlg._gh_token.echoMode() == QLineEdit.Password
    assert dlg._gh_token.text() == ""
    assert "Not signed in" in dlg._gh_status.text()

    dlg._gh_token.setText("  ghp_typed_by_user  ")
    dlg._on_save_github_token()
    assert ga.get_stored_token() == "ghp_typed_by_user"   # trimmed
    assert "Signed in via a saved token" in dlg._gh_status.text()

    dlg._on_clear_github_token()
    assert ga.get_stored_token() == ""
    assert dlg._gh_token.text() == ""
    assert "Not signed in" in dlg._gh_status.text()


def test_dialog_prefills_an_already_stored_token(ai_env, qtbot):
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    ai_env["providers"] = []
    ai_env["github_auth"].set_stored_token("ghp_previously_saved")
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    assert dlg._gh_token.text() == "ghp_previously_saved"
    assert "Signed in" in dlg._gh_status.text()


@pytest.mark.parametrize("source,blurb", [
    ("token", "a saved token"),
    ("env", "the GITHUB_TOKEN env var"),
    ("gh", "the GitHub CLI"),
    ("something-new", "something-new"),      # unknown source falls back
])
def test_dialog_github_status_names_the_auth_source(ai_env, qtbot, monkeypatch,
                                                     source, blurb):
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    ai_env["providers"] = []
    monkeypatch.setattr(ai_env["github_auth"], "auth_source", lambda: source)
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    assert blurb in dlg._gh_status.text()
    assert "issues send directly" in dlg._gh_status.text()


def test_dialog_system_prompt_save_and_reset(ai_env, qtbot):
    from spacr.qt.ai.prompts import default_system_prompt
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    ai_env["providers"] = []
    s = ai_env["settings"]
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    assert dlg._prompt_edit.toPlainText() == default_system_prompt()
    assert dlg._prompt_status.text() == "Using the default spaCR-aware prompt."

    dlg._prompt_edit.setPlainText("  Answer in haiku.  ")
    dlg._on_prompt_save()
    assert s.get_system_prompt() == "Answer in haiku."        # trimmed
    assert dlg._prompt_status.text() == \
        "Using your custom prompt (overrides default)."

    dlg._on_prompt_reset()
    assert s.is_system_prompt_overridden() is False
    assert dlg._prompt_edit.toPlainText() == default_system_prompt()
    assert dlg._prompt_status.text() == "Using the default spaCR-aware prompt."


def test_dialog_saving_a_blank_prompt_clears_the_override(ai_env, qtbot):
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    ai_env["providers"] = []
    s = ai_env["settings"]
    s.set_system_prompt("custom")
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    dlg._prompt_edit.setPlainText("    ")
    dlg._on_prompt_save()
    assert s.is_system_prompt_overridden() is False
    assert dlg._prompt_status.text() == "Using the default spaCR-aware prompt."


def test_dialog_provider_rows_show_install_state_and_commands(ai_env, qtbot):
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    ai_env["providers"] = [
        _ScriptedProvider("claude", "Claude", installed=True),
        _ScriptedProvider("codex", "Codex", installed=False),
    ]
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    texts = [w.text() for w in dlg.findChildren(QLabel)]
    joined = "\n".join(texts)
    assert "● installed" in joined
    assert "● missing" in joined
    assert "<b>Claude</b>" in texts
    assert "<b>Codex</b>" in texts
    # Both commands are offered, read-only, for copy/paste
    edits = [e for e in dlg.findChildren(QLineEdit) if e.isReadOnly()]
    values = {e.text() for e in edits}
    assert "pip install nothing" in values
    assert "fake login" in values


def test_dialog_copy_button_puts_the_command_on_the_clipboard(ai_env, qtbot):
    from PySide6.QtGui import QGuiApplication
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    ai_env["providers"] = [_ScriptedProvider()]
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    dlg._copy_to_clipboard("npm install -g fake-cli")
    cb = QGuiApplication.clipboard()
    assert cb is not None
    assert cb.text() == "npm install -g fake-cli"


def test_dialog_copy_is_a_noop_without_a_clipboard(ai_env, qtbot, monkeypatch):
    """A headless session can have no clipboard at all — copying must not
    take the dialog down."""
    from PySide6.QtGui import QGuiApplication
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    ai_env["providers"] = [_ScriptedProvider()]
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    monkeypatch.setattr(QGuiApplication, "clipboard",
                        staticmethod(lambda: None))
    assert dlg._copy_to_clipboard("never lands anywhere") is None


def test_dialog_has_both_tabs_and_refresh_close_buttons(ai_env, qtbot):
    from PySide6.QtWidgets import QDialogButtonBox, QTabWidget
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    ai_env["providers"] = []
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    tabs = dlg.findChild(QTabWidget)
    assert [tabs.tabText(i) for i in range(tabs.count())] == \
        ["Providers", "Settings"]

    box = dlg.findChild(QDialogButtonBox)
    labels = {b.text() for b in box.buttons()}
    assert labels == {"Refresh", "Close"}
    refresh = next(b for b in box.buttons() if b.text() == "Refresh")
    with qtbot.waitSignal(dlg.accepted, timeout=1000):
        refresh.click()
    assert dlg.result() == QDialog.Accepted


def test_dialog_close_button_rejects(ai_env, qtbot):
    from PySide6.QtWidgets import QDialogButtonBox
    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog

    ai_env["providers"] = []
    dlg = _ProvidersDialog()
    qtbot.addWidget(dlg)
    box = dlg.findChild(QDialogButtonBox)
    close = next(b for b in box.buttons() if b.text() == "Close")
    with qtbot.waitSignal(dlg.rejected, timeout=1000):
        close.click()
    assert dlg.result() == QDialog.Rejected


# ---------------------------------------------------------------------------
# Send / cancel button mode
# ---------------------------------------------------------------------------

def test_send_mode_toggles_label_style_and_the_click_target(panel):
    calls = []
    panel._send_from_input = lambda: calls.append("send")
    panel._cancel_stream = lambda: calls.append("cancel")

    panel._set_send_mode("cancel")
    assert panel._btn_send.text() == "Cancel"
    assert panel._btn_send.objectName() == "DangerButton"
    panel._btn_send.click()
    assert calls == ["cancel"]

    panel._set_send_mode("send")
    assert panel._btn_send.text() == "Send"
    assert panel._btn_send.objectName() == "PrimaryButton"
    panel._btn_send.click()
    assert calls == ["cancel", "send"]      # exactly one handler each time


def test_send_mode_never_stacks_duplicate_handlers(panel):
    calls = []
    panel._send_from_input = lambda: calls.append("send")
    for _ in range(4):
        panel._set_send_mode("send")
    panel._btn_send.click()
    assert calls == ["send"]


@pytest.mark.parametrize("mode,label", [("cancel", "Cancel"), ("send", "Send")])
def test_send_mode_recovers_when_nothing_is_connected(panel, mode, label):
    """Switching mode with an already-empty `clicked` signal must still
    wire up the new handler.

    PySide6 answers a redundant ``disconnect()`` with a RuntimeWarning and
    ``False`` — but under ``-W error`` / ``PYTHONWARNINGS=error`` (a normal
    CI setting) that warning is raised as a ``SystemError`` instead, which
    is exactly what the ``except Exception`` guard is there for.
    """
    import warnings

    calls = []
    panel._send_from_input = lambda: calls.append("send")
    panel._cancel_stream = lambda: calls.append("cancel")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel._btn_send.clicked.disconnect()      # nothing left to remove

    with warnings.catch_warnings():
        warnings.simplefilter("error")            # redundant disconnect raises
        panel._set_send_mode(mode)

    assert panel._btn_send.text() == label
    panel._btn_send.click()
    assert calls == [mode]


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------

def test_empty_input_sends_nothing(panel):
    panel._input.setPlainText("   \n  ")
    panel._send_from_input()
    assert panel._messages == []
    assert panel._thread is None
    assert panel._input.toPlainText() == "   \n  "   # not cleared


def test_send_clears_input_appends_bubble_and_streams(panel, qtbot, ai_env):
    provider = ai_env["providers"][0]
    ai_env["settings"].set_system_prompt("Be brief.")
    panel._input.setPlainText("  why did masks fail?  ")
    panel._send_from_input()

    assert panel._input.toPlainText() == ""
    assert panel._messages[0] == {"role": "user",
                                  "content": "why did masks fail?"}
    assert panel._btn_send.text() == "Cancel"
    assert panel.is_streaming() is True

    qtbot.waitUntil(lambda: panel._thread is None, timeout=5000)
    assert provider.seen[0]["messages"] == [
        {"role": "user", "content": "why did masks fail?"}]
    assert provider.seen[0]["system"] == "Be brief."
    assert panel._messages[1] == {"role": "assistant",
                                  "content": "Hello world"}
    assert panel._status.text() == "Ready."
    assert panel._btn_send.text() == "Send"


def test_streamed_text_lands_in_the_assistant_bubble(panel, qtbot):
    from spacr.qt.widgets.ai_chat_panel import _MessageBubble

    panel._input.setPlainText("hi")
    panel._send_from_input()
    qtbot.waitUntil(lambda: panel._thread is None, timeout=5000)

    bubbles = [panel._chat_layout.itemAt(i).widget()
               for i in range(panel._chat_layout.count() - 1)]
    bubbles = [b for b in bubbles if isinstance(b, _MessageBubble)]
    assert [b.role for b in bubbles] == ["user", "assistant"]
    assert bubbles[0]._text.text() == "hi"
    assert bubbles[1]._text.text() == "Hello world"


def test_second_send_while_streaming_is_refused_with_a_hint(panel, qtbot):
    from PySide6.QtCore import QThread

    panel._thread = QThread()        # pretend a stream is in flight
    try:
        panel._input.setPlainText("second question")
        panel._send_from_input()
        assert panel._messages == []
        assert panel._input.toPlainText() == "second question"
        assert "already streaming" in panel._status.text()
    finally:
        panel._thread = None


def test_start_stream_without_a_provider_reports_and_does_nothing(panel,
                                                                   ai_env):
    ai_env["providers"] = []          # registry forgot it; combo still set
    panel._start_stream(system="s")
    assert panel._status.text() == "No provider configured."
    assert panel._thread is None
    assert panel._pending_bubble is None


def test_start_stream_parents_the_thread_to_the_panel(panel, qtbot):
    """REGRESSION: an unparented QThread whose last Python ref is dropped
    in _on_stream_finished aborts with 'Destroyed while still running'."""
    panel._append_user("go")
    panel._start_stream(system="")
    thread = panel._thread
    assert thread is not None
    assert thread.parent() is panel
    qtbot.waitUntil(lambda: panel._thread is None, timeout=5000)


def test_finished_stream_is_retired_not_dropped(panel, qtbot):
    """REGRESSION: _on_stream_finished used to just null self._thread. If
    that was the last Python reference and the OS thread hadn't fully
    wound down, Qt aborted the process."""
    panel._input.setPlainText("hi")
    panel._send_from_input()
    thread, worker = panel._thread, panel._worker
    qtbot.waitUntil(lambda: panel._thread is None, timeout=5000)
    # The pair is handed to _retired in the very same call that nulls the
    # active slots, so a reference always survives the handover.
    assert panel._retired == [(thread, worker)]

    # Once the OS thread has exited (and Qt's deleteLater has possibly
    # already reclaimed the C++ half) the next prune drops it.
    def _pruned_empty():
        panel._prune_retired()
        return panel._retired == []

    qtbot.waitUntil(_pruned_empty, timeout=5000)


def test_prune_keeps_running_threads_and_drops_finished_ones(panel):
    class _Thread:
        def __init__(self, running):
            self._running = running

        def isRunning(self):
            return self._running

    running, finished = _Thread(True), _Thread(False)
    panel._retired = [(finished, "w0"), (running, "w1"), (_Thread(False), "w2")]
    panel._prune_retired()
    assert panel._retired == [(running, "w1")]
    panel._retired = []          # stubs can't survive shutdown() teardown


def test_prune_drops_a_thread_whose_cpp_half_qt_already_deleted(panel):
    class _Deleted:
        def isRunning(self):
            raise RuntimeError("Internal C++ object already deleted.")

    panel._retired = [(_Deleted(), object())]
    panel._prune_retired()
    assert panel._retired == []


def test_repeated_sends_do_not_leak_retired_threads(panel, qtbot):
    for i in range(4):
        panel._input.setPlainText(f"turn {i}")
        panel._send_from_input()
        qtbot.waitUntil(lambda: panel._thread is None, timeout=5000)
    for _ in range(10):
        qtbot.wait(20)
    panel._prune_retired()
    assert panel._retired == []
    assert len(panel._messages) == 8          # 4 user + 4 assistant


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

def test_cancel_stream_delegates_to_the_worker(panel, ai_env):
    class _FakeWorker:
        def __init__(self):
            self.cancelled = 0

        def cancel(self):
            self.cancelled += 1

    panel._worker = _FakeWorker()
    panel._cancel_stream()
    assert panel._worker.cancelled == 1
    assert panel._status.text() == "Cancelling…"


def test_cancel_without_an_active_worker_is_a_noop(panel):
    panel._status.setText("untouched")
    panel._worker = None
    panel._cancel_stream()
    assert panel._status.text() == "untouched"


# ---------------------------------------------------------------------------
# Stage / chunk / finish handlers
# ---------------------------------------------------------------------------

def test_stage_messages_name_the_provider(panel):
    panel._on_stage_changed("connecting")
    assert panel._status.text() == "Connecting to Fake Claude…"
    panel._on_stage_changed("streaming")
    assert panel._status.text() == "Streaming from Fake Claude…"
    panel._on_stage_changed("something-else")
    assert panel._status.text() == "Streaming from Fake Claude…"   # unchanged


def test_stage_messages_survive_a_vanished_provider(panel, ai_env):
    ai_env["providers"] = []
    panel._on_stage_changed("connecting")
    assert panel._status.text() == "Connecting to …"


def test_chunks_accumulate_into_the_pending_bubble(panel):
    from spacr.qt.widgets.ai_chat_panel import _MessageBubble

    panel._pending_bubble = _MessageBubble("assistant", "…")
    panel._pending_buf = []
    for c in ("The ", "root ", "cause"):
        panel._on_chunk(c)
    assert panel._pending_buf == ["The ", "root ", "cause"]
    assert panel._pending_bubble._text.text() == "The root cause"


def test_chunks_without_a_bubble_are_still_buffered(panel):
    panel._pending_bubble = None
    panel._pending_buf = []
    panel._on_chunk("orphan")
    assert panel._pending_buf == ["orphan"]


def test_finish_ok_records_the_assistant_turn(panel):
    from spacr.qt.widgets.ai_chat_panel import _MessageBubble

    panel._messages = [{"role": "user", "content": "q"}]
    panel._pending_bubble = _MessageBubble("assistant", "partial")
    panel._pending_buf = ["partial"]
    panel._on_stream_finished(True, "partial")

    assert panel._messages[-1] == {"role": "assistant", "content": "partial"}
    assert panel._status.text() == "Ready."
    assert panel._pending_bubble is None
    assert panel._pending_buf == []
    assert panel._btn_send.text() == "Send"


def test_finish_ok_with_no_chunks_shows_the_empty_response_notice(panel):
    from spacr.qt.widgets.ai_chat_panel import _MessageBubble

    bubble = _MessageBubble("assistant", "…")
    panel._pending_bubble = bubble
    panel._pending_buf = []
    panel._on_stream_finished(True, "")
    assert bubble._text.text() == \
        "(empty response — try again or switch provider)"
    assert panel._messages[-1] == {"role": "assistant", "content": ""}
    assert panel._status.text() == "Ready."


def test_finish_failure_marks_the_bubble_and_the_status(panel):
    from spacr.qt.widgets.ai_chat_panel import _MessageBubble

    bubble = _MessageBubble("assistant", "…")
    panel._pending_bubble = bubble
    panel._pending_buf = ["half "]
    panel._on_stream_finished(False, "RuntimeError: CLI not on PATH")
    assert bubble._text.text() == "[error] RuntimeError: CLI not on PATH"
    assert panel._status.text() == "Failed: RuntimeError: CLI not on PATH"
    # a failed turn must NOT be recorded as assistant history
    assert panel._messages == []


def test_cancelled_stream_reports_failure_and_keeps_history_clean(panel,
                                                                   qtbot):
    panel._input.setPlainText("long question")
    panel._send_from_input()
    panel._cancel_stream()
    qtbot.waitUntil(lambda: panel._thread is None, timeout=5000)
    assert [m["role"] for m in panel._messages] == ["user"]
    assert panel._status.text().startswith("Failed:")
    assert panel._btn_send.text() == "Send"


def test_provider_error_surfaces_in_the_bubble(ai_env, qtbot):
    from spacr.qt.widgets.ai_chat_panel import AIChatPanel, _MessageBubble

    ai_env["providers"] = [
        _ScriptedProvider(raise_exc=RuntimeError("Could not run 'claude'"))]
    p = AIChatPanel()
    qtbot.addWidget(p)
    try:
        p._input.setPlainText("hi")
        p._send_from_input()
        qtbot.waitUntil(lambda: p._thread is None, timeout=5000)
        bubbles = [p._chat_layout.itemAt(i).widget()
                   for i in range(p._chat_layout.count() - 1)]
        bubbles = [b for b in bubbles if isinstance(b, _MessageBubble)]
        assert bubbles[-1]._text.text() == \
            "[error] RuntimeError: Could not run 'claude'"
        assert "Could not run" in p._status.text()
    finally:
        p.shutdown()


# ---------------------------------------------------------------------------
# clear_chat
# ---------------------------------------------------------------------------

def test_clear_chat_removes_history_and_every_bubble(panel):
    panel._append_user("one")
    panel._append_user("two")
    assert panel._chat_layout.count() == 3        # 2 bubbles + stretch
    panel.clear_chat()
    assert panel._messages == []
    assert panel._chat_layout.count() == 1        # only the stretch survives


def test_clear_chat_is_idempotent(panel):
    panel.clear_chat()
    panel.clear_chat()
    assert panel._chat_layout.count() == 1


def test_clear_chat_also_removes_non_widget_layout_items(panel):
    """The transcript layout can legitimately hold spacers/stretches; the
    clear loop must drain them too instead of spinning on one."""
    panel._append_user("one")
    panel._chat_layout.insertSpacing(0, 12)
    panel._chat_layout.insertStretch(0, 1)
    assert panel._chat_layout.count() == 4
    panel.clear_chat()
    assert panel._chat_layout.count() == 1


def test_failed_stream_after_clear_chat_still_reports(panel):
    """clear_chat drops the pending bubble; a later failure must fall back
    to the status line rather than dereferencing it."""
    panel._append_user("q")
    panel.clear_chat()
    panel._on_stream_finished(False, "RuntimeError: gone")
    assert panel._status.text() == "Failed: RuntimeError: gone"
    assert panel._messages == []


def test_clear_chat_mid_stream_does_not_crash_the_next_chunk(panel):
    """REGRESSION: clear_chat deleted the in-flight assistant bubble but
    kept the reference, so the next chunk raised
    'Internal C++ object already deleted' inside a Qt slot."""
    from PySide6.QtCore import QCoreApplication, QEvent
    from spacr.qt.widgets.ai_chat_panel import _MessageBubble

    panel._append_user("q")
    panel._pending_bubble = _MessageBubble("assistant", "…")
    panel._chat_layout.insertWidget(panel._chat_layout.count() - 1,
                                     panel._pending_bubble)
    panel._pending_buf = ["partial"]

    panel.clear_chat()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    panel._on_chunk("late chunk")                 # must not raise
    assert panel._pending_bubble is None
    assert panel._pending_buf == ["late chunk"]


# ---------------------------------------------------------------------------
# open_error_flow
# ---------------------------------------------------------------------------

def test_open_error_flow_sends_the_wrapped_traceback(panel, qtbot, ai_env):
    from spacr.qt.ai.prompts import error_explainer_prompt

    provider = ai_env["providers"][0]
    tb = "Traceback (most recent call last):\nValueError: channels is None"
    panel.open_error_flow(tb, active_app="mask")
    qtbot.waitUntil(lambda: panel._thread is None, timeout=5000)

    sent = provider.seen[0]
    assert sent["system"] == error_explainer_prompt()
    body = sent["messages"][0]["content"]
    assert "Active app: mask" in body
    assert "ValueError: channels is None" in body
    assert body.endswith("step-by-step fix (<=6 steps).")
    # the user turn is visible in the transcript
    assert panel._messages[0]["role"] == "user"
    assert panel._messages[-1] == {"role": "assistant",
                                   "content": "Hello world"}


def test_open_error_flow_without_a_provider_tells_the_user_what_to_do(panel,
                                                                      ai_env):
    ai_env["providers"] = []
    panel.open_error_flow("boom", active_app="measure")
    assert panel._status.text() == "Install a vendor CLI first (Providers…)."
    assert panel._messages == []
    assert panel._thread is None


def test_open_error_flow_omits_the_app_line_when_unknown(panel, qtbot,
                                                          ai_env):
    provider = ai_env["providers"][0]
    panel.open_error_flow("KeyError: 'plate'")
    qtbot.waitUntil(lambda: panel._thread is None, timeout=5000)
    body = provider.seen[0]["messages"][0]["content"]
    assert "Active app:" not in body
    assert body.startswith("Traceback:")


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------

def test_shutdown_cancels_an_active_stream_and_clears_refs(ai_env, qtbot):
    from spacr.qt.widgets.ai_chat_panel import AIChatPanel

    slow = _ScriptedProvider(chunks=[f"c{i}" for i in range(200)])
    ai_env["providers"] = [slow]
    p = AIChatPanel()
    qtbot.addWidget(p)
    p._input.setPlainText("stream a lot")
    p._send_from_input()
    assert p.is_streaming() is True

    p.shutdown()
    assert p._thread is None
    assert p._worker is None
    assert p._retired == []
    assert slow.cancelled >= 1


def test_shutdown_is_idempotent_when_idle(panel):
    panel.shutdown()
    panel.shutdown()
    assert panel._thread is None
    assert panel._worker is None


def test_shutdown_completes_even_if_the_registry_and_worker_misbehave(
        panel, monkeypatch):
    """Shutdown runs on the quit path — a raising collaborator must never
    leave the panel holding a live thread reference."""
    from spacr.qt import ai as ai_module

    def _boom_registry():
        raise RuntimeError("providers module unloaded")

    class _BadWorker:
        def cancel(self):
            raise RuntimeError("worker already gone")

    monkeypatch.setattr(ai_module, "list_providers", _boom_registry)
    panel._worker = _BadWorker()
    panel.shutdown()
    assert panel._thread is None
    assert panel._worker is None


def test_shutdown_terminates_a_thread_that_ignores_quit(panel):
    """Escalation policy: quit() -> wait() -> terminate() -> wait().

    Driven with a scripted QThread stand-in because a real QThread that
    genuinely ignores quit() would cost the full 3 s wait budget.
    """
    class _StubbornThread:
        def __init__(self):
            self.calls = []

        def isRunning(self):
            self.calls.append("isRunning")
            return True                      # never stops, whatever we do

        def quit(self):
            self.calls.append("quit")

        def wait(self, ms):
            self.calls.append(("wait", ms))

        def terminate(self):
            self.calls.append("terminate")

    class _AlreadyDone:
        def __init__(self):
            self.calls = []

        def isRunning(self):
            self.calls.append("isRunning")
            return False

    t = _StubbornThread()
    done = _AlreadyDone()
    panel._retired = [(done, object()), (t, object())]
    panel.shutdown()
    assert done.calls == ["isRunning"]      # nothing to do, moves on
    assert t.calls == ["isRunning", "quit", ("wait", 3000),
                       "isRunning", "terminate", ("wait", 1000)]
    assert panel._retired == []


def test_shutdown_ignores_a_retired_thread_qt_already_deleted(panel):
    class _Deleted:
        def isRunning(self):
            raise RuntimeError("Internal C++ object already deleted.")

    panel._retired = [(_Deleted(), object())]
    panel.shutdown()
    assert panel._retired == []


def test_close_event_drains_the_stream(ai_env, qtbot):
    from spacr.qt.widgets.ai_chat_panel import AIChatPanel

    ai_env["providers"] = [_ScriptedProvider(chunks=["x"] * 200)]
    p = AIChatPanel()
    qtbot.addWidget(p)
    p._input.setPlainText("go")
    p._send_from_input()
    p.close()
    assert p._thread is None
    assert p._worker is None


# ---------------------------------------------------------------------------
# Toolbar wiring
# ---------------------------------------------------------------------------

def test_clear_button_is_wired_to_clear_chat(panel):
    panel._append_user("hello")
    assert panel._messages
    panel._btn_clear.click()
    assert panel._messages == []


def test_providers_button_opens_the_dialog(panel, monkeypatch):
    from spacr.qt.widgets import ai_chat_panel as mod

    opened = []
    monkeypatch.setattr(mod._ProvidersDialog, "exec",
                        lambda self: opened.append(1) or QDialog.Rejected)
    panel._btn_keys.click()
    assert opened == [1]


def test_empty_state_cta_opens_the_dialog(ai_env, qtbot, monkeypatch):
    from spacr.qt.widgets import ai_chat_panel as mod

    ai_env["providers"] = []
    p = mod.AIChatPanel()
    qtbot.addWidget(p)
    opened = []
    monkeypatch.setattr(mod._ProvidersDialog, "exec",
                        lambda self: opened.append(1) or QDialog.Rejected)
    p._empty_state.cta_button.click()
    assert opened == [1]
