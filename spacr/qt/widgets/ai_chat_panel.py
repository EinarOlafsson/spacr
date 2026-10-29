"""
AIChatPanel — reusable chat widget hosting the spacr AI Console.

Meant to be embedded next to the pipeline Console (via a QDockWidget on
the main window) so users never leave the app they're running. Exposes
`open_error_flow(traceback, active_app)` for AppScreen's "Explain
error" button.

Design notes
------------
* One instance is shared across the whole main window; the user's
  chat context persists as they switch between Mask/Measure/etc.
* Streaming is done in a QThread via `spacr.qt.ai.worker.StreamWorker`.
  The worker reference is stored on `self`; if Python garbage-collected it,
  Qt might never deliver the finished signal.
* While a stream is in flight, the Send button turns into Cancel, and a
  message submitted from the input box is refused with a status-line
  notice rather than starting a second stream.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import ai as ai_module
from .. import iconset
from ..ai import settings as ai_settings
from ..ai.providers import ChatProvider
from ..ai.worker import StreamWorker, make_stream_thread
from ..i18n import retranslate_widget_tree, set_translatable_text, tr


def _current_system_prompt() -> str:
    """Delegate to ``ai_settings`` so a user override kicks in when set."""
    return ai_settings.get_system_prompt()
from ..theme import SPACING, active_palette, make_transparent
from .divider import Divider
from .empty_state import EmptyState



def _ai_reply_html(text):
    """Render assistant Markdown without activating model HTML or image resources."""
    import re
    from html import escape
    from html.parser import HTMLParser
    from urllib.parse import urlsplit

    from PySide6.QtGui import QTextDocument

    class PassiveDocument(QTextDocument):
        """Reject resource requests so rendered assistant replies remain passive."""
        def loadResource(self, kind, url):
            """Refuse image and other resource requests from the rendered document.

            :param kind: Qt resource type, deliberately not loaded.
            :param url: requested resource location, deliberately not opened.
            :returns: None so the document cannot fetch a resource.
            """
            return None

    class Markup(HTMLParser):
        """Retain passive document tags, escaped text and approved link schemes."""
        allowed = {'p', 'span', 'b', 'strong', 'i', 'em', 'u', 's', 'ul', 'ol',
                   'li', 'pre', 'code', 'blockquote', 'h1', 'h2', 'h3', 'h4',
                   'h5', 'h6', 'table', 'thead', 'tbody', 'tr', 'td', 'th',
                   'a', 'br', 'hr'}

        def __init__(self):
            """Initialize collected markup and the ignored document-head state."""
            super().__init__(convert_charrefs=True)
            self.parts = []
            self.head = False

        def handle_starttag(self, tag, attrs):
            """Keep permitted tags and safe links, substituting image alternative text.

            :param tag: parsed HTML element name.
            :param attrs: parsed attribute pairs to filter before serialization.
            """
            if tag == 'head':
                self.head = True
            if self.head:
                return
            values = dict(attrs)
            if tag == 'img':
                self.parts.append(escape(values.get('alt', '')))
            elif tag in self.allowed:
                kept = []
                for name, value in attrs:
                    if name == 'style' or (tag == 'ol' and name in ('start', 'type')):
                        kept.append((name, value))
                    elif tag == 'a' and name == 'href':
                        try:
                            safe = urlsplit(value).scheme.lower() in ('http', 'https', 'mailto')
                        except ValueError:
                            safe = False
                        if safe:
                            kept.append((name, value))
                attributes = ''.join(' ' + name + '="' + escape(value, quote=True) + '"'
                                     for name, value in kept)
                self.parts.append('<' + tag + attributes + '>')

        def handle_endtag(self, tag):
            """Close allowed non-void tags outside the ignored document head.

            :param tag: parsed closing element name.
            """
            if tag == 'head':
                self.head = False
            elif not self.head and tag in self.allowed and tag not in ('br', 'hr'):
                self.parts.append('</' + tag + '>')

        def handle_data(self, data):
            """Escape visible text while discarding document-head contents.

            :param data: parsed text to include when it belongs to visible content.
            """
            if not self.head:
                self.parts.append(escape(data))

    lines = []
    fence = None
    for line in str(text).splitlines(keepends=True):
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            token = marker.group(1)[0]
            fence = None if fence == token else token
            lines.append(line)
        elif fence:
            lines.append(line)
        else:
            line = re.sub(r'\\`([^`\n]*?)\\`', r'`\1`', line)
            pieces = re.split(r'(`+[^`\n]*`+)', line)
            for index in range(0, len(pieces), 2):
                part = re.sub(r'\\\*\\\*(.*?)\\\*\\\*', r'**\1**', pieces[index])
                part = re.sub(r'^(\s*\d+)\\\.(\s+)', r'\1.\2', part)
                pieces[index] = re.sub(r'^(\s*)\\([*+-])(\s+)', r'\1\2\3', part)
            lines.append(''.join(pieces))
    document = PassiveDocument()
    document.setMarkdown(''.join(lines), QTextDocument.MarkdownDialectGitHub | QTextDocument.MarkdownNoHTML)
    markup = Markup()
    markup.feed(document.toHtml())
    return '<div style="line-height:130%;">' + ''.join(markup.parts) + '</div>'


class _MessageBubble(QWidget):
    """Aligned QLabel bubble — right-aligned for ``"user"``, left for other.

    :param role: ``"user"`` or anything else, which is read as the AI. There
        is no third alignment, and an unrecognised role is drawn as the AI
        rather than refused.
    :param text: the initial message; may be set later instead.
    :param parent: parent widget; ownership only.
    """

    def __init__(self, role: str, text: str = "", parent=None):
        """Build the bubble and align it by role."""
        super().__init__(parent)
        self.role = role
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["sm"])
        self._text = QLabel()
        self._text.setProperty("i18nSkipText", True)
        self._text.setWordWrap(True)
        self._text.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse
        )
        self._text.setOpenExternalLinks(True)
        self._text.setObjectName(
            "ChatBubbleUser" if role == "user" else "ChatBubbleAssistant"
        )
        self._text.setMaximumWidth(720)
        self.set_text(text)
        if role == "user":
            layout.addStretch(1)
            layout.addWidget(self._text)
        else:
            layout.addWidget(self._text)
            layout.addStretch(1)

    def set_text(self, text: str) -> None:
        """Replace the bubble's body text."""
        import re

        rich = self.role != 'user' and bool(re.search(r'[\n*_`<>#]|!?\[[^]]+\]\([^)]+\)|^\s*(?:[-+]|\d+[.)])\s', text))
        self._text.setTextFormat(Qt.RichText if rich else Qt.PlainText)
        self._text.setText(_ai_reply_html(text) if rich else text)



class _ProvidersDialog(QDialog):
    """Tabbed dialog: install + login guidance for each vendor CLI on
    one tab, response-speed + system-prompt settings on another.

    :param parent: optional parent widget.
    """

    def __init__(self, parent=None):
        """Build the providers and settings tabs."""
        super().__init__(parent)
        self._provider_rows = {}
        self._provider_status_jobs = {}
        self._provider_signins = {}
        self.setWindowTitle("AI Console — providers & settings")
        from ..preferences import scaled_px
        
        self.setMinimumWidth(scaled_px(620))
        self.setMinimumHeight(scaled_px(560))
        outer = QVBoxLayout(self)

        tabs = QTabWidget()
        tabs.addTab(self._build_providers_tab(), "Providers")
        tabs.addTab(self._build_settings_tab(), "Settings")
        outer.addWidget(tabs, 1)

        buttons = QDialogButtonBox()
        refresh_btn = buttons.addButton("Refresh", QDialogButtonBox.AcceptRole)
        close_btn = buttons.addButton("Close", QDialogButtonBox.RejectRole)
        refresh_btn.clicked.connect(self.accept)
        close_btn.clicked.connect(self.reject)
        outer.addWidget(buttons)
        from ..screens.settings_model import install_api_tooltips
        install_api_tooltips(self, "ai", {
            self._speed_combo: "response_speed",
            self._auto_issue_chk: "auto_file_issues",
            self._route_errors_chk: "route_errors_through_ai",
            self._console_aware_chk: "console_aware",
            self._prompt_edit: "system_prompt",
        })
        retranslate_widget_tree(self)

    def _build_providers_tab(self) -> QWidget:
        """The page listing each provider and how to sign in to it."""
        page = QWidget()
        col = QVBoxLayout(page)

        intro = QLabel(
            "The AI Console talks to the <b>vendor coding-agent CLI</b> "
            "for each provider, using your chat subscription "
            "(Claude.ai Pro, ChatGPT Plus/Pro/Team, Google account).<br><br>"
            "For each provider you want to use, install the CLI then "
            "log in <em>once</em>. Copy the commands below into a "
            "terminal."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        col.addWidget(intro)

        for p in ai_module.list_providers():
            col.addWidget(self._make_provider_row(p))

        note = QLabel(
            "Once a CLI is installed <em>and</em> you're logged in, press "
            "<b>Refresh</b> below and it will appear in the provider "
            "dropdown."
        )
        note.setObjectName("SubtitleSmall")
        note.setWordWrap(True)
        note.setTextFormat(Qt.RichText)
        col.addWidget(note)
        col.addStretch(1)
        holder = QScrollArea()
        holder.setWidgetResizable(True)
        holder.setFrameShape(QScrollArea.NoFrame)
        holder.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        holder.setWidget(page)
        make_transparent(holder, page)
        return holder

    def _build_settings_tab(self) -> QWidget:
        """The page for response speed, the system prompt and issue filing."""
        page = QWidget()
        col = QVBoxLayout(page)
        col.setSpacing(SPACING["md"])

        speed_label = QLabel(
            "<b>Response speed</b><br>"
            "<span style='color:gray;'>Same three levels for every provider. "
            "Each level selects one of the provider's models: Fast the "
            "smallest and quickest, Deep the largest.</span>"
        )
        speed_label.setTextFormat(Qt.RichText)
        speed_label.setWordWrap(True)
        col.addWidget(speed_label)

        self._speed_combo = QComboBox()
        for value, human in (
            ("fast",     "Fast — snappy replies, smallest model"),
            ("balanced", "Balanced — default, mid-tier model"),
            ("deep",     "Deep — most thorough, largest model"),
        ):
            self._speed_combo.addItem(human, value)
        current = ai_settings.get_response_speed()
        for i in range(self._speed_combo.count()):
            if self._speed_combo.itemData(i) == current:
                self._speed_combo.setCurrentIndex(i)
                break
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        self._speed_combo._spacr_setting_label = speed_label
        col.addWidget(self._speed_combo)

        col.addWidget(Divider())

        auto_label = QLabel(
            "<b>Report errors as GitHub issues</b><br>"
            "<span style='color:gray;'>Reports go to the public spaCR "
            "repository on GitHub, with paths, names and credentials "
            "redacted. With issue reporting set to always, the default, a "
            "failed run files its report automatically, once per error, "
            "when GitHub is signed in below. With ask, a \"File as issue\" "
            "button appears under the console and the report is sent only "
            "when you press Send report in its preview. Change it under "
            "Help → Set spaCR up again….</span>"
        )
        auto_label.setTextFormat(Qt.RichText)
        auto_label.setWordWrap(True)
        col.addWidget(auto_label)

        from .toggle import Toggle
        self._auto_issue_chk = Toggle(
            "Enable — report a failed run as a GitHub issue"
        )
        self._auto_issue_chk.setChecked(ai_settings.get_auto_file_issues())
        self._auto_issue_chk.stateChanged.connect(self._on_auto_issue_changed)
        col.addWidget(self._auto_issue_chk)

        self._route_errors_chk = Toggle(
            "Route errors through AI — show the AI's explanation instead of "
            "the raw traceback"
        )
        self._route_errors_chk.setChecked(
            ai_settings.get_route_errors_through_ai())
        self._route_errors_chk.stateChanged.connect(
            lambda _s: ai_settings.set_route_errors_through_ai(
                self._route_errors_chk.isChecked()))
        col.addWidget(self._route_errors_chk)

        self._console_aware_chk = Toggle(
            "Console aware — send the console with your question so the AI "
            "can explain what went wrong"
        )
        self._console_aware_chk.setChecked(ai_settings.get_console_aware())
        self._console_aware_chk.stateChanged.connect(
            lambda _s: ai_settings.set_console_aware(
                self._console_aware_chk.isChecked()))
        col.addWidget(self._console_aware_chk)

        col.addWidget(Divider())
        from ..ai import github_auth
        gh_label = QLabel(
            "<b>GitHub sign-in</b><br>"
            "<span style='color:gray;'>Install the official GitHub CLI and "
            "run <code>gh auth login</code>. GitHub stores the credential in "
            "the platform credential manager; spaCR never captures or stores "
            "the token. Without CLI login, reports open in your authenticated "
            "browser instead."
            "</span>")
        gh_label.setTextFormat(Qt.RichText)
        gh_label.setWordWrap(True)
        col.addWidget(gh_label)

        self._gh_status = QLabel()
        self._gh_status.setProperty("i18nSkipText", True)
        col.addWidget(self._gh_status)

        gh_command = QLineEdit("gh auth login")
        gh_command.setReadOnly(True)
        col.addWidget(gh_command)
        self._refresh_github_status()

        col.addWidget(Divider())

        prompt_label = QLabel(
            "<b>System prompt</b><br>"
            "<span style='color:gray;'>The persona spaCR sends to the "
            "assistant before your first message. Edit to change how "
            "answers are framed, then Save. Reset restores the default.</span>"
        )
        prompt_label.setTextFormat(Qt.RichText)
        prompt_label.setWordWrap(True)
        col.addWidget(prompt_label)

        self._prompt_edit = QTextEdit()
        self._prompt_edit._spacr_setting_label = prompt_label
        self._prompt_edit.setPlainText(ai_settings.get_system_prompt())
        self._prompt_edit.setMinimumHeight(240)
        col.addWidget(self._prompt_edit, 1)

        self._prompt_status = QLabel(self._prompt_status_text())
        self._prompt_status.setObjectName("SubtitleSmall")
        self._prompt_status.setProperty("i18nSkipText", True)
        col.addWidget(self._prompt_status)

        prompt_buttons = QHBoxLayout()
        save_btn = QPushButton("Save prompt")
        save_btn.clicked.connect(self._on_prompt_save)
        prompt_buttons.addWidget(save_btn)
        reset_btn = QPushButton("Reset to default")
        reset_btn.clicked.connect(self._on_prompt_reset)
        prompt_buttons.addWidget(reset_btn)
        prompt_buttons.addStretch(1)
        wrap = QWidget(); wrap.setLayout(prompt_buttons)
        col.addWidget(wrap)

        return page

    def _on_speed_changed(self, _idx: int) -> None:
        """Store the chosen speed, ignoring a value the store would reject.

        Guarded rather than trusted: the combo is built from the valid set, so a
        value outside it means the two have drifted, and writing it would put a
        setting in the store that nothing can read back.
        """
        value = self._speed_combo.currentData()
        if value in ai_settings.VALID_SPEEDS:
            ai_settings.set_response_speed(value)

    def _on_auto_issue_changed(self, _state: int) -> None:
        """Store whether a failed run is reported as a GitHub issue."""
        ai_settings.set_auto_file_issues(self._auto_issue_chk.isChecked())

    def _refresh_github_status(self) -> None:
        """Say whether issues can be sent directly, and by what.

        NAMES THE SOURCE -- a token, the environment, or the CLI -- because "signed
        in" alone does not tell a user which credential is about to be used, and
        that is the thing they change when it is the wrong one.
        """
        from ..ai import github_auth
        src = github_auth.auth_source()
        labels = {
            "token": "a process-only API token",
            "env": "the GITHUB_TOKEN env var",
            "gh": "the GitHub CLI",
        }
        if src:
            source = "✓ Signed in via {source} — issues send directly."
            self._gh_status.setText(
                "<span style='color:#3fb950;'>"
                + tr(source, source=tr(labels.get(src, src))) + "</span>")
        else:
            source = (
                "Not signed in — issues open in your browser. Run gh auth "
                "login for direct submission after preview."
            )
            self._gh_status.setText(
                "<span style='color:#d29922;'>"
                + tr(source).replace(
                    "gh auth login", "<code>gh auth login</code>"
                ) + "</span>")
        self._gh_status.setTextFormat(Qt.RichText)

    def _on_prompt_save(self) -> None:
        """Store the edited system prompt and say it is now custom."""
        text = self._prompt_edit.toPlainText().strip()
        ai_settings.set_system_prompt(text)
        self._prompt_status.setText(self._prompt_status_text())

    def _on_prompt_reset(self) -> None:
        """Restore the default prompt and show it in the editor.

        The editor is REFILLED rather than cleared, so the reader can see what
        the default actually says before deciding to keep it.
        """
        ai_settings.reset_system_prompt()
        self._prompt_edit.setPlainText(ai_settings.get_system_prompt())
        self._prompt_status.setText(self._prompt_status_text())

    def _prompt_status_text(self) -> str:
        """Whether the custom system prompt or the default is in force."""
        if ai_settings.is_system_prompt_overridden():
            return tr("Using your custom prompt (overrides default).")
        return tr("Using the default spaCR-aware prompt.")

    def _make_provider_row(self, provider: ChatProvider) -> QWidget:
        """One provider's card: its name, its state and how to sign in."""
        card = QWidget()
        col = QVBoxLayout(card)
        col.setContentsMargins(SPACING["sm"], SPACING["sm"],
                                SPACING["sm"], SPACING["sm"])
        col.setSpacing(2)

        header = QHBoxLayout()
        title = QLabel(f"<b>{provider.label}</b>")
        title.setProperty("i18nSkipText", True)
        title.setTextFormat(Qt.RichText)
        header.addWidget(title)
        header.addStretch(1)
        installed = bool(self._provider_login_argv(provider)) or provider.is_installed()
        status_label = QLabel(
            f"<span style='color:#3fb950;'>● {tr('installed')}</span>"
            if installed
            else f"<span style='color:#f85149;'>● {tr('missing')}</span>"
        )
        status_label.setProperty("i18nSkipText", True)
        status_label.setTextFormat(Qt.RichText)
        header.addWidget(status_label)
        header_wrap = QWidget(); header_wrap.setLayout(header)
        col.addWidget(header_wrap)

        official = {
            'claude': 'https://code.claude.com/docs/en/setup',
            'codex': 'https://developers.openai.com/codex/cli',
            'gemini': 'https://geminicli.com/docs/get-started/installation/',
        }.get(provider.name)
        if official:
            from html import escape

            link = QLabel(f'<a href="{official}">{escape(provider.label)}</a>')
            link.setObjectName('ProviderInstallLink_' + provider.name)
            link.setProperty('i18nSkipText', True)
            link.setOpenExternalLinks(True)
            col.addWidget(link)
        if provider.name == 'gemini':
            requirement = QLabel(tr(
                'Gemini requires Node.js. On macOS and Linux, the command below installs '
                'Node.js and Gemini CLI in your user account. On Windows, install Node.js '
                'before running the npm command.'))
            requirement.setWordWrap(True)
            col.addWidget(requirement)

        install_row = QHBoxLayout()
        install_row.addWidget(QLabel("Install:"))
        install_edit = QLineEdit(provider.install_hint)
        install_edit.setReadOnly(True)
        install_row.addWidget(install_edit, 1)
        copy_i = QPushButton("Copy")
        copy_i.clicked.connect(lambda _, e=install_edit: self._copy_to_clipboard(e.text()))
        install_row.addWidget(copy_i)
        install_wrap = QWidget(); install_wrap.setLayout(install_row)
        col.addWidget(install_wrap)

        login_row = QHBoxLayout()
        login_row.addWidget(QLabel("Login:"))
        login_edit = QLineEdit(provider.login_command)
        login_edit.setReadOnly(True)
        login_row.addWidget(login_edit, 1)
        sign_in = QPushButton(tr('Sign in'))
        sign_in.setObjectName('ProviderSignIn_' + provider.name)
        sign_in.setEnabled(bool(self._provider_login_argv(provider)))
        sign_in.clicked.connect(lambda _checked=False, p=provider: self._sign_in_provider(p))
        login_row.addWidget(sign_in)
        copy_l = QPushButton("Copy")
        copy_l.clicked.connect(lambda _, e=login_edit: self._copy_to_clipboard(e.text()))
        login_row.addWidget(copy_l)
        login_wrap = QWidget(); login_wrap.setLayout(login_row)
        col.addWidget(login_wrap)
        auth_status = QLabel('')
        auth_status.setObjectName('ProviderAuthStatus_' + provider.name)
        auth_status.setWordWrap(True)
        if not sign_in.isEnabled():
            sign_in.setToolTip(tr('Copy the login command and run it in a terminal on this platform.'))
        col.addWidget(auth_status)
        self._provider_rows[provider.name] = (provider, card, status_label, auth_status, sign_in)

        palette = active_palette()
        card.setStyleSheet(
            f"QWidget {{background: {palette['surface_alt']}; "
            f"border: 1px solid {palette['border_soft']}; "
            f"border-radius: 6px;}}"
        )
        return card

    @staticmethod
    def _provider_login_argv(provider):
        """Use only known provider login arguments and a discovered executable."""
        from ..ai.cli_install import locate, put_on_path

        arguments = {'claude': ['auth', 'login'], 'codex': ['login'], 'gemini': []}
        if provider.name not in arguments:
            return []
        location = locate(provider)
        if not location:
            return []
        put_on_path(location)
        return [location, *arguments[provider.name]]

    def _sign_in_provider(self, provider):
        """Launch an explicit native sign-in and check the tool's resulting state."""
        import os

        from ..ai.pty_sign_in import SignInDialog, pty_available

        argv = self._provider_login_argv(provider)
        if not argv:
            self._refresh_provider_status(provider)
            return
        try:
            if os.name == 'nt':
                self._start_windows_sign_in(provider, argv)
                return
            if not pty_available():
                self._provider_rows[provider.name][3].setText(tr(
                    'Copy the login command and run it in a terminal on this platform.'))
                return
            dialog = SignInDialog(provider.label, argv, self)
            try:
                dialog.exec()
            finally:
                dialog.deleteLater()
        except (OSError, ValueError) as error:
            self._provider_rows[provider.name][3].setText(tr('Failed: {detail}', detail=str(error)))
        self._refresh_provider_status(provider)

    def _start_windows_sign_in(self, provider, argv):
        """Open fixed login arguments in a separate native Windows console."""
        import os
        import subprocess

        from PySide6.QtCore import QTimer

        if provider.name in self._provider_signins:
            return
        command = argv
        if argv[0].lower().endswith(('.cmd', '.bat')):
            if any(character in argv[0] for character in '\r\n"%!'):
                raise ValueError(tr('Copy the login command and run it in a terminal on this platform.'))
            shell = os.environ.get('COMSPEC', os.path.join(os.environ.get('SystemRoot', r'C:\Windows'),
                                                          'System32', 'cmd.exe'))
            command = subprocess.list2cmdline([shell]) + ' /d /s /c ""' + argv[0] + '"'
            if len(argv) > 1:
                command += ' ' + ' '.join(argv[1:])
            command += '"'
        process = subprocess.Popen(command, shell=False,
                                   creationflags=getattr(subprocess, 'CREATE_NEW_CONSOLE', 0x10))
        timer = QTimer(self)
        timer.setInterval(250)
        timer.timeout.connect(lambda p=provider: self._poll_provider_sign_in(p))
        self._provider_signins[provider.name] = (process, timer)
        self._provider_rows[provider.name][4].setEnabled(False)
        timer.start()

    def _poll_provider_sign_in(self, provider):
        """Recheck authentication after the separate console exits."""
        pair = self._provider_signins.get(provider.name)
        if pair is None or pair[0].poll() is None:
            return
        pair[1].stop()
        pair[1].deleteLater()
        del self._provider_signins[provider.name]
        self._refresh_provider_status(provider)

    def _refresh_provider_status(self, provider):
        """Check installed, authenticated and configured state off the GUI thread."""
        from functools import partial

        from ..ai import providers
        from ..job_runner import JobRunner

        self._provider_login_argv(provider)
        providers._PATH_LOOKUPS.clear()
        row = self._provider_rows[provider.name]
        row[3].setText(tr('Checking sign-in…'))
        row[4].setEnabled(False)
        runner = self._provider_status_jobs.get(provider.name)
        if runner is None:
            runner = JobRunner(self, app_key='ai', user_visible=False)
            self._provider_status_jobs[provider.name] = runner
        runner.cancel()
        runner.submit(partial(self._read_provider_status, provider),
                      on_done=partial(self._show_provider_status, provider.name))

    @staticmethod
    def _read_provider_status(provider):
        """Never treat the interactive process's exit status as proof of authentication."""
        installed = provider.is_installed()
        signed_in = provider.check_signed_in() if installed else False
        return installed, signed_in, provider.is_configured() if installed else False

    def _show_provider_status(self, name, state):
        """Expose actual status without promoting an unknown sign-in to success."""
        provider, card, installed_label, auth_label, button = self._provider_rows[name]
        installed, signed_in, configured = state
        installed_label.setText(tr('installed') if installed else tr('missing'))
        auth_label.setText(tr('Signed in.') if signed_in is True else tr('Not signed in.')
                           if signed_in is False else tr(
                               'Sign-in status is unavailable; use the CLI to confirm your account.'))
        card.setProperty('providerConfigured', configured)
        card.setProperty('providerSignedIn', signed_in)
        button.setEnabled(bool(self._provider_login_argv(provider)))

    def done(self, result):
        """Cancel status deliveries and owned console processes when this dialog closes."""
        import subprocess

        for runner in self._provider_status_jobs.values():
            runner.shutdown(timeout_ms=0)
        for process, timer in self._provider_signins.values():
            timer.stop()
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                        process.wait(timeout=0.1)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                except OSError:
                    pass
        self._provider_signins.clear()
        super().done(result)

    def _copy_to_clipboard(self, text: str) -> None:
        """Put ``text`` on the clipboard, if there is one.

        A headless or clipboard-less session gets nothing rather than an
        exception: copying a sign-in command is a convenience, and failing it
        must not take the dialog down.
        """
        from PySide6.QtGui import QGuiApplication
        cb = QGuiApplication.clipboard()
        if cb is not None:
            cb.setText(text)



class _ChatInput(QTextEdit):
    """Multi-line input: Enter sends, Shift+Enter inserts a newline.

    :param parent: parent widget; ownership only.
    """

    submitted = Signal()

    def __init__(self, parent=None):
        """Build the input, bounded so it grows a little and no further."""
        super().__init__(parent)
        self.setMinimumHeight(56)
        self.setMaximumHeight(140)
        self.setAcceptRichText(False)

    def keyPressEvent(self, event: QKeyEvent):
        """Emit ``submitted`` on plain Enter; forward Shift+Enter as newline."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
                return
            self.submitted.emit()
            return
        super().keyPressEvent(event)



class AIChatPanel(QWidget):
    """Full chat panel — embed inside a QDockWidget or any container.

    :param parent: parent widget.
    """

    _error_batch_ready = Signal(str, str, bool, str, object)

    def __init__(self, parent: Optional[QWidget] = None):
        """Build the chat panel.

        :param parent: parent widget.
        """
        super().__init__(parent)
        self._error_batch_ready.connect(self._open_batched_error)
        self._messages: List[Dict] = []
        self._thread: Optional[QThread] = None
        self._worker: Optional[StreamWorker] = None
        self._pending_bubble: Optional[_MessageBubble] = None
        self._pending_buf: List[str] = []
        self._retired: List = []

        self._build_ui()
        self.refresh_provider_combo()

    def _build_ui(self):
        """Lay out the transcript over the input row."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["md"], SPACING["md"],
                                  SPACING["md"], SPACING["md"])
        outer.setSpacing(SPACING["sm"])

        toolbar = QHBoxLayout()
        toolbar.setSpacing(SPACING["sm"])
        toolbar.addWidget(QLabel("Provider"))
        self._provider_combo = QComboBox()
        self._provider_combo.setProperty("i18nSkipItems", True)
        self._provider_combo.setMinimumWidth(160)
        toolbar.addWidget(self._provider_combo)
        self._btn_keys = QPushButton("Providers")
        self._btn_keys.setIcon(iconset.icon("settings"))
        self._btn_keys.setCursor(Qt.PointingHandCursor)
        self._btn_keys.setToolTip("Install + login instructions for the "
                                    "vendor coding-agent CLIs.")
        self._btn_keys.clicked.connect(self._on_open_keys_dialog)
        toolbar.addWidget(self._btn_keys)
        toolbar.addStretch(1)
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setObjectName("GhostButton")
        self._btn_clear.setIcon(iconset.icon("clear"))
        from ..preferences import _SMALL_ICON_PX, _set_scaled_icon_size
        _set_scaled_icon_size(self._btn_clear, _SMALL_ICON_PX)
        self._btn_clear.setCursor(Qt.PointingHandCursor)
        self._btn_clear.clicked.connect(self.clear_chat)
        toolbar.addWidget(self._btn_clear)
        tb_wrap = QWidget(); tb_wrap.setLayout(toolbar)
        outer.addWidget(tb_wrap)

        outer.addWidget(Divider())

        self._empty_state = EmptyState(
            title="Install a vendor CLI to chat",
            subtitle=(
                "The AI Console uses your chat subscription via the "
                "vendor coding-agent CLIs: `claude`, `codex`, or "
                "`gemini`. Install any one of them and log in. Open "
                "Providers ▸ Copy the commands and paste in a terminal."
            ),
            icon=iconset.accent_icon("info"),
            cta_label="Providers…",
            on_action=self._on_open_keys_dialog,
        )
        self._chat_scroll = QScrollArea()
        self._chat_scroll.setWidgetResizable(True)
        self._chat_scroll.setFrameShape(QScrollArea.NoFrame)
        self._chat_holder = QWidget()
        self._chat_layout = QVBoxLayout(self._chat_holder)
        self._chat_layout.setContentsMargins(0, 0, 0, 0)
        self._chat_layout.setSpacing(SPACING["sm"])
        self._chat_layout.addStretch(1)
        self._chat_scroll.setWidget(self._chat_holder)
        make_transparent(self._chat_scroll, self._chat_holder)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._empty_state)
        self._stack.addWidget(self._chat_scroll)
        outer.addWidget(self._stack, 1)

        input_row = QHBoxLayout()
        input_row.setSpacing(SPACING["sm"])
        self._input = _ChatInput()
        self._input.setPlaceholderText(
            "Ask a question (Enter to send · Shift+Enter for newline)"
        )
        self._input.submitted.connect(self._send_from_input)
        input_row.addWidget(self._input, 1)
        self._btn_send = QPushButton("Send")
        self._btn_send.setObjectName("PrimaryButton")
        self._btn_send.setIcon(iconset.contrast_icon("run"))
        self._btn_send.setCursor(Qt.PointingHandCursor)
        self._btn_send.clicked.connect(self._send_from_input)
        input_row.addWidget(self._btn_send)
        input_wrap = QWidget(); input_wrap.setLayout(input_row)
        outer.addWidget(input_wrap)

        self._status = QLabel("")
        self._status.setObjectName("SubtitleSmall")
        outer.addWidget(self._status)
        retranslate_widget_tree(self)

    def refresh_provider_combo(self) -> None:
        """Rebuild the provider dropdown from the currently configured CLIs."""
        self._provider_combo.blockSignals(True)
        self._provider_combo.clear()
        configured = ai_module.configured_providers()
        for p in configured:
            self._provider_combo.addItem(p.label, userData=p.name)
        self._provider_combo.blockSignals(False)
        if configured:
            self._stack.setCurrentWidget(self._chat_scroll)
            self._input.setEnabled(True)
            self._set_send_mode("send")
            self._btn_send.setEnabled(True)
        else:
            self._stack.setCurrentWidget(self._empty_state)
            self._input.setEnabled(False)
            self._set_send_mode("send")
            self._btn_send.setEnabled(False)

    def _current_provider(self) -> Optional[ChatProvider]:
        """The provider this panel will send through.

        :returns: the provider, or None when none is configured.
        """
        name = self._provider_combo.currentData()
        return ai_module.get_provider(name) if name else None

    def _on_open_keys_dialog(self):
        """Open the dialog that installs or signs in to a provider."""
        dlg = _ProvidersDialog(self)
        if dlg.exec() == QDialog.Accepted:
            self.refresh_provider_combo()

    def _set_send_mode(self, mode: str):
        """Switch between sending a question and cancelling a stream.

        ONE BUTTON, TWO MEANINGS, because a stream can take a while and a
        separate cancel button sits dead for most of the panel's life.

        :param mode: which meaning the button now carries.
        """
        set_translatable_text(
            self._btn_send,
            "Cancel" if mode == "cancel" else "Send",
        )
        if mode == "cancel":
            self._btn_send.setObjectName("DangerButton")
            try:
                self._btn_send.clicked.disconnect()
            except Exception:
                pass
            self._btn_send.clicked.connect(self._cancel_stream)
        else:
            self._btn_send.setObjectName("PrimaryButton")
            try:
                self._btn_send.clicked.disconnect()
            except Exception:
                pass
            self._btn_send.clicked.connect(self._send_from_input)
        self._btn_send.style().unpolish(self._btn_send)
        self._btn_send.style().polish(self._btn_send)

    def _send_from_input(self):
        """Send whatever is typed, and clear the box only if it was accepted."""
        text = self._input.toPlainText().strip()
        if not text:
            return
        if self._thread is not None:
            set_translatable_text(
                self._status,
                "A response is already streaming — press Cancel to interrupt.",
            )
            return
        self._input.clear()
        self._append_user(text)
        self._start_stream(system=_current_system_prompt())

    def _cancel_stream(self):
        """Stop the answer that is streaming in."""
        if self._worker is not None:
            self._worker.cancel()
            set_translatable_text(self._status, "Cancelling…")

    def _append_user(self, text: str):
        """Put the user's own message in the transcript.

        :param text: what they asked.
        """
        self._messages.append({"role": "user", "content": text})
        bubble = _MessageBubble("user", text)
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, bubble)
        self._scroll_to_bottom()

    def _start_stream(self, system: str):
        """Ask the provider, streaming the answer in as it arrives.

        :param system: the system prompt to send with it.
        """
        provider = self._current_provider()
        if provider is None:
            set_translatable_text(self._status, "No provider configured.")
            return
        self._pending_buf = []
        self._pending_bubble = _MessageBubble("assistant", "…")
        self._chat_layout.insertWidget(self._chat_layout.count() - 1,
                                        self._pending_bubble)
        self._scroll_to_bottom()

        thread, worker = make_stream_thread(
            provider, list(self._messages), system=system, parent=self,
        )
        worker.stage_changed.connect(self._on_stage_changed)
        worker.chunk_ready.connect(self._on_chunk)
        worker.finished.connect(self._on_stream_finished)
        self._thread = thread
        self._worker = worker
        self._set_send_mode("cancel")
        set_translatable_text(
            self._status, "Connecting to {provider}…",
            provider=provider.label,
        )
        thread.start()

    def _on_stage_changed(self, stage: str):
        """Show which stage the answer is at.

        :param stage: the stage's name.
        """
        provider = self._current_provider()
        label = provider.label if provider else ""
        if stage == "connecting":
            set_translatable_text(
                self._status, "Connecting to {provider}…", provider=label)
        elif stage == "streaming":
            set_translatable_text(
                self._status, "Streaming from {provider}…", provider=label)

    def _on_chunk(self, text: str):
        """Append one chunk of a streaming answer.

        :param text: the chunk.
        """
        self._pending_buf.append(text)
        if self._pending_bubble is not None:
            self._pending_bubble.set_text("".join(self._pending_buf))
            self._scroll_to_bottom()

    def _on_stream_finished(self, ok: bool, final_text: str):
        """Close off the answer, successfully or not.

        :param ok: True when the stream completed.
        :param final_text: the whole answer, for the transcript.
        """
        self._prune_retired()
        thread, worker = self._thread, self._worker
        self._thread = None
        self._worker = None
        if thread is not None:
            self._retired.append((thread, worker))
        self._set_send_mode("send")
        if ok:
            self._messages.append({"role": "assistant", "content": final_text})
            if self._pending_bubble is not None and not self._pending_buf:
                self._pending_bubble.set_text(tr(
                    "(empty response — try again or switch provider)"))
            set_translatable_text(self._status, "Ready.")
        else:
            if self._pending_bubble is not None:
                self._pending_bubble.set_text(f"[error] {final_text}")
            set_translatable_text(
                self._status, "Failed: {detail}", detail=final_text)
        self._pending_bubble = None
        self._pending_buf = []

    def _prune_retired(self) -> None:
        """Forget retired (thread, worker) pairs whose QThread has exited.

        A pair whose C++ object Qt already deleted raises RuntimeError on
        ``isRunning()`` — that is also safe to drop.
        """
        alive = []
        for thread, worker in self._retired:
            try:
                if thread.isRunning():
                    alive.append((thread, worker))
            except RuntimeError:
                pass
        self._retired = alive

    def is_streaming(self) -> bool:
        """Return True while a response is being streamed."""
        return self._thread is not None

    def shutdown(self) -> None:
        """Request cancellation and drain current and retired streaming threads.

        Call before the panel is destroyed so a running QThread keeps an owner. The active stream's thread is passed to the drain helper with a 3000 ms wait and each retired thread with a 1000 ms wait. Threads still running after those waits are parked by the helper rather than terminated; this method does not guarantee that every thread has exited when it returns.

        Never terminates a thread: see
        :func:`spacr.qt.bridge.drain_thread` and the note in
        :meth:`spacr.qt.widgets.console_panel.ConsolePanel.shutdown` for
        why ``QThread.terminate()`` on a Python thread trades one crash
        for a worse, unattributable one.
        """
        from ..bridge import drain_thread

        worker, thread = self._worker, self._thread
        try:
            for p in ai_module.list_providers():
                p.cancel_stream()
        except Exception:
            pass
        if worker is not None:
            try:
                worker.cancel()
            except Exception:
                pass
        drain_thread(thread, worker, timeout_ms=3000)
        for pair in list(self._retired):
            drain_thread(pair[0], pair[1], timeout_ms=1000)
        self._thread = None
        self._worker = None
        self._retired.clear()

    def closeEvent(self, event) -> None:
        """Drain the streaming thread before Qt destroys the panel.

        :param event: the close event; it is not inspected, only passed on to
            the base class after :meth:`shutdown`.
        """
        self.shutdown()
        super().closeEvent(event)

    def _scroll_to_bottom(self):
        """Follow the answer as it streams in.

        ONLY WHEN ALREADY AT THE BOTTOM would be the other choice, and this
        panel does it unconditionally: the stream is the thing the user just
        asked for, so following it is what they want.
        """
        sb = self._chat_scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_chat(self) -> None:
        """Discard chat history and remove every bubble from the scroll area."""
        self._messages.clear()
        self._pending_bubble = None
        self._pending_buf = []
        while self._chat_layout.count() > 1:
            item = self._chat_layout.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def open_error_flow(self, traceback_text: str, active_app: str = "") -> None:
        """Send a traceback to the AI explainer and stream the reply.

        :param traceback_text: raw traceback captured from the pipeline.
        :param active_app: optional app label used in the framing prompt.
        """
        from ..ai.providers import _batch_widget_error
        if not _batch_widget_error(self, traceback_text, active_app):
            self._open_error_flow_now(traceback_text, active_app)

    def _open_batched_error(self, text, active_app, show_raw, provider_name, diagnostics):
        """Recheck the selected provider and network before a delayed request."""
        from ... import crashreport
        provider = self._current_provider()
        policy = crashreport._ERROR_POLICY_CACHE.get("affirmed_policy") or {}
        if (provider is not None and provider.name == provider_name
                and policy.get("provider") != "none" and crashreport.network_allowed()):
            self._pending_error_diagnostics = diagnostics
            self._open_error_flow_now(text, active_app)

    def _open_error_flow_now(self, traceback_text: str, active_app: str = "") -> None:
        """Send a traceback to the AI explainer and stream the reply.

        :param traceback_text: raw traceback captured from the pipeline.
        :param active_app: optional app label used in the framing prompt.
        """
        from ..ai.prompts import wrap_error_for_prompt, error_explainer_prompt
        provider = self._current_provider()
        if provider is None:
            set_translatable_text(
                self._status, "Install a vendor CLI first (Providers…).")
            return
        prompt = wrap_error_for_prompt(traceback_text, active_app)
        self._append_user(prompt)
        self._messages[-1]["_spacr_batch_errors"] = getattr(self, "_pending_error_diagnostics", [])
        self._pending_error_diagnostics = []
        self._start_stream(system=error_explainer_prompt())
