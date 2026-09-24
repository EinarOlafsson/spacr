"""Item 420: the AI sign-ins run inside spaCR, on a pseudo-terminal.

2026-09-16, the maintainer, and the item's first DONE WHEN: "installed,
logged in and marked READY, with no terminal used". The vendors' tools sign
in by conversation and refuse to hold it without a terminal; a stand-in tool
below behaves the same way -- it checks it IS on a terminal, prints a coloured
sign-in link, asks for a code, and says whether it worked.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("PySide6")

from spacr.qt.ai import pty_sign_in as psi  # noqa: E402

pytestmark = pytest.mark.skipif(not psi.pty_available(),
                                reason="needs a POSIX pseudo-terminal")


@pytest.fixture(autouse=True)
def local_processes_only(monkeypatch, _no_real_embedded_provider_sign_in):
    """This file runs only Python stand-ins written under tmp_path."""
    monkeypatch.setattr(psi, "_spawn", subprocess.Popen)


@pytest.fixture
def fake_cli(tmp_path):
    script = tmp_path / "fake_login.py"
    script.write_text(textwrap.dedent('''
        import sys
        if not sys.stdin.isatty():
            print("this tool needs a terminal"); sys.exit(3)
        print("\\x1b[1;32mOpen this page to sign in:\\x1b[0m "
              "https://example.org/oauth?state=abc123 ")
        sys.stdout.write("Paste code here: "); sys.stdout.flush()
        code = sys.stdin.readline().strip()
        print("Logged in." if code == "XYZ-42" else "Wrong code.")
        sys.exit(0 if code == "XYZ-42" else 1)
    '''))
    return [sys.executable, str(script)]


def test_terminal_codes_are_removed_and_links_found():
    text = psi.strip_terminal_codes("\x1b[1mGo to\x1b[0m https://a.b/c?x=1).\r\n")
    assert text == "Go to https://a.b/c?x=1).\n"
    assert psi.find_urls(text) == ["https://a.b/c?x=1"]


def test_the_tool_believes_it_is_on_a_terminal_and_signs_in(qtbot, fake_cli):
    opened = []
    dialog = psi.SignInDialog("Claude", fake_cli, open_url=opened.append)
    qtbot.addWidget(dialog)
    qtbot.waitUntil(lambda: "Paste code" in dialog.log.toPlainText(), timeout=10000)
    assert opened == ["https://example.org/oauth?state=abc123"]
    assert dialog.open_btn.isEnabled()
    dialog.answer.setText("XYZ-42")
    dialog.send_btn.click()
    qtbot.waitUntil(lambda: dialog.succeeded is not None, timeout=10000)
    assert dialog.succeeded is True
    assert "Logged in." in dialog.log.toPlainText()
    assert not dialog.session._reader.is_alive()
    with pytest.raises(OSError):
        os.fstat(dialog.session._master)


def test_a_failed_sign_in_says_so(qtbot, fake_cli):
    dialog = psi.SignInDialog("GPT", fake_cli, open_url=lambda url: None)
    qtbot.addWidget(dialog)
    qtbot.waitUntil(lambda: "Paste code" in dialog.log.toPlainText(), timeout=10000)
    dialog.answer.setText("nope")
    dialog.send_btn.click()
    qtbot.waitUntil(lambda: dialog.succeeded is not None, timeout=10000)
    assert dialog.succeeded is False
    assert "Wrong code." in dialog.log.toPlainText()


def test_closing_the_window_stops_the_sign_in(qtbot, fake_cli):
    dialog = psi.SignInDialog("Gemini", fake_cli, open_url=lambda url: None)
    qtbot.addWidget(dialog)
    qtbot.waitUntil(lambda: "Paste code" in dialog.log.toPlainText(), timeout=10000)
    dialog.reject()
    assert dialog.session.poll() is not None


def test_completion_drains_the_last_output_before_showing_the_result(qtbot):
    class FinishedSession:
        text = ""

        def poll(self):
            return 1

        def stop(self):
            self.text = "The local stand-in refused the supplied code."

        def read(self):
            return self.text

    dialog = psi.SignInDialog("Local stand-in", ["unused"],
                             session_factory=lambda argv: FinishedSession(),
                             open_url=lambda url: None)
    qtbot.addWidget(dialog)
    dialog._tick()
    assert dialog.succeeded is False
    assert "refused the supplied code" in dialog.log.toPlainText()


def test_destroying_the_parent_stops_the_sign_in(qtbot, fake_cli):
    from PySide6.QtWidgets import QWidget
    from shiboken6 import delete

    parent = QWidget()
    dialog = psi.SignInDialog("Local stand-in", fake_cli, parent,
                             open_url=lambda url: None)
    session = dialog.session
    try:
        qtbot.waitUntil(lambda: "Paste code" in dialog.log.toPlainText(),
                        timeout=10000)
        delete(parent)
        assert session.poll() is not None
        qtbot.waitUntil(lambda: not session._reader.is_alive(), timeout=2000)
    finally:
        session.stop()


def test_a_failed_spawn_closes_both_terminal_descriptors(monkeypatch):
    opened = []
    original = os.openpty

    def record():
        pair = original()
        opened.extend(pair)
        return pair

    monkeypatch.setattr(psi.os, "openpty", record)
    with pytest.raises(FileNotFoundError):
        psi.PtySession(["/does/not/exist/spacr-test-login"])
    try:
        for descriptor in opened:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        for descriptor in opened:
            try:
                os.close(descriptor)
            except OSError:
                pass


@pytest.mark.parametrize("stubborn_launcher", [False, True])
def test_stopping_a_launcher_also_stops_its_stubborn_child(
        qtbot, tmp_path, stubborn_launcher):
    script = tmp_path / "launcher.py"
    script.write_text(textwrap.dedent('''
        import subprocess
        import signal
        import sys
        import time
        if sys.argv[1] == "True":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        subprocess.Popen([sys.executable, "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('child-ready', flush=True); time.sleep(3600)"])
        time.sleep(3600)
    '''))
    session = psi.PtySession([sys.executable, str(script), str(stubborn_launcher)])
    output = []

    def ready():
        output.append(session.read())
        return "child-ready" in "".join(output)

    try:
        qtbot.waitUntil(ready, timeout=10000)
        session.stop()
        assert session.poll() is not None
        qtbot.waitUntil(lambda: not session._reader.is_alive(), timeout=2000)
        descriptor = os.open(os.devnull, os.O_RDONLY)
        try:
            session.stop()
            assert os.fstat(descriptor)
        finally:
            os.close(descriptor)
    finally:
        try:
            os.killpg(session.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        session.proc.wait(timeout=5)
