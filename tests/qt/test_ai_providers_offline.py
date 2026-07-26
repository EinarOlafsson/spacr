"""Fully-offline tests for ``spacr.qt.ai.providers`` and ``spacr.qt.ai.worker``.

The three providers shell out to a vendor CLI rather than talking HTTP,
so the transport boundary here is :func:`subprocess.Popen`. Every test
below either

* replaces ``providers.subprocess.Popen`` with a recorder and asserts the
  EXACT argv / env / stdin that WOULD have been executed, or
* spawns a tiny ``sys.executable -c`` child (no network, milliseconds).

Nothing here touches the network, and no test spawns a real ``claude`` /
``codex`` / ``gemini`` binary even when one happens to be on PATH.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading

import pytest


# ---------------------------------------------------------------------------
# Fake Popen — the transport boundary
# ---------------------------------------------------------------------------

class _FakeStdout:
    """Iterable stdout that can be told to blow up on close()."""

    def __init__(self, lines, raise_on_close=False):
        self._lines = list(lines)
        self._i = 0
        self.closed = False
        self._raise_on_close = raise_on_close

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self._lines):
            raise StopIteration
        line = self._lines[self._i]
        self._i += 1
        return line

    def close(self):
        if self._raise_on_close:
            raise OSError("stdout already gone")
        self.closed = True


class _FakeStdin:
    def __init__(self, broken=False):
        self.written = []
        self.closed = False
        self._broken = broken

    def write(self, text):
        if self._broken:
            raise BrokenPipeError("child went away")
        self.written.append(text)

    def close(self):
        self.closed = True


class _FakeProc:
    """Stand-in for subprocess.Popen with scriptable wait() behaviour."""

    def __init__(self, lines=(), *, stdin=None, wait_raises=0,
                 raise_on_close=False, terminate_raises=False):
        self.stdout = _FakeStdout(lines, raise_on_close=raise_on_close)
        self.stdin = stdin
        self.calls = []
        self._wait_raises = wait_raises      # how many wait()s raise first
        self._terminate_raises = terminate_raises

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        if self._wait_raises > 0:
            self._wait_raises -= 1
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0)
        return 0

    def terminate(self):
        self.calls.append(("terminate", None))
        if self._terminate_raises:
            raise OSError("no such process")

    def kill(self):
        self.calls.append(("kill", None))

    def poll(self):
        return 0


def _install_fake_popen(monkeypatch, proc):
    """Route providers.subprocess.Popen to `proc`, recording its kwargs."""
    from spacr.qt.ai import providers as pmod
    rec = {}

    def _popen(argv, **kwargs):
        rec["argv"] = list(argv)
        rec.update(kwargs)
        return proc

    monkeypatch.setattr(pmod.subprocess, "Popen", _popen)
    return rec


# ---------------------------------------------------------------------------
# _stream_process — the shared subprocess helper
# ---------------------------------------------------------------------------

def test_stream_process_yields_lines_and_drops_permission_noise(monkeypatch):
    """Claude Code's per-file permission reminders must not reach the user."""
    from spacr.qt.ai.providers import _stream_process

    lines = [
        "Permission deny rule matched Read(/etc/shadow)\n",
        "hello there\n",
        "Permission allow rule matched Bash(ls)\n",
        "second line\n",
        "Permission ask rule matched Write(x)\n",
        "  Permission deny rule (indented — NOT a prefix match)\n",
    ]
    proc = _FakeProc(lines)
    _install_fake_popen(monkeypatch, proc)

    out = list(_stream_process(["claude", "-p", "hi"]))
    assert out == [
        "hello there\n",
        "second line\n",
        "  Permission deny rule (indented — NOT a prefix match)\n",
    ]


def test_stream_process_passes_argv_env_and_merges_stderr(monkeypatch):
    from spacr.qt.ai.providers import _stream_process

    proc = _FakeProc(["ok\n"])
    rec = _install_fake_popen(monkeypatch, proc)
    monkeypatch.setenv("SPACR_TEST_MARKER", "outer")

    assert list(_stream_process(["gemini", "-p", "q"],
                                 env_extra={"EXTRA_VAR": "1"})) == ["ok\n"]

    assert rec["argv"] == ["gemini", "-p", "q"]
    assert rec["stdout"] is subprocess.PIPE
    assert rec["stderr"] is subprocess.STDOUT   # stderr folded into stdout
    assert rec["stdin"] is None                 # no stdin_text -> no pipe
    assert rec["text"] is True
    assert rec["bufsize"] == 1                  # line buffered
    # env_extra is layered on top of a copy of os.environ, not replacing it
    assert rec["env"]["EXTRA_VAR"] == "1"
    assert rec["env"]["SPACR_TEST_MARKER"] == "outer"
    assert os.environ.get("EXTRA_VAR") is None  # caller env untouched


def test_stream_process_writes_stdin_text_then_closes(monkeypatch):
    from spacr.qt.ai.providers import _stream_process

    stdin = _FakeStdin()
    proc = _FakeProc(["reply\n"], stdin=stdin)
    rec = _install_fake_popen(monkeypatch, proc)

    assert list(_stream_process(["codex", "exec"], stdin_text="prompt body")) \
        == ["reply\n"]
    assert rec["stdin"] is subprocess.PIPE
    assert stdin.written == ["prompt body"]
    assert stdin.closed is True


def test_stream_process_survives_broken_stdin_pipe(monkeypatch):
    """A child that exits before reading stdin must not kill the stream."""
    from spacr.qt.ai.providers import _stream_process

    stdin = _FakeStdin(broken=True)
    proc = _FakeProc(["still got output\n"], stdin=stdin)
    _install_fake_popen(monkeypatch, proc)

    assert list(_stream_process(["codex", "exec"], stdin_text="x")) == [
        "still got output\n"
    ]
    assert stdin.written == []          # the write raised
    assert stdin.closed is False        # close() never reached


def test_stream_process_missing_cli_raises_actionable_runtime_error(monkeypatch):
    from spacr.qt.ai import providers as pmod

    def _boom(argv, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", argv[0])

    monkeypatch.setattr(pmod.subprocess, "Popen", _boom)
    gen = pmod._stream_process(["claude", "-p", "hi"])
    with pytest.raises(RuntimeError) as exc:
        next(gen)
    assert "'claude'" in str(exc.value)
    assert "on PATH" in str(exc.value)
    assert isinstance(exc.value.__cause__, FileNotFoundError)


def test_stream_process_registers_and_clears_current_proc(monkeypatch):
    from spacr.qt.ai import providers as pmod

    provider = pmod.ClaudeCliProvider()
    proc = _FakeProc(["a\n", "b\n"])
    _install_fake_popen(monkeypatch, proc)

    gen = pmod._stream_process(["claude"], provider=provider)
    assert provider._current_proc is None
    assert next(gen) == "a\n"
    assert provider._current_proc is proc     # registered once running
    assert list(gen) == ["b\n"]
    assert provider._current_proc is None     # cleared in finally


def test_stream_process_cleanup_escalates_wait_terminate_kill(monkeypatch):
    """When the child ignores us the teardown must escalate, and a
    stdout.close() that throws must not mask that."""
    from spacr.qt.ai.providers import _stream_process

    proc = _FakeProc(["x\n"], wait_raises=2, raise_on_close=True)
    _install_fake_popen(monkeypatch, proc)

    assert list(_stream_process(["claude"])) == ["x\n"]
    kinds = [c[0] for c in proc.calls]
    # first wait raised -> terminate -> second wait raised -> kill
    assert kinds == ["wait", "terminate", "wait", "kill"]


def test_stream_process_cleanup_survives_terminate_failure(monkeypatch):
    from spacr.qt.ai.providers import _stream_process

    proc = _FakeProc(["x\n"], wait_raises=1, terminate_raises=True)
    _install_fake_popen(monkeypatch, proc)

    assert list(_stream_process(["claude"])) == ["x\n"]
    assert [c[0] for c in proc.calls] == ["wait", "terminate"]


def test_stream_process_real_child_filters_and_streams():
    """One real (local, offline) child process end-to-end."""
    from spacr.qt.ai.providers import _stream_process

    code = (
        "import sys\n"
        "sys.stdout.write('Permission deny rule matched Read(x)\\n')\n"
        "sys.stdout.write('alpha\\n')\n"
        "sys.stderr.write('beta-on-stderr\\n')\n"
        "sys.stdout.flush(); sys.stderr.flush()\n"
    )
    out = "".join(_stream_process([sys.executable, "-c", code]))
    assert "Permission deny rule" not in out
    assert "alpha" in out
    assert "beta-on-stderr" in out       # stderr merged into stdout


# ---------------------------------------------------------------------------
# cancel_stream — escalation + failure tolerance
# ---------------------------------------------------------------------------

def test_cancel_stream_terminates_then_waits():
    from spacr.qt.ai.providers import ClaudeCliProvider

    p = ClaudeCliProvider()
    proc = _FakeProc()
    p._current_proc = proc
    p.cancel_stream()
    # A child that dies on SIGTERM is reaped by the first wait — no kill.
    assert proc.calls == [("terminate", None), ("wait", 1)]


def test_cancel_stream_is_a_noop_when_nothing_is_running():
    from spacr.qt.ai.providers import ClaudeCliProvider

    p = ClaudeCliProvider()
    assert p._current_proc is None
    assert p.cancel_stream() is None      # must not raise
    assert p._current_proc is None


def test_cancel_stream_kills_when_terminate_is_ignored():
    from spacr.qt.ai.providers import CodexCliProvider

    p = CodexCliProvider()
    proc = _FakeProc(wait_raises=2)     # both waits time out
    p._current_proc = proc
    p.cancel_stream()
    assert [c[0] for c in proc.calls] == ["terminate", "wait", "kill", "wait"]


def test_cancel_stream_swallows_terminate_errors():
    from spacr.qt.ai.providers import GeminiCliProvider

    p = GeminiCliProvider()
    proc = _FakeProc(terminate_raises=True)
    p._current_proc = proc
    p.cancel_stream()                    # must not raise
    assert [c[0] for c in proc.calls] == ["terminate"]


# ---------------------------------------------------------------------------
# Provider argv construction — the whole point of each stream_chat
# ---------------------------------------------------------------------------

@pytest.fixture()
def speed(monkeypatch, tmp_path):
    """Isolate ai settings QSettings into a temp .ini file."""
    from PySide6.QtCore import QSettings
    from spacr.qt.ai import settings as ai_settings

    store = QSettings(str(tmp_path / "ai.ini"), QSettings.IniFormat)
    monkeypatch.setattr(ai_settings, "_settings", lambda: store)
    return ai_settings


def test_claude_argv_uses_speed_model_and_appends_system(monkeypatch, speed):
    from spacr.qt.ai.providers import ClaudeCliProvider

    speed.set_response_speed("deep")
    proc = _FakeProc(["hi\n"])
    rec = _install_fake_popen(monkeypatch, proc)

    p = ClaudeCliProvider()
    msgs = [{"role": "user", "content": "why did masks fail"}]
    assert list(p.stream_chat(msgs, system="be terse")) == ["hi\n"]

    argv = rec["argv"]
    assert argv[0] == "claude"
    assert argv[1] == "-p"
    assert "why did masks fail" in argv[2]
    assert "System:\nbe terse" in argv[2]
    assert argv[3:5] == ["--append-system-prompt", "be terse"]
    assert argv[5:] == ["--model", "opus"]       # "deep" -> opus


def test_claude_argv_explicit_model_beats_speed_setting(monkeypatch, speed):
    from spacr.qt.ai.providers import ClaudeCliProvider

    speed.set_response_speed("fast")             # would give --model haiku
    proc = _FakeProc([])
    rec = _install_fake_popen(monkeypatch, proc)

    p = ClaudeCliProvider()
    assert list(p.stream_chat([{"role": "user", "content": "q"}],
                               model="sonnet")) == []
    assert rec["argv"] == ["claude", "-p", "User:\nq", "--model", "sonnet"]
    assert "haiku" not in rec["argv"]
    assert "--append-system-prompt" not in rec["argv"]   # no system given


def test_codex_argv_uses_exec_subcommand(monkeypatch, speed):
    from spacr.qt.ai.providers import CodexCliProvider

    speed.set_response_speed("balanced")
    proc = _FakeProc(["out\n"])
    rec = _install_fake_popen(monkeypatch, proc)

    p = CodexCliProvider()
    assert list(p.stream_chat([{"role": "user", "content": "hey"}],
                               system="persona")) == ["out\n"]
    argv = rec["argv"]
    assert argv[0:2] == ["codex", "exec"]
    # codex has no --append-system-prompt: the system prompt is folded in
    assert argv[2].startswith("System:\npersona")
    assert argv[3:] == ["--model", "gpt-5"]


def test_codex_argv_explicit_model(monkeypatch, speed):
    from spacr.qt.ai.providers import CodexCliProvider

    speed.set_response_speed("deep")
    rec = _install_fake_popen(monkeypatch, _FakeProc([]))
    p = CodexCliProvider()
    list(p.stream_chat([{"role": "user", "content": "q"}], model="o3"))
    assert rec["argv"] == ["codex", "exec", "User:\nq", "--model", "o3"]


def test_gemini_argv_translates_model_flag_to_dash_m(monkeypatch, speed):
    """SPEED_MAP speaks --model; the gemini CLI wants -m."""
    from spacr.qt.ai.providers import GeminiCliProvider

    speed.set_response_speed("fast")
    rec = _install_fake_popen(monkeypatch, _FakeProc(["g\n"]))
    p = GeminiCliProvider()
    assert list(p.stream_chat([{"role": "user", "content": "q"}])) == ["g\n"]
    assert rec["argv"] == ["gemini", "-p", "User:\nq",
                           "-m", "gemini-2.5-flash"]
    assert "--model" not in rec["argv"]


def test_gemini_argv_passes_non_model_speed_args_through(monkeypatch, speed):
    """If SPEED_MAP ever grows a non---model flag it must pass verbatim."""
    from spacr.qt.ai import providers as pmod
    from spacr.qt.ai import settings as ai_settings

    monkeypatch.setattr(ai_settings, "provider_args",
                        lambda name: ["--thinking", "high"])
    rec = _install_fake_popen(monkeypatch, _FakeProc([]))
    list(pmod.GeminiCliProvider().stream_chat(
        [{"role": "user", "content": "q"}]))
    assert rec["argv"] == ["gemini", "-p", "User:\nq", "--thinking", "high"]


def test_gemini_argv_with_empty_speed_args(monkeypatch, speed):
    from spacr.qt.ai import providers as pmod
    from spacr.qt.ai import settings as ai_settings

    monkeypatch.setattr(ai_settings, "provider_args", lambda name: [])
    rec = _install_fake_popen(monkeypatch, _FakeProc([]))
    list(pmod.GeminiCliProvider().stream_chat(
        [{"role": "user", "content": "q"}], model=None))
    assert rec["argv"] == ["gemini", "-p", "User:\nq"]


def test_gemini_argv_explicit_model(monkeypatch, speed):
    from spacr.qt.ai import providers as pmod

    rec = _install_fake_popen(monkeypatch, _FakeProc([]))
    list(pmod.GeminiCliProvider().stream_chat(
        [{"role": "user", "content": "q"}], model="gemini-3"))
    assert rec["argv"] == ["gemini", "-p", "User:\nq", "-m", "gemini-3"]


def test_every_stream_chat_registers_provider_for_cancellation(monkeypatch,
                                                                speed):
    """Each provider must hand itself to _stream_process, else cancel is a
    no-op and the worker thread wedges forever."""
    from spacr.qt.ai import providers as pmod

    for cls in (pmod.ClaudeCliProvider, pmod.CodexCliProvider,
                pmod.GeminiCliProvider):
        proc = _FakeProc(["z\n"])
        _install_fake_popen(monkeypatch, proc)
        p = cls()
        gen = p.stream_chat([{"role": "user", "content": "q"}])
        assert next(gen) == "z\n"
        assert p._current_proc is proc, f"{cls.__name__} never registered"
        list(gen)
        assert p._current_proc is None


# ---------------------------------------------------------------------------
# _format_conversation
# ---------------------------------------------------------------------------

def test_format_conversation_without_system_or_history():
    from spacr.qt.ai.providers import _format_conversation
    assert _format_conversation([{"role": "user", "content": "solo"}]) == \
        "User:\nsolo"


def test_format_conversation_empty_history_is_empty_string():
    from spacr.qt.ai.providers import _format_conversation
    assert _format_conversation([]) == ""
    assert _format_conversation([], system="ctx") == "System:\nctx\n"


def test_format_conversation_defaults_missing_fields():
    from spacr.qt.ai.providers import _format_conversation
    out = _format_conversation([{}, {"role": "assistant"}])
    assert out == "User:\n\n\nAssistant:\n"


def test_format_conversation_maps_every_non_user_role_to_assistant():
    from spacr.qt.ai.providers import _format_conversation
    out = _format_conversation([
        {"role": "system", "content": "sys-turn"},
        {"role": "user", "content": "last"},
    ])
    assert "Assistant:\nsys-turn" in out
    assert out.endswith("User:\nlast")


# ---------------------------------------------------------------------------
# Install / login state
# ---------------------------------------------------------------------------

def test_is_logged_in_and_is_configured_follow_installed(monkeypatch):
    from spacr.qt.ai import providers as pmod

    monkeypatch.setattr(pmod.shutil, "which",
                        lambda n: "/usr/local/bin/codex" if n == "codex"
                        else None)
    codex = pmod.get_provider("codex")
    claude = pmod.get_provider("claude")
    assert codex.is_logged_in() is True
    assert codex.is_configured() is True
    assert claude.is_logged_in() is False
    assert claude.is_configured() is False
    assert codex.source_of_key() == "CLI found at /usr/local/bin/codex"


def test_list_providers_returns_a_fresh_list_each_call():
    from spacr.qt.ai import providers as pmod
    a, b = pmod.list_providers(), pmod.list_providers()
    assert a == b and a is not b        # copy, so callers can't mutate registry
    a.clear()
    assert len(pmod.list_providers()) == 3


def test_configured_providers_empty_when_nothing_installed(monkeypatch):
    from spacr.qt.ai import providers as pmod
    monkeypatch.setattr(pmod.shutil, "which", lambda n: None)
    assert pmod.configured_providers() == []
    for p in pmod.list_providers():
        assert p.source_of_key() == "CLI not installed"


def test_get_provider_resolves_ids_and_rejects_unknown_ones():
    from spacr.qt.ai import providers as pmod
    for name in ("claude", "codex", "gemini"):
        assert pmod.get_provider(name).name == name
    assert pmod.get_provider("gpt4") is None
    assert pmod.get_provider("") is None
    assert pmod.get_provider("Claude") is None      # id lookup is exact


def test_abstract_base_cannot_be_instantiated():
    from spacr.qt.ai.providers import ChatProvider
    with pytest.raises(TypeError):
        ChatProvider()


# ---------------------------------------------------------------------------
# StreamWorker
# ---------------------------------------------------------------------------

class _ScriptedProvider:
    """Minimal ChatProvider-shaped double with a scripted stream."""

    name = "scripted"
    label = "Scripted"

    def __init__(self, chunks=(), raise_exc=None, cancel_raises=False):
        self._chunks = list(chunks)
        self._raise = raise_exc
        self.cancelled = 0
        self._cancel_raises = cancel_raises
        self.seen = {}

    def cancel_stream(self):
        self.cancelled += 1
        if self._cancel_raises:
            raise OSError("cannot signal child")

    def stream_chat(self, messages, system="", model=None):
        self.seen = {"messages": messages, "system": system, "model": model}
        if self._raise is not None:
            raise self._raise
        for c in self._chunks:
            yield c


def _collect(worker):
    got = {"stages": [], "chunks": [], "finished": []}
    worker.stage_changed.connect(got["stages"].append)
    worker.chunk_ready.connect(got["chunks"].append)
    worker.finished.connect(lambda ok, txt: got["finished"].append((ok, txt)))
    return got


def test_worker_streams_chunks_and_joins_them(qtbot, qt_theme_applied):
    from spacr.qt.ai.worker import StreamWorker

    provider = _ScriptedProvider(["Hel", "lo ", "world"])
    msgs = [{"role": "user", "content": "hi"}]
    w = StreamWorker(provider, msgs, system="sys", model="m1")
    got = _collect(w)
    w.run()

    assert got["stages"] == ["connecting", "streaming"]
    assert got["chunks"] == ["Hel", "lo ", "world"]
    assert got["finished"] == [(True, "Hello world")]
    # the worker forwarded exactly what it was constructed with
    assert provider.seen == {"messages": msgs, "system": "sys", "model": "m1"}


def test_worker_drops_empty_chunks_but_keeps_streaming(qtbot, qt_theme_applied):
    from spacr.qt.ai.worker import StreamWorker

    w = StreamWorker(_ScriptedProvider(["a", "", None, "b"]), [])
    got = _collect(w)
    w.run()
    assert got["chunks"] == ["a", "b"]          # falsy chunks not emitted
    assert got["finished"] == [(True, "ab")]


def test_worker_empty_stream_finishes_ok_with_empty_text(qtbot,
                                                          qt_theme_applied):
    from spacr.qt.ai.worker import StreamWorker

    w = StreamWorker(_ScriptedProvider([]), [])
    got = _collect(w)
    w.run()
    assert got["stages"] == ["connecting", "streaming"]
    assert got["chunks"] == []
    assert got["finished"] == [(True, "")]


def test_worker_cancel_stops_mid_stream_and_reports_cancelled(
        qtbot, qt_theme_applied):
    from spacr.qt.ai.worker import StreamWorker

    provider = _ScriptedProvider(["one", "two", "three", "four"])
    w = StreamWorker(provider, [])
    got = _collect(w)
    # Cancel as soon as the second chunk lands.
    w.chunk_ready.connect(lambda _c: w.cancel() if len(got["chunks"]) == 2
                          else None)
    w.run()

    assert got["chunks"] == ["one", "two"]
    assert got["finished"] == [(False, "Cancelled.")]
    assert provider.cancelled == 1      # delegated to the provider


def test_worker_cancel_before_run_yields_no_chunks(qtbot, qt_theme_applied):
    from spacr.qt.ai.worker import StreamWorker

    provider = _ScriptedProvider(["a", "b"])
    w = StreamWorker(provider, [])
    got = _collect(w)
    w.cancel()
    w.run()
    assert got["chunks"] == []
    assert got["finished"] == [(False, "Cancelled.")]


def test_worker_cancel_swallows_provider_errors(qtbot, qt_theme_applied):
    from spacr.qt.ai.worker import StreamWorker

    provider = _ScriptedProvider([], cancel_raises=True)
    w = StreamWorker(provider, [])
    w.cancel()                       # must not raise
    assert provider.cancelled == 1
    assert w._cancelled is True


def test_worker_reports_provider_exception_as_failure(qtbot, qt_theme_applied,
                                                       capfd):
    from spacr.qt.ai.worker import StreamWorker

    w = StreamWorker(_ScriptedProvider(raise_exc=RuntimeError("no CLI")), [])
    got = _collect(w)
    w.run()
    assert got["chunks"] == []
    assert got["finished"] == [(False, "RuntimeError: no CLI")]
    # traceback goes to the REAL stderr (sys.__stderr__) for debugging
    err = capfd.readouterr().err
    assert "[AI worker] error" in err
    assert "RuntimeError: no CLI" in err


def test_worker_survives_a_baseexception(qtbot, qt_theme_applied):
    """A KeyboardInterrupt inside a blocking read must still resolve the UI."""
    from spacr.qt.ai.worker import StreamWorker

    w = StreamWorker(_ScriptedProvider(raise_exc=KeyboardInterrupt()), [])
    got = _collect(w)
    w.run()                                   # must NOT propagate
    assert got["finished"] == [(False, "KeyboardInterrupt: ")]


def test_worker_stderr_print_failure_does_not_mask_the_error(
        qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt.ai import worker as wmod

    class _DeadStderr:
        def write(self, *a, **k):
            raise ValueError("stderr closed")
        def flush(self):
            pass

    monkeypatch.setattr(wmod.sys, "__stderr__", _DeadStderr())
    w = wmod.StreamWorker(_ScriptedProvider(raise_exc=ValueError("boom")), [])
    got = _collect(w)
    w.run()
    assert got["finished"] == [(False, "ValueError: boom")]


# ---------------------------------------------------------------------------
# make_stream_thread
# ---------------------------------------------------------------------------

def test_make_stream_thread_parents_thread_and_detaches_worker(
        qtbot, qt_theme_applied):
    from PySide6.QtCore import QObject, QThread
    from spacr.qt.ai.worker import make_stream_thread

    owner = QObject()
    provider = _ScriptedProvider(["x"])
    thread, worker = make_stream_thread(provider, [], parent=owner)
    try:
        assert isinstance(thread, QThread)
        assert thread.parent() is owner       # Qt owns the C++ lifetime
        assert worker.parent() is None        # worker lives on the thread
        assert worker.thread() is thread
        assert not thread.isRunning()         # caller must start() it
    finally:
        thread.setParent(None)


def _thread_stopped(thread) -> bool:
    """True once the QThread has exited — or Qt already reclaimed it.

    Polled rather than waited on: ``qtbot.waitSignal(thread.finished)``
    races here, because pytest-qt's blocker is a plain Python callable, so
    Qt invokes it directly on the WORKER thread and it then calls
    ``QEventLoop.quit()`` on the GUI thread's loop from the wrong thread.
    """
    try:
        return not thread.isRunning()
    except RuntimeError:            # C++ half already deleteLater'd
        return True


def test_make_stream_thread_runs_the_stream_and_quits(qtbot, qt_theme_applied):
    from PySide6.QtCore import QObject
    from spacr.qt.ai.worker import make_stream_thread

    owner = QObject()
    provider = _ScriptedProvider(["alpha", "beta"])
    thread, worker = make_stream_thread(provider, [{"role": "user",
                                                    "content": "go"}],
                                        system="s", parent=owner)
    chunks = []
    done = []
    worker.chunk_ready.connect(chunks.append)
    worker.finished.connect(lambda ok, t: done.append((ok, t)))

    thread.start()
    qtbot.waitUntil(lambda: len(done) == 1, timeout=5000)

    assert chunks == ["alpha", "beta"]
    assert done == [(True, "alphabeta")]
    assert provider.seen == {"messages": [{"role": "user", "content": "go"}],
                             "system": "s", "model": None}
    # the thread quits itself once the worker reports finished
    qtbot.waitUntil(lambda: _thread_stopped(thread), timeout=5000)


def test_make_stream_thread_worker_runs_off_the_gui_thread(qtbot,
                                                            qt_theme_applied):
    from PySide6.QtCore import QObject
    from spacr.qt.ai.worker import make_stream_thread

    seen = {}
    done = []

    class _ThreadNoting(_ScriptedProvider):
        def stream_chat(self, messages, system="", model=None):
            seen["tid"] = threading.get_ident()
            yield "done"

    owner = QObject()
    thread, worker = make_stream_thread(_ThreadNoting(), [], parent=owner)
    worker.finished.connect(lambda ok, t: done.append((ok, t)))
    thread.start()
    qtbot.waitUntil(lambda: len(done) == 1, timeout=5000)

    assert done == [(True, "done")]
    assert seen["tid"] != threading.get_ident()
    qtbot.waitUntil(lambda: _thread_stopped(thread), timeout=5000)
