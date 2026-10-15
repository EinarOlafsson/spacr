"""Terminal read boundaries must not corrupt localized sign-in prompts."""
import errno
import queue
from types import SimpleNamespace

import pytest

from spacr.qt.ai import pty_sign_in as terminal


@pytest.mark.parametrize("close_kind", ["eof", "error"])
@pytest.mark.parametrize("chunks,expected", [
    ([b"Log ", b"\xc3", b"\x93", b" ", b"\xe4", b"\xb8", b"\xad",
      b"\xf0", b"\x9f", b"\x94", b"\x91"], "Log Ó 中🔑"),
    ([b"token ", b"\xf0", b"\x9f"], "token \ufffd"),
    ([b"bad ", b"\xff", b" ", b"\xe2", b"(", b"\xa1"], "bad \ufffd \ufffd(\ufffd"),
    ([b"ASCII prompt"], "ASCII prompt"),
])
def test_terminal_pump_decodes_short_reads_and_flushes_once(monkeypatch, chunks, expected, close_kind):
    reads = iter(chunks)
    def read(fd, size):
        assert fd == -420 and size == 4096
        try:
            return next(reads)
        except StopIteration:
            if close_kind == "error":
                raise OSError(errno.EIO, "disposable terminal closed")
            return b""
    monkeypatch.setattr(terminal, "os", SimpleNamespace(read=read))
    session = terminal.PtySession.__new__(terminal.PtySession)
    session._master = -420
    session._queue = queue.Queue()
    session._pump()
    assert session.read() == expected
    assert session.read() == ""
