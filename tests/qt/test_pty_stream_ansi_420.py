"""Terminal formatting is independent of arbitrary read boundaries."""
import queue

import pytest

from spacr.qt.ai import pty_sign_in as terminal


def session():
    value = terminal.PtySession.__new__(terminal.PtySession)
    value._queue = queue.Queue()
    value._output_closed = False
    return value


@pytest.mark.parametrize("text", [
    "Open \x1b[1;32mhttps://example.org/auth?state=abc\x1b[0m now\n",
    "Open \x1b]0;Private sign-in title\x07https://example.org/auth now\n",
    "Open \x1b]0;Private sign-in title\x1b\\https://example.org/auth now\n",
    "Open \x1bMhttps://example.org/auth now\n",
    "ASCII and Ó 中文 🔑 https://example.org/auth now\n",
])
def test_clean_terminal_text_is_independent_of_every_single_split(text):
    expected = terminal.strip_terminal_codes(text)
    for split in range(1, len(text)):
        value = session()
        value._queue.put(text[:split])
        first = value.read()
        value._queue.put(text[split:])
        second = value.read()
        value._output_closed = True
        final = value.read()
        assert first + second + final == expected, (split, first, second, final)
        assert value.read() == ""


def test_single_character_reads_preserve_prompt_and_urls_without_escape_fragments():
    text = "\x1b]title\x1b\\\x1b[32mhttps://example.org/auth\x1b[0m prompt: "
    value = session()
    output = []
    for character in text:
        value._queue.put(character)
        output.append(value.read())
    value._output_closed = True
    output.append(value.read())
    assert "".join(output) == "https://example.org/auth prompt: "
    assert all("\x1b" not in part for part in output)
    assert terminal.find_urls("".join(output)) == ["https://example.org/auth"]
    assert value.read() == ""


@pytest.mark.parametrize("text", ["prompt \x1b", "prompt \x1b[31", "prompt \x1b]title"])
def test_eof_preserves_stateless_incomplete_sequence_policy_exactly_once(text):
    value = session()
    value._queue.put(text)
    first = value.read()
    value._output_closed = True
    final = value.read()
    assert first + final == terminal.strip_terminal_codes(text)
    assert value.read() == ""


def test_reader_closure_during_queue_drain_waits_for_final_queued_escape_bytes():
    value = session()
    value._queue.put("\x1b[")
    assert value.read() == ""

    class ClosingQueue(queue.Queue):
        def get_nowait(self):
            try:
                return super().get_nowait()
            except queue.Empty:
                if not value._output_closed:
                    self.put("32mcomplete\x1b[0m")
                    value._output_closed = True
                raise

    value._queue = ClosingQueue()
    assert value.read() == ""
    assert value.read() == "complete"
    assert value.read() == ""


@pytest.mark.parametrize("text", ["prompt \x1b[31\n", "prompt \x1b! value"])
def test_nonmatching_escape_fragments_keep_existing_stateless_policy(text):
    value = session()
    value._queue.put(text)
    assert value.read() == terminal.strip_terminal_codes(text)
    assert value.read() == ""
