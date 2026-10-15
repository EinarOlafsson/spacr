"""Terminal newline normalization is independent of read boundaries."""
import queue

import pytest

from spacr.qt.ai import pty_sign_in as terminal


def session():
    value = terminal.PtySession.__new__(terminal.PtySession)
    value._queue = queue.Queue()
    value._output_closed = False
    return value


@pytest.mark.parametrize("text", [
    "first\nsecond\n",
    "first\r\nsecond\r\n",
    "first\rsecond\r",
    "first\r\x1b[0m\nsecond\r\x1b]title\x07\n",
    "first\r\r\nsecond\r\n",
])
def test_every_split_matches_existing_complete_text_normalization(text):
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


def test_character_at_time_newlines_and_ansi_match_the_complete_prompt():
    value = session()
    text = "first\r\x1b[32m\nsecond\r\x1b]title\x1b\\\nlast\r"
    output = []
    for character in text:
        value._queue.put(character)
        output.append(value.read())
    value._output_closed = True
    output.append(value.read())
    assert "".join(output) == "first\nsecond\nlast\n"
    assert value.read() == ""


def test_pending_carriage_return_flushes_once_at_eof_without_new_output():
    value = session()
    value._queue.put("prompt\r")
    assert value.read() == "prompt"
    assert value.read() == ""
    value._output_closed = True
    assert value.read() == "\n"
    assert value.read() == ""


def test_carriage_return_before_non_lf_retains_standalone_newline_policy():
    value = session()
    value._queue.put("prompt\r")
    assert value.read() == "prompt"
    value._queue.put("answer")
    assert value.read() == "\nanswer"
    value._output_closed = True
    assert value.read() == ""


def test_malformed_nested_escape_retains_original_single_pass_stripping():
    value = session()
    text = "\x1b[\x1b[31m31m"
    value._queue.put(text)
    assert value.read() == terminal.strip_terminal_codes(text) == "\x1b[31m"
    value._output_closed = True
    assert value.read() == ""
