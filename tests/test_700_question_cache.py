"""Answers to a fresh question are saved on this computer, for every provider."""

from __future__ import annotations

import pytest

from spacr import crashreport as cr

pytest.importorskip("PySide6")

from spacr.qt.ai import providers  # noqa: E402
from spacr.qt.ai.worker import StreamWorker  # noqa: E402


class _Provider(providers.ChatProvider):
    name = "claude"
    label = "Claude"

    def __init__(self, answer="Use **diameter** 30.\nCAUSE: none"):
        self.answer = answer
        self.calls = 0

    def stream_chat(self, messages, system="", model=None):
        self.calls += 1
        yield self.answer

    def cancel_stream(self):
        return None


def _run(provider, text):
    worker = StreamWorker(provider, [{"role": "user", "content": text}])
    out = []
    worker.finished.connect(lambda ok, final: out.append((ok, final)))
    worker.run()
    return worker, out[-1]


def test_the_question_is_normalised_like_the_relay_does_it():
    assert cr.normalise_question("  How do I -- pick a DIAMETER?!\n") == "how do i pick a diameter"
    assert cr.normalise_question("Hvað er <PATH> þvermál?") == "hvað er þvermál"
    assert cr.question_key("Pick, a diameter?", "mask") == cr.question_key("pick a diameter", "mask")
    assert cr.question_key("pick a diameter", "mask") != cr.question_key("pick a diameter", "measure")
    assert cr.question_key("pick a diameter").startswith("q:")


def test_personal_data_is_scrubbed_before_the_key_is_made():
    assert cr.question_key("why does /home/alice/plate1 fail") == cr.question_key("why does /home/bob/x fail")


def test_long_pasted_context_is_never_saved():
    assert cr.question_key("why?", "", 600) == ""
    assert cr.question_key("x " * 300) == ""
    assert cr.question_key("") == ""


def test_only_a_fresh_short_question_gets_a_key():
    one = [{"role": "user", "content": "How do I pick a diameter?"}]
    assert providers.question_cache_key(one).startswith("q:")
    follow_up = one + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "and then?"}]
    assert providers.question_cache_key(follow_up) == ""
    pasted = [{"role": "user", "content": "Why?\n<spacr_console_context>" + "log line\n" * 80
               + "</spacr_console_context>"}]
    assert providers.question_cache_key(pasted) == ""
    traceback = [{"role": "user", "content": "Why?\nTraceback (most recent call last):\n  x"}]
    assert providers.question_cache_key(traceback) == ""


def test_a_cli_provider_answers_once_and_the_saved_answer_comes_back():
    first = _Provider()
    worker, (ok, text) = _run(first, "How do I pick a diameter?")
    assert ok and first.calls == 1 and worker.saved is False
    second = _Provider("something else")
    worker, (ok, again) = _run(second, "how do i pick a DIAMETER")
    assert ok and again == text
    assert second.calls == 0 and worker.saved is True


def test_a_failed_or_cancelled_answer_is_not_saved():
    class _Failing(_Provider):
        def stream_chat(self, messages, system="", model=None):
            self.calls += 1
            raise providers.ProviderFailed("signed out")
            yield ""

    _run(_Failing(), "What is a plate?")
    later = _Provider("fresh")
    _, (ok, text) = _run(later, "What is a plate?")
    assert later.calls == 1 and text == "fresh"


def test_saved_answers_expire_with_the_version(monkeypatch):
    _run(_Provider("old"), "What is Measure?")
    monkeypatch.setattr(cr, "_spacr_version", lambda: "9.9.0")
    fresh = _Provider("new")
    _, (ok, text) = _run(fresh, "What is Measure?")
    assert fresh.calls == 1 and text == "new"


def test_a_prose_cause_sentence_is_not_taken_for_the_cause_line():
    assert cr.split_cause("Cause: a missing column in measure_crop.") == (
        "Cause: a missing column in measure_crop.", "")
    assert cr.split_cause("Fix it.\nCause: missing-column") == ("Fix it.", "missing-column")
    assert cr.split_cause("Fix it.\n**CAUSE:** Missing Column") == ("Fix it.", "missing-column")


def test_a_follow_up_is_never_answered_from_the_saved_answers():
    _run(_Provider("first"), "What is a well?")
    worker = StreamWorker(_Provider("second"), [
        {"role": "user", "content": "What is a well?"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "What is a well?"}])
    out = []
    worker.finished.connect(lambda ok, final: out.append(final))
    worker.run()
    assert out == ["second"] and worker.saved is False


def test_a_damaged_entry_or_a_failed_write_costs_only_a_new_answer(monkeypatch):
    key = cr.question_key("What is a tile?")
    cr.save_explanation(key, "kept")
    data = cr._read_saved()
    data[key]["created"] = "not a time"
    cr._saved_path().write_text(cr.json.dumps(data), encoding="utf-8")
    assert cr.saved_explanation(key) is None

    def _refuse(*_a, **_k):
        raise OSError("read-only")

    monkeypatch.setattr(cr.os, "replace", _refuse)
    cr.save_explanation(cr.question_key("What is a site?"), "lost")
    assert cr.saved_explanation(cr.question_key("What is a site?")) is None
