"""Saved questions must not hide prompt updates or a hosted relay deployment."""
from spacr import crashreport
from spacr.qt.ai.providers import HOSTED_PROVIDER_NAME, question_cache_key
from spacr.qt.ai.worker import StreamWorker


class Provider:
    def __init__(self, name):
        self.name = name
        self.calls = 0

    def stream_chat(self, messages, system='', model=None):
        self.calls += 1
        yield f'{self.name}: {system}: {model}'


def run(provider, system='current', model='model-a'):
    worker = StreamWorker(provider, [{'role': 'user', 'content': 'how do i use spacr?'}],
                          system=system, model=model)
    replies = []
    worker.finished.connect(lambda ok, text: replies.append((ok, text)))
    worker.run()
    assert replies and replies[0][0]
    return worker, replies[0][1]


def test_hosted_ignores_stale_desktop_question_cache_and_always_checks_relay(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(crashreport, '_reports_dir', lambda: tmp_path)
    messages = [{'role': 'user', 'content': 'how do i use spacr?'}]
    old_key = question_cache_key(messages)
    crashreport.save_explanation(old_key, 'Type Mask Module name in the Navigator.')
    provider = Provider(HOSTED_PROVIDER_NAME)
    for _ in range(2):
        worker, answer = run(provider)
        assert not worker.saved
        assert 'Module name' not in answer
    assert provider.calls == 2
    assert list(crashreport._read_saved()) == [old_key]


def test_local_questions_reuse_only_same_prompt_provider_and_model(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(crashreport, '_reports_dir', lambda: tmp_path)
    provider = Provider('codex')
    first, text = run(provider)
    assert not first.saved
    same, repeat = run(provider)
    assert same.saved and repeat == text and provider.calls == 1
    changed, text2 = run(provider, system='new prompt')
    assert not changed.saved and text2 != text
    model_changed, _ = run(provider, model='model-b')
    assert not model_changed.saved
    other = Provider('gemini')
    switched, _ = run(other)
    assert not switched.saved and other.calls == 1
    assert provider.calls == 3
