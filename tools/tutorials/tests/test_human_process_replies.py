"""Human-process reply framing and reentrant adapter routing, without Qt/GNOME."""
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import human_input  # noqa: E402
from capture_host_adapter import QTestAdapter  # noqa: E402


class _Requests(io.StringIO):
    def __init__(self, pipe):
        super().__init__()
        self.pipe = pipe

    def write(self, text):
        result = super().write(text)
        request = json.loads(text)
        self.pipe.sent.append(request['op'])
        if request['op'] == 'hello':
            self.pipe.reply({'value': {'size': [1, 1]}})
        elif self.pipe.on_request is not None:
            self.pipe.on_request(request)
        return result


class _Pipe:
    def __init__(self):
        reader, self.writer = os.pipe()
        self.stdout = os.fdopen(reader, 'r', encoding='utf-8', buffering=1)
        self.stdin = _Requests(self)
        self.sent = []
        self.on_request = None

    def reply(self, *replies):
        os.write(self.writer, ''.join(json.dumps(r) + '\n' for r in replies).encode())

    def close_writer(self):
        os.close(self.writer)
        self.writer = None

    def close(self):
        if self.writer is not None:
            self.close_writer()
        self.stdout.close()
        self.stdin.close()


@pytest.fixture
def transport(monkeypatch):
    pipe = _Pipe()
    monkeypatch.setattr(human_input, 'subprocess', SimpleNamespace(
        PIPE=-1, Popen=lambda *_args, **_kwargs: pipe))
    try:
        human = human_input.HumanProcess()
        assert (human.width, human.height) == (1, 1)
        yield human, pipe
    finally:
        pipe.close()


def test_batched_replies_remain_ready_in_fifo_order(transport):
    human, pipe = transport
    pipe.reply({'value': 'first'}, {'value': 'second'})
    assert human.reply_ready(0)
    assert human.collect() == 'first'
    assert human.reply_ready(0)
    assert human.collect() == 'second'
    assert not human.reply_ready(0)


def test_reentrant_adapter_receives_each_call_own_reply(transport):
    human, pipe = transport
    pending = []

    def answer_pair(request):
        pending.append(request['op'])
        if len(pending) == 2:
            pipe.reply(*({'value': op} for op in pending))

    pipe.on_request = answer_pair

    class Pump:
        entered = False
        inner = None

        def once(self):
            assert not self.entered, 'both replies arrived; the inner call must not wait again'
            self.entered = True
            self.inner = adapter.call('trace_mark')

    pump = Pump()
    adapter = QTestAdapter(human, pump=pump)
    assert adapter.call('position') == 'position'
    assert pump.inner == 'trace_mark'
    assert pipe.sent == ['hello', 'position', 'trace_mark']


def test_partial_utf8_and_json_wait_for_line_terminator(transport):
    human, pipe = transport
    line = (json.dumps({'value': 'λ'}, ensure_ascii=False) + '\n').encode()
    split = line.index('λ'.encode()) + 1
    os.write(pipe.writer, line[:split])
    assert not human.reply_ready(0)
    os.write(pipe.writer, line[split:-1])
    assert not human.reply_ready(0)
    os.write(pipe.writer, line[-1:])
    assert human.reply_ready(0)
    assert human.collect() == 'λ'


def test_error_reply_preserves_following_buffered_success(transport):
    human, pipe = transport
    pipe.reply({'error': 'expected failure'}, {'value': 42})
    assert human.reply_ready(0)
    with pytest.raises(RuntimeError, match='position: expected failure'):
        human.collect('position')
    assert human.reply_ready(0)
    assert human.collect() == 42


def test_eof_reports_server_exit(transport):
    human, pipe = transport
    pipe.close_writer()
    assert human.reply_ready(0)
    with pytest.raises(RuntimeError, match='server died during position'):
        human.collect('position')
