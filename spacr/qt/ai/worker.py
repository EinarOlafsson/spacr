"""
QThread worker that streams chat completions from a ChatProvider so
the UI stays responsive during long generations.

Emits:
    chunk_ready(str)    — a partial completion chunk
    finished(bool, str) — (ok, full_text_or_error)
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QThread, Signal

from .providers import ChatProvider


class StreamWorker(QObject):
    chunk_ready = Signal(str)
    finished = Signal(bool, str)

    def __init__(
        self,
        provider: ChatProvider,
        messages: List[Dict],
        system: str = "",
        model: Optional[str] = None,
    ):
        super().__init__()
        self._provider = provider
        self._messages = messages
        self._system = system
        self._model = model

    def run(self):
        buf: List[str] = []
        try:
            for chunk in self._provider.stream_chat(
                self._messages, system=self._system, model=self._model
            ):
                buf.append(chunk)
                self.chunk_ready.emit(chunk)
            self.finished.emit(True, "".join(buf))
        except Exception as e:
            self.finished.emit(False, f"{type(e).__name__}: {e}")


def make_stream_thread(
    provider: ChatProvider,
    messages: List[Dict],
    system: str = "",
    model: Optional[str] = None,
):
    """Return (QThread, StreamWorker) — connect signals, then start()."""
    thread = QThread()
    worker = StreamWorker(provider, messages, system=system, model=model)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    return thread, worker
