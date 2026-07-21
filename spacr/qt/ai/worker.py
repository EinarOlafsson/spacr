"""
QThread worker that streams chat completions from a ChatProvider so
the UI stays responsive during long generations.

Emits:
    stage_changed(str)  — coarse progress: "connecting", "streaming"
    chunk_ready(str)    — a partial completion chunk
    finished(bool, str) — (ok, full_text_or_error)
"""
from __future__ import annotations

import sys
import traceback
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QThread, Signal

from .providers import ChatProvider


class StreamWorker(QObject):
    stage_changed = Signal(str)
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
        self._cancelled = False

    def cancel(self) -> None:
        """Ask the worker loop to stop after the next chunk."""
        self._cancelled = True

    def run(self):
        buf: List[str] = []
        try:
            self.stage_changed.emit("connecting")
            stream = self._provider.stream_chat(
                self._messages, system=self._system, model=self._model
            )
            self.stage_changed.emit("streaming")
            for chunk in stream:
                if self._cancelled:
                    break
                if chunk:
                    buf.append(chunk)
                    self.chunk_ready.emit(chunk)
            if self._cancelled:
                self.finished.emit(False, "Cancelled.")
            else:
                self.finished.emit(True, "".join(buf))
        except BaseException as e:
            # BaseException — even a KeyboardInterrupt during a
            # blocking network call should let the UI recover instead
            # of leaving _thread wedged forever.
            tb = traceback.format_exc()
            # Print to real stderr so users can see it while we iterate.
            try:
                print(f"[AI worker] error: {tb}", file=sys.__stderr__, flush=True)
            except Exception:
                pass
            self.finished.emit(False, f"{type(e).__name__}: {e}")


def make_stream_thread(
    provider: ChatProvider,
    messages: List[Dict],
    system: str = "",
    model: Optional[str] = None,
):
    """Return (QThread, StreamWorker) — connect signals, then start().

    Callers MUST keep a reference to the returned worker (not just the
    thread) — otherwise Python may drop it before Qt has a chance to
    fire its finished signal.
    """
    from PySide6.QtCore import Qt
    thread = QThread()
    worker = StreamWorker(provider, messages, system=system, model=model)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    # Queue the deletion + quit so they run AFTER user-facing slots
    # (which are also queued but connected earlier).
    worker.finished.connect(thread.quit, Qt.QueuedConnection)
    worker.finished.connect(worker.deleteLater, Qt.QueuedConnection)
    thread.finished.connect(thread.deleteLater, Qt.QueuedConnection)
    return thread, worker
