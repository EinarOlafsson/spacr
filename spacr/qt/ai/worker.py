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
    """QObject that drives one provider stream on a worker QThread.

    :ivar stage_changed: coarse progress signal ("connecting", "streaming").
    :ivar chunk_ready: emitted with each partial completion chunk.
    :ivar finished: emitted with ``(ok, full_text_or_error)`` on completion.
    """

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
        """Prepare the worker; call :meth:`run` from a QThread's ``started`` signal.

        :param provider: the ChatProvider to stream from.
        :param messages: conversation history to send.
        :param system: optional system prompt.
        :param model: optional model override.
        """
        super().__init__()
        self._provider = provider
        self._messages = messages
        self._system = system
        self._model = model
        self._cancelled = False

    def cancel(self) -> None:
        """Cancel: kill the subprocess so the reader unblocks.

        Setting a Python flag alone isn't enough — the worker is
        blocked in a `for line in proc.stdout` iteration until the
        subprocess writes or closes. We terminate the subprocess
        directly via `provider.cancel_stream()`; the reader then
        exits with an empty read and run() completes cleanly.
        """
        self._cancelled = True
        try:
            self._provider.cancel_stream()
        except Exception:
            pass

    def run(self) -> None:
        """Consume the provider stream, emitting stage/chunk/finished signals."""
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
    parent: Optional[QObject] = None,
) -> tuple[QThread, StreamWorker]:
    """Return (QThread, StreamWorker) — connect signals, then start().

    IMPORTANT: pass a `parent` (typically the panel that owns this
    stream). Without a Qt parent the QThread's C++ object gets tied
    exclusively to Python's refcount — and dropping the ref while
    QThread.isRunning() is still True (which happens in the tiny
    window between worker.run returning and thread.finished firing)
    triggers Qt's `QThread: Destroyed while thread is still running /
    Aborted` crash. A parent keeps the C++ object alive until
    deleteLater runs.

    Callers must ALSO keep a Python reference to the worker until
    the stream truly finishes (see ConsolePanel._retire).

    Two wiring details are load-bearing; both are the same contract
    :func:`spacr.qt.bridge.make_thread` documents, and this function used
    to get them wrong:

    * ``worker.finished -> thread.quit`` is a **DirectConnection**. The
      QThread object is created here, on the GUI thread, so it is
      GUI-affine — a queued ``quit()`` is posted to the *GUI* thread's
      event queue, not to the worker's. Measured: with a queued
      connection, a GUI thread that goes straight into ``thread.wait()``
      (which is exactly what ``ConsolePanel.shutdown`` and every "drain
      before closing" path does) waits out its whole timeout on a worker
      that has already finished, because the event that would stop the
      thread is sitting behind the wait. ``QThread::quit`` is explicitly
      thread-safe, so calling it inline from the worker thread is correct.
    * There is deliberately **no** ``worker.deleteLater``. The worker's
      affinity is the worker thread, so a deferred delete is posted into a
      loop that is stopping, while the panel drops the object's last
      Python reference from the GUI thread — two owners, one object.
      ``bridge.make_thread``'s ownership essay records the gdb trace
      (``QThread -> sendPostedEvents -> ~QObject -> Sbk_GetPyOverride``)
      and the measurement: 3 crashes in 8 runs. A PySide6 object built in
      Python is already owned by Python; ``ConsolePanel``/``AIChatPanel``
      hold it in ``_retired`` until the thread has exited and free it
      there, on the thread that holds it.
    """
    from PySide6.QtCore import Qt
    thread = QThread(parent)
    worker = StreamWorker(provider, messages, system=system, model=model)
    worker.setParent(None)              # worker moves to thread, no parent
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit, Qt.DirectConnection)
    # The QThread is GUI-affine, so its deferred delete is flushed by the
    # GUI thread's own loop. That one is safe, and it is the only one.
    thread.finished.connect(thread.deleteLater, Qt.QueuedConnection)
    return thread, worker
