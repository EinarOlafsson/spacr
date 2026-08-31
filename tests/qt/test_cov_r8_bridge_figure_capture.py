"""Whether a pipeline run intercepts Matplotlib, and what it costs not to.

`PipelineWorker` normally replaces `plt.show()` so figures land in the
UI instead of a blocking window. Read-only UI jobs turn that off --
`capture_figures=False` -- and the reason is in the constructor's own
docstring: polling must not load the plotting stack merely by opening a
screen.

That opt-out is implemented by raising `_SkipFigureCapture` inside the
same try that would have done the import, so one handler covers both
"we chose not to" and "we tried and could not". Neither had been run.
"""
from __future__ import annotations

import sys

import pytest

from spacr.qt.bridge import PipelineWorker

pytestmark = pytest.mark.qt


def _run_worker(worker):
    """Run the slot to completion, as QThread.started would."""
    worker.run()
    return worker


class TestARunThatDeclinesFigureCapture:

    def test_a_read_only_job_does_not_load_matplotlib_pyplot(self,
                                                             monkeypatch):
        """THE UNCOVERED OPT-OUT.

        A screen that polls for history must not pull the plotting stack
        in as a side effect. The worker signals that by raising
        _SkipFigureCapture before the import runs.
        """
        loaded = []
        real_import = __import__

        def watch(name, *a, **k):
            if name == "matplotlib.pyplot":
                loaded.append(name)
            return real_import(name, *a, **k)

        monkeypatch.delitem(sys.modules, "matplotlib.pyplot", raising=False)
        monkeypatch.setattr("builtins.__import__", watch)

        worker = PipelineWorker(lambda settings: None, {},
                                journal=False, capture_figures=False)
        _run_worker(worker)

        assert loaded == [], (
            "a read-only job imported matplotlib.pyplot anyway")

    def test_the_run_still_completes_without_capture(self):
        """Declining the capture must not decline the work."""
        seen = []
        worker = PipelineWorker(lambda settings: seen.append(settings),
                                {"a": 1}, journal=False,
                                capture_figures=False)
        _run_worker(worker)
        assert seen == [{"a": 1}], "the pipeline function never ran"

    def test_a_capturing_run_also_completes(self):
        """The other side, so the opt-out is visibly an opt-out."""
        seen = []
        worker = PipelineWorker(lambda settings: seen.append(settings),
                                {"b": 2}, journal=False,
                                capture_figures=True)
        _run_worker(worker)
        assert seen == [{"b": 2}]

    def test_a_cancelled_run_does_no_setup_at_all(self):
        """Stop can be clicked in the tick right after Run.

        Nothing is imported, hashed or journalled for work that never
        started -- the slot acknowledges and lets the thread retire.
        """
        worker = PipelineWorker(lambda settings: pytest.fail(
            "a cancelled run executed its pipeline"), {}, journal=False,
            capture_figures=False)
        worker.cancel_token.cancel()
        _run_worker(worker)
        assert worker.was_cancelled is True
