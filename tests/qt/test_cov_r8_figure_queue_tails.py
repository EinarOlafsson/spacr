"""figure_queue: a PDF that will not rasterise, and a section that drew nothing.

Each of these is what the queue does when a figure or a run turns out to
be less than expected. None of them may raise: the queue is what the
user is looking at while a pipeline runs, and a viewer that disappears
because one page would not draw is worse than a viewer showing a
placeholder.
"""
from __future__ import annotations

import pytest

from spacr.qt.widgets import figure_queue as fq

pytestmark = pytest.mark.qt


class TestRasterisingAPdf:

    def test_a_file_that_is_not_a_pdf_returns_nothing(self, tmp_path,
                                                      caplog):
        """THE UNCOVERED GUARD.

        The renderer runs off the GUI thread and returns an image or
        None; it must never propagate. A figure that cannot be drawn is
        reported by the caller as a placeholder, and the reason goes to
        the log rather than to the user's face.
        """
        broken = tmp_path / "not-really.pdf"
        broken.write_bytes(b"this is not a pdf at all")
        with caplog.at_level("DEBUG"):
            assert fq.render_pdf_to_image(str(broken)) is None

    def test_a_path_that_does_not_exist_returns_nothing(self, tmp_path):
        assert fq.render_pdf_to_image(str(tmp_path / "absent.pdf")) is None

    def test_an_empty_file_returns_nothing(self, tmp_path):
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")
        assert fq.render_pdf_to_image(str(empty)) is None

    def test_a_build_without_qtpdf_returns_nothing(self, tmp_path,
                                                   monkeypatch, caplog):
        """THE GUARD ITSELF, and the case it is really for.

        The three refusals above return None BEFORE the guard: a missing
        file, a document that will not load, and a document with no
        pages are each answered directly. What the try/except is left
        holding is everything else -- and the likeliest "everything
        else" is the import at the top of it, because PySide6.QtPdf is a
        separate component a build can be missing.

        It has to come back as None on the worker thread rather than
        raise: the queue is what the user is looking at while a pipeline
        runs.
        """
        import builtins

        real = builtins.__import__

        def refuse(name, g=None, l=None, fromlist=(), level=0):
            if "QtPdf" in name or "QPdfDocument" in (fromlist or ()):
                raise ImportError("this PySide6 build has no QtPdf")
            return real(name, g, l, fromlist, level)

        pdf = tmp_path / "figure.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        monkeypatch.setattr(builtins, "__import__", refuse)
        with caplog.at_level("DEBUG"):
            assert fq.render_pdf_to_image(str(pdf)) is None

    def test_a_renderer_that_raises_is_survived(self, tmp_path,
                                                monkeypatch):
        """Anything the renderer itself throws is caught the same way."""
        import PySide6.QtPdf as qtpdf

        class _Hostile:
            def __init__(self, *a, **k):
                raise RuntimeError("no rendering backend on this machine")

        pdf = tmp_path / "figure.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        monkeypatch.setattr(qtpdf, "QPdfPageRenderer", _Hostile)
        assert fq.render_pdf_to_image(str(pdf)) is None


class TestForgettingARunWithNoFigures:
    """`if count <= 0:` inside `forget_run` cannot fire.

    `forget_run` finds its span through `run_sections()`, and that
    method only emits a section when `end > start` -- a run that drew
    nothing is filtered out before `forget_run` ever sees it, so the
    lookup returns None and the earlier `span is None` arm answers
    instead.

    Both arms return 0, so the behaviour a caller sees is the same
    either way. Pinned to the filter that makes it so.
    """

    def test_run_sections_never_reports_an_empty_section(self, qtbot):
        queue = fq.FigureQueue()
        qtbot.addWidget(queue)
        queue._count = 3
        # two marks at the same start: the first section spans nothing
        queue._runs = [{"label": "empty run", "start": 0},
                       {"label": "real run", "start": 0}]

        sections = queue.run_sections()
        assert all(count > 0 for _label, _start, count in sections), (
            "an empty section is reported again; the `count <= 0` arm in "
            "forget_run is now reachable and wants a test")
        assert "empty run" not in [label for label, _s, _c in sections]

    def test_forgetting_a_run_that_drew_nothing_drops_nothing(self, qtbot):
        """It answers 0 through the `span is None` arm, not the other one."""
        queue = fq.FigureQueue()
        qtbot.addWidget(queue)
        queue._count = 3
        queue._runs = [{"label": "empty run", "start": 0},
                       {"label": "real run", "start": 0}]
        assert queue.forget_run("empty run") == 0

    def test_a_label_the_queue_never_had_drops_nothing(self, qtbot):
        queue = fq.FigureQueue()
        qtbot.addWidget(queue)
        assert queue.forget_run("no such run") == 0

    def test_an_empty_queue_has_no_sections_at_all(self, qtbot):
        queue = fq.FigureQueue()
        qtbot.addWidget(queue)
        assert queue.run_sections() == []


class TestAPageThatCameBackBlank:

    def test_a_null_image_is_remembered_as_failed(self, qtbot):
        """THE UNCOVERED ARM.

        Recording "failed" is what stops the queue asking for the same
        page again on every repaint -- and what lets it show a
        placeholder that says so, instead of an empty frame that looks
        like a figure with nothing in it.
        """
        queue = fq.FigureQueue()
        qtbot.addWidget(queue)

        idx = 0
        token = object()
        queue._pdf_state = {idx: token}
        queue._on_pdf_rendered((idx, token, None))

        assert queue._pdf_state.get(idx) == "failed"

    def test_a_stale_token_is_ignored(self, qtbot):
        """A render that finished after the page moved on must not land."""
        queue = fq.FigureQueue()
        qtbot.addWidget(queue)

        idx = 0
        queue._pdf_state = {idx: "current"}
        queue._on_pdf_rendered((idx, "stale", None))
        assert queue._pdf_state.get(idx) == "current", (
            "a stale render overwrote the state of a newer one")
