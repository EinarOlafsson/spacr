"""Figure queue helpers: rasterising a PDF page, the print child, text and colours.

``render_pdf_to_image`` runs on a worker and must return a QImage scaled to
the long edge or ``None`` -- never raise -- for a missing, empty, unloadable,
pageless or non-PDF file, a build without QtPdf, a renderer that raises, never
answers or delivers a null page.  The print-render child says ready, answers
each request with WROTE or REFUSED, and ends quietly when the pipe closes.
``figure_text_items`` finds every text a figure carries (duck-typed legends
included) and ``_style_figure_colors`` paints backgrounds, spines and text,
or leaves a figure it cannot restyle untouched.

One general test per behaviour family, each walking a case table with pytest's
``subtests`` fixture.  QtPdf is stubbed for the gates that need a scripted
document, and the real library is used where it loads.
"""
from __future__ import annotations

import builtins
import pickle
import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QObject, QSizeF, QTimer, Signal  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402

from spacr.qt import preferences  # noqa: E402
from spacr.qt.widgets import figure_queue as fq  # noqa: E402

pytestmark = pytest.mark.qt

_NO_ANSWER = object()


def _an_image(width=8, height=6):
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor("#3366aa"))
    return image


def _install_stub_qtpdf(monkeypatch, *, load_ok=True, pages=1, answer=_NO_ANSWER):
    """A scripted ``PySide6.QtPdf``: one A4 page, a renderer that answers ``answer``."""
    import PySide6

    log = types.SimpleNamespace(loads=[], counted=0, measured=[], renderers=[])

    class _Error:
        None_ = 0
        FileNotFound = 1

    class _Doc:
        Error = _Error

        def load(self, path):
            log.loads.append(path)
            return _Error.None_ if load_ok else _Error.FileNotFound

        def pageCount(self):  # noqa: N802 - Qt name
            log.counted += 1
            return pages

        def pagePointSize(self, index):  # noqa: N802 - Qt name
            log.measured.append(index)
            return QSizeF(612.0, 792.0)

    class _Renderer(QObject):
        pageRendered = Signal(int, object, object, object, int)

        class RenderMode:
            SingleThreaded = 0
            MultiThreaded = 1

        def __init__(self):
            super().__init__()
            self.mode, self.document, self.requests = None, None, []
            log.renderers.append(self)

        def setRenderMode(self, mode):  # noqa: N802 - Qt name
            self.mode = mode

        def setDocument(self, document):  # noqa: N802 - Qt name
            self.document = document

        def requestPage(self, page, size):  # noqa: N802 - Qt name
            self.requests.append((page, (size.width(), size.height())))
            if answer is _NO_ANSWER:
                return
            QTimer.singleShot(0, lambda: self.pageRendered.emit(page, size, answer, None, 7))

    module = types.ModuleType("PySide6.QtPdf")
    module.QPdfDocument = _Doc
    module.QPdfPageRenderer = _Renderer
    monkeypatch.setitem(sys.modules, "PySide6.QtPdf", module)
    monkeypatch.setattr(PySide6, "QtPdf", module, raising=False)
    return log


def test_render_pdf_to_image_returns_a_scaled_page_or_none(qapp, monkeypatch, tmp_path, subtests):
    """Every gate refuses with None before rendering; an answered page comes back scaled to the long edge."""
    page = tmp_path / "figure.pdf"
    page.write_bytes(b"%PDF-1.4\n")
    missing = tmp_path / "gone.pdf"

    with subtests.test(msg="missing_file_never_loaded"):
        log = _install_stub_qtpdf(monkeypatch)
        assert fq.render_pdf_to_image(str(missing), timeout_ms=100) is None
        assert log.loads == []
    with subtests.test(msg="unloadable_document_never_counted"):
        log = _install_stub_qtpdf(monkeypatch, load_ok=False)
        assert fq.render_pdf_to_image(str(page), timeout_ms=100) is None
        assert (log.loads, log.counted) == ([str(page)], 0)
    with subtests.test(msg="no_pages_never_measured_or_rendered"):
        log = _install_stub_qtpdf(monkeypatch, pages=0)
        assert fq.render_pdf_to_image(str(page), timeout_ms=100) is None
        assert (log.counted, log.measured, log.renderers) == (1, [], [])
    with subtests.test(msg="answered_page_scaled_to_long_edge"):
        answer = _an_image(20, 30)
        log = _install_stub_qtpdf(monkeypatch, answer=answer)
        assert fq.render_pdf_to_image(str(page), max_px=64, timeout_ms=5000) is answer
        (renderer,) = log.renderers
        assert renderer.mode == 1
        assert renderer.document is not None
        assert renderer.requests == [(0, (49, 64))]
        assert log.measured == [0]
    with subtests.test(msg="silent_renderer_ended_by_guard"):
        log = _install_stub_qtpdf(monkeypatch)
        assert fq.render_pdf_to_image(str(page), max_px=64, timeout_ms=40) is None
        assert log.renderers[0].requests == [(0, (49, 64))]
    with subtests.test(msg="null_page_is_no_crisp_version"):
        _install_stub_qtpdf(monkeypatch, answer=QImage())
        assert fq.render_pdf_to_image(str(page), max_px=64, timeout_ms=5000) is None
    monkeypatch.undo()

    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    rubbish = tmp_path / "rubbish.pdf"
    rubbish.write_bytes(b"this is not a pdf at all\n")
    for case_id, path in (("real_missing", missing), ("real_empty", empty),
                          ("real_not_a_pdf", rubbish)):
        with subtests.test(msg=case_id):
            assert fq.render_pdf_to_image(str(path), timeout_ms=500) is None
    with subtests.test(msg="build_without_qtpdf"):
        real_import = builtins.__import__

        def refuse(name, g=None, l=None, fromlist=(), level=0):  # noqa: E741
            if "QtPdf" in name or "QPdfDocument" in (fromlist or ()):
                raise ImportError("this PySide6 build has no QtPdf")
            return real_import(name, g, l, fromlist, level)

        monkeypatch.delitem(sys.modules, "PySide6.QtPdf", raising=False)
        monkeypatch.setattr(builtins, "__import__", refuse)
        assert fq.render_pdf_to_image(str(page)) is None
        monkeypatch.undo()
    with subtests.test(msg="renderer_that_raises"):
        qtpdf = pytest.importorskip("PySide6.QtPdf")

        class _Hostile:
            def __init__(self, *a, **k):
                raise RuntimeError("no rendering backend on this machine")

        monkeypatch.setattr(qtpdf, "QPdfPageRenderer", _Hostile)
        assert fq.render_pdf_to_image(str(page)) is None
        monkeypatch.undo()


class _Scripted:
    """The child's end of the pipe, answering from a script."""

    def __init__(self, requests, fail_send_at=None):
        self.requests = list(requests)
        self.sent = []
        self.fail_send_at = fail_send_at

    def send(self, message):
        if self.fail_send_at is not None and len(self.sent) == self.fail_send_at:
            raise OSError("closed")
        self.sent.append(message)

    def recv(self):
        if not self.requests:
            raise EOFError
        return self.requests.pop(0)


def _figure_blob():
    figure = Figure(figsize=(2, 2))
    figure.add_subplot(111).plot([0, 1], [0, 1])
    return pickle.dumps(figure)


def test_print_render_child_answers_each_request_until_the_pipe_closes(monkeypatch, tmp_path, subtests):
    """The child says ready, answers WROTE/REFUSED per request, and ends when it cannot send."""
    good, bad = tmp_path / "good.png", tmp_path / "bad.png"
    cases = {
        "ready_then_each_answer": (
            [(_figure_blob(), str(good), False), (b"not a pickle", str(bad), False), None],
            None, True,
            [fq._PRINT_PROCESS_READY, fq._PRINT_PROCESS_WROTE, fq._PRINT_PROCESS_REFUSED],
            [], True),
        "cannot_say_ready": ([None], 0, False, [], [None], False),
        "parent_left_after_render": (
            [(_figure_blob(), str(tmp_path / "a.png"), False), None], 1, False,
            [fq._PRINT_PROCESS_READY], [None], False),
    }
    for case_id, (requests, fail_at, safe, sent, left, safe_after) in cases.items():
        with subtests.test(msg=case_id):
            monkeypatch.setattr(preferences, "_SAFE_MODE", False)
            connection = _Scripted(requests, fail_send_at=fail_at)
            fq._serve_print_renders(connection, safe)
            assert connection.sent == sent
            assert connection.requests == left
            assert preferences._SAFE_MODE is safe_after
    assert good.stat().st_size > 0 and not bad.exists()
    assert (tmp_path / "a.png").stat().st_size > 0


def _figure():
    figure = Figure(figsize=(4.0, 3.0))
    axes = figure.add_subplot(111)
    axes.plot([0, 1, 2], [1.0, 3.0, 2.0], label="series")
    axes.set_title("a title")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.legend(title="legend title")
    axes.text(0.5, 0.5, "annotation")
    figure.suptitle("figure title")
    return figure


class _Legend:
    def __init__(self, texts, title):
        self._texts, self._title = texts, title

    def get_texts(self):
        return list(self._texts)

    def get_title(self):
        return self._title


class _Ax:
    def __init__(self, legend):
        self.title, self.texts = "title", []
        self.xaxis = types.SimpleNamespace(label="x")
        self.yaxis = types.SimpleNamespace(label="y")
        self._legend = legend

    def get_xticklabels(self):
        return []

    def get_yticklabels(self):
        return []

    def get_legend(self):
        return self._legend


def test_figure_text_items_finds_every_text_in_traversal_order(subtests):
    """Titles, labels, ticks, free text, legend entries and titles, and figure text, in order."""
    with subtests.test(msg="real_figure"):
        figure = _figure()
        axes = figure.axes[0]
        ticks = [t.get_text() for t in list(axes.get_xticklabels()) + list(axes.get_yticklabels())]
        assert [t.get_text() for t in fq.figure_text_items(figure)] == (
            ["a title", "x", "y"] + ticks
            + ["annotation", "series", "legend title", "figure title"])
    with subtests.test(msg="no_axes"):
        assert fq.figure_text_items(Figure()) == []
    with subtests.test(msg="legend_without_title_object"):
        figure = types.SimpleNamespace(
            axes=[_Ax(_Legend(["a"], "LEGEND TITLE")), _Ax(_Legend(["b"], None))],
            texts=["suptitle"])
        assert fq.figure_text_items(figure) == [
            "title", "x", "y", "a", "LEGEND TITLE", "title", "x", "y", "b", "suptitle"]


def test_style_figure_colors_paints_ground_lines_and_text_or_leaves_it_alone(subtests):
    """Spines take the line colour (or the text colour when blank), text the fg and size; a broken figure is untouched."""
    cases = {
        "ink_and_text_differ": (("#101010", "#e0e0e0", 13, "#ff8800"),
                                (1.0, 0x88 / 255, 0.0), "#e0e0e0", 13, (0x10 / 255,) * 3),
        "blank_line_reuses_text": (("#ffffff", "#000000", 0, "   "),
                                   (0.0, 0.0, 0.0), "#000000", None, (1.0, 1.0, 1.0)),
        "red_line": (("#ffffff", "#000000", 12, "#ff0000"),
                     (1.0, 0.0, 0.0), "#000000", 12, (1.0, 1.0, 1.0)),
    }
    for case_id, (args, spine, text, size, ground) in cases.items():
        with subtests.test(msg=case_id):
            figure = _figure()
            title_size = figure.axes[0].title.get_fontsize()
            fq._style_figure_colors(figure, *args)
            axes = figure.axes[0]
            assert axes.spines["left"].get_edgecolor()[:3] == pytest.approx(spine)
            assert axes.title.get_color() == text
            assert axes.title.get_fontsize() == (size or title_size)
            assert figure.patch.get_facecolor()[:3] == pytest.approx(ground)
    with subtests.test(msg="unrestylable_left_alone"):
        asked = []

        class Awkward:
            patch = None

            def get_axes(self):
                asked.append(True)
                return []

        fq._style_figure_colors(Awkward(), "#ffffff", "#000000", 12, "#000000")
        assert asked == []
