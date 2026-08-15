"""Does ``Preferences → Figure format → PDF (vector, editable)`` produce one?

The preference has existed for a while and the honest answer, before these
tests, was "half of it". A ``.pdf`` *was* written beside every display raster,
and it was a real PDF. But it made two of the three promises in its own label
untrue, and it kept a third failure completely quiet:

* **"editable"** — matplotlib's default ``pdf.fonttype`` is 3, which draws
  each glyph as a private content stream. Illustrator and Inkscape open that
  as unselectable outlines, so the one thing a user opens a vector figure to
  do — retype a label — was the one thing they could not do.
* **the DPI preference never reached it** — the PDF was saved with no ``dpi``
  at all, so every ``imshow`` panel inside it (cell montages, mask overlays,
  plate heatmaps: most of what spaCR draws) was embedded at matplotlib's
  default 100 DPI while the user had chosen 300 or 600.
* **a failed export was invisible** — the save sat inside a bare
  ``except Exception: pass`` and the caller returned ``True`` regardless. A
  PDF that could not be written was indistinguishable, from every vantage
  point in the app, from one that had been.

So the tests here do not stop at "a file appeared". They parse the bytes with
nothing but :mod:`struct` and :mod:`zlib` — no ``pypdf``, no ``PyMuPDF`` — and
ask whether the page is *actually vector*: whether its content streams contain
path and text operators rather than one page-sized image XObject. An assertion
like that is worthless if it cannot fail, so
:func:`test_the_detector_calls_a_rasterised_figure_a_bitmap` runs the same
detector over deliberately rasterised versions of the same figure and requires
the opposite verdict.

The figures come from :class:`spacr.plot.spacrGraph` — a real spaCR plotting
call over a small synthetic frame — rather than a hand-built one-liner, so what
is measured is the path a pipeline actually takes.

Two more things these tests pin down, both about resolution:

* the display raster is written at a **capped** DPI
  (``min(dpi, 4000 / longest_side_inches)``), which is why "300 dpi" gives a
  250-DPI PNG for the 16x12" figures spaCR routinely produces, and why the
  600 and 1200 options are unreachable for anything at spaCR's minimum
  10-inch canvas. That cap is defensible for a screen and is asserted rather
  than worked around;
* the **PDF is not a screen**, so it is written at the full requested DPI. The
  cap stopping at the raster is the whole point, and it has a test.

Everything is read out of the file itself: the PNG's real DPI comes from its
``pHYs`` chunk, not from what was passed to ``savefig``.
"""
from __future__ import annotations

import logging
import re
import struct
import zlib
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("seaborn")
pytest.importorskip("statsmodels")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER_NAME = "spacr.qt.figure_queue"


# ---------------------------------------------------------------------------
# preference isolation
# ---------------------------------------------------------------------------

@pytest.fixture
def prefs(monkeypatch, tmp_path):
    """The real preference module, writing to a throwaway ini file.

    The setters under test are the real ones — that is the point of the item —
    and the real ones write to ``QSettings("spacr", "qt")``, which on this
    machine is the developer's own configuration. Rather than saving and
    restoring individual keys (and hoping the list stays complete), the single
    accessor every getter and setter funnels through is redirected at a file
    under ``tmp_path``. Nothing outside it can be reached, so nothing outside
    it needs restoring, and ``monkeypatch`` undoes the redirect either way.
    """
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences as preferences_module

    store = tmp_path / "prefs.ini"
    monkeypatch.setattr(
        preferences_module, "_settings",
        lambda: QSettings(str(store), QSettings.Format.IniFormat))
    # Pin the figure colours. Left on "auto" they follow the desktop theme,
    # which would make every byte-level assertion below depend on whether the
    # machine running the suite is in dark mode.
    preferences_module.set_figure_colors("#ffffff", "#000000")
    preferences_module.set_figure_text_size(0)
    assert preferences_module.get_figure_colors() == ("#ffffff", "#000000")
    return preferences_module


# ---------------------------------------------------------------------------
# a figure from a real spaCR plotting call
# ---------------------------------------------------------------------------

def _deterministic_values(n, shift):
    """Normal quantiles — a fixed sample, so the plot is byte-stable."""
    from scipy.stats import norm
    return norm.ppf(np.linspace(0.5 / n, 1 - 0.5 / n, n)) + shift


def _synthetic_frame(n=20):
    rows = []
    for group_index, group in enumerate(("ctrl", "treat")):
        for i, value in enumerate(_deterministic_values(n, 10.0 + 2 * group_index)):
            rows.append({"grp": group, "v1": float(value),
                         "prc": f"plate{1 + i % 2}_r{1 + i % 3}_c{1 + i % 4}"})
    return pd.DataFrame(rows)


@pytest.fixture
def spacr_figure():
    """A Figure built by :class:`spacr.plot.spacrGraph`, closed afterwards.

    A real plotting call rather than ``plt.plot([1, 2, 3])``: it brings axes,
    spines, tick labels, a title, bar patches and error bars, which is what
    makes "is this vector?" a question with a meaningful answer. Its canvas is
    10x10 inches — ``spacrGraph._standerdize_figure_format`` enforces a square
    of at least ten inches — which matters for the DPI tests below, because
    that is exactly the size at which the display cap starts to bite.
    """
    from spacr.plot import spacrGraph

    graph = spacrGraph(_synthetic_frame(), "grp", "v1", graph_type="bar")
    graph.create_plot()
    figure = graph.get_figure()
    assert figure is not None
    yield figure
    plt.close(figure)


@pytest.fixture
def rasterised_twin(spacr_figure):
    """The same figure with every artist flattened to a bitmap.

    The negative control for the vector detector. Without it, an assertion
    that "the PDF contains drawing operators" proves nothing — a PDF wrapping
    a single page-sized image contains a few too.
    """
    for axes in spacr_figure.get_axes():
        axes.set_rasterized(True)
        axes.set_rasterization_zorder(np.inf)
    return spacr_figure


@pytest.fixture
def image_only_figure():
    """A second control: a figure that is nothing *but* a bitmap."""
    figure, axes = plt.subplots(figsize=(4, 3))
    axes.imshow(np.random.default_rng(0).random((200, 200)))
    axes.set_axis_off()
    figure.subplots_adjust(0, 0, 1, 1)
    yield figure
    plt.close(figure)


# ---------------------------------------------------------------------------
# PDF parsing — stdlib only
# ---------------------------------------------------------------------------

_STREAM = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.S)

#: Content-stream operators that draw. ``re`` is a rectangle, ``m``/``l``/``c``
#: build a path, ``S``/``f`` stroke and fill it, ``cm`` sets a transform.
#: Matched with their trailing newline because matplotlib writes one operator
#: per line and a bare ``b" m"`` also matches the middle of a number.
_DRAW_OPS = (b" re\n", b" m\n", b" l\n", b" c\n", b" S\n", b" f\n", b" cm\n")
#: Text operators. ``BT`` opens a text object, ``Tj``/``TJ`` show a string.
#: Newline-anchored for the same reason, and it is not theoretical: an
#: inflated image stream is random bytes, and the loose token ``b"Tj"`` scores
#: 18 hits inside the 200x200 noise bitmap used as a control below.
_TEXT_OPS = (b"BT\n", b" Tj\n", b" TJ\n")


def _content_streams(data: bytes) -> bytes:
    """Every stream object in ``data``, flate-decompressed where it can be.

    matplotlib compresses content streams by default and leaves the odd one
    (and every image) raw, so both cases are handled and concatenated.
    """
    pieces = []
    for match in _STREAM.finditer(data):
        raw = match.group(1)
        try:
            pieces.append(zlib.decompress(raw))
        except zlib.error:
            pieces.append(raw)
    return b"\n".join(pieces)


def _parse_xref(data: bytes) -> dict:
    """Follow ``startxref`` to the cross-reference table and validate it.

    Deliberately strict: it is easy to write a file that begins ``%PDF-1.4``
    and ends ``%%EOF`` and is garbage in between, and a test that only checked
    those two would pass on a truncated save — exactly the failure the old
    silent ``except`` used to produce.
    """
    marker = data.rfind(b"startxref")
    assert marker != -1, "no startxref"
    offset = int(data[marker + len(b"startxref"):].split()[0])
    assert 0 < offset < len(data), f"startxref points outside the file: {offset}"
    assert data[offset:offset + 4] == b"xref", "startxref does not point at an xref"

    # `xref\n` then a subsection header `<first> <count>\n`, then `count`
    # fixed-width 20-byte entries: `nnnnnnnnnn ggggg n \n`.
    header_start = data.index(b"\n", offset + 4) + 1
    table_start = data.index(b"\n", header_start) + 1
    first, count = (int(token) for token in data[header_start:table_start].split())
    assert first == 0, f"unexpected first object number {first}"

    entries = []
    for i in range(count):
        entry = data[table_start + i * 20:table_start + i * 20 + 20]
        assert len(entry) == 20, "truncated xref entry"
        position, _generation, kind = entry.split()[:3]
        entries.append((int(position), kind))

    in_use = [position for position, kind in entries if kind == b"n"]
    for position in in_use:
        assert data[position:position + 40].split()[2:3] == [b"obj"], (
            f"xref entry points at {data[position:position + 20]!r}, not an object")

    trailer = data.rfind(b"trailer")
    assert trailer != -1, "no trailer"
    assert b"/Root" in data[trailer:], "trailer names no document catalogue"
    return {"objects": count, "in_use": len(in_use)}


def _pdf_report(path) -> dict:
    """Everything the assertions below need, read straight out of the bytes."""
    data = Path(path).read_bytes()
    streams = _content_streams(data)
    return {
        "size": len(data),
        "header_ok": data.startswith(b"%PDF-1."),
        "eof_ok": data.rstrip().endswith(b"%%EOF"),
        "xref": _parse_xref(data),
        "draw_ops": sum(streams.count(op) for op in _DRAW_OPS),
        "text_ops": sum(streams.count(op) for op in _TEXT_OPS),
        "images": data.count(b"/Subtype /Image") + data.count(b"/Subtype/Image"),
        "truetype_fonts": data.count(b"/FontFile2"),
        "type3_fonts": data.count(b"/Type3"),
        "raster_widths": [int(w) for w in re.findall(rb"/Width (\d+)", data)],
    }


#: A page below this many drawing operators is not drawing a figure.
_VECTOR_FLOOR = 20


def _verdict(report: dict) -> str:
    """``"vector"`` or ``"bitmap"`` for one PDF.

    Presence of an image XObject is not on its own disqualifying — a perfectly
    good spaCR figure has an ``imshow`` panel inside a vector frame — so the
    test is on the *operators*. A page whose content is one big picture has a
    handful (place the image, close the page); a drawn figure has hundreds.
    """
    if report["draw_ops"] >= _VECTOR_FLOOR:
        return "vector"
    return "bitmap" if report["images"] >= 1 else "empty"


# ---------------------------------------------------------------------------
# PNG parsing — stdlib only
# ---------------------------------------------------------------------------

def _png_chunks(path) -> dict:
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    chunks, position = {}, 8
    while position < len(data):
        (length,) = struct.unpack(">I", data[position:position + 4])
        kind = data[position + 4:position + 8]
        chunks.setdefault(kind, data[position + 8:position + 8 + length])
        position += 12 + length
        if kind == b"IEND":
            break
    return chunks


def _png_size(path) -> tuple:
    """``(width, height)`` from the IHDR chunk — bytes 16..24 of the file."""
    width, height = struct.unpack(">II", Path(path).read_bytes()[16:24])
    return width, height


def _png_dpi(path) -> int:
    """The DPI the PNG was actually written at, from its ``pHYs`` chunk.

    Reading this rather than trusting the argument passed to ``savefig`` is
    what makes the resolution tests non-circular: ``pHYs`` records pixels per
    metre, which matplotlib fills in from the ``dpi`` it really used.
    """
    physical = _png_chunks(path).get(b"pHYs")
    assert physical is not None, "no pHYs chunk: the PNG records no resolution"
    pixels_per_metre_x, pixels_per_metre_y, unit = struct.unpack(">IIB", physical)
    assert unit == 1, "pHYs is not in metres"
    assert pixels_per_metre_x == pixels_per_metre_y
    return round(pixels_per_metre_x * 0.0254)


# ---------------------------------------------------------------------------
# the PDF actually gets written, and it is a PDF
# ---------------------------------------------------------------------------

def test_pdf_mode_writes_a_valid_pdf_beside_the_png(prefs, spacr_figure, tmp_path):
    from spacr.qt.widgets.figure_queue import render_figure_to_png, _sibling_pdf

    prefs.set_figure_format("pdf")
    prefs.set_figure_png_dpi(300)
    assert prefs.get_figure_format() == "pdf"

    png = tmp_path / "fig_00000.png"
    assert render_figure_to_png(spacr_figure, str(png)) is True
    assert png.is_file()

    pdf = _sibling_pdf(png)
    assert pdf.is_file(), "PDF mode produced no vector page"

    report = _pdf_report(pdf)
    assert report["header_ok"], "no %PDF-1.x header"
    assert report["eof_ok"], "no %%EOF trailer"
    assert report["xref"]["in_use"] >= 5, report["xref"]
    assert report["size"] > 4000, (
        f"the page is only {report['size']} bytes — nothing was drawn on it")


def test_png_mode_writes_no_pdf(prefs, spacr_figure, tmp_path):
    """The preference is a switch, not a decoration."""
    from spacr.qt.widgets.figure_queue import render_figure_to_png, _sibling_pdf

    prefs.set_figure_format("png")
    png = tmp_path / "fig_00000.png"
    assert render_figure_to_png(spacr_figure, str(png)) is True
    assert png.is_file()
    assert not _sibling_pdf(png).is_file()


# ---------------------------------------------------------------------------
# and it is vector, not a bitmap in a wrapper
# ---------------------------------------------------------------------------

def test_the_pdf_is_vector_not_a_bitmap(prefs, spacr_figure, tmp_path):
    from spacr.qt.widgets.figure_queue import render_figure_to_png, _sibling_pdf

    prefs.set_figure_format("pdf")
    prefs.set_figure_png_dpi(300)
    png = tmp_path / "fig_00000.png"
    render_figure_to_png(spacr_figure, str(png))

    report = _pdf_report(_sibling_pdf(png))
    assert _verdict(report) == "vector", report
    # Measured: 68 for this bar chart (axes, spines, ticks, bars, error bars)
    # against 4 for the flattened controls in the next test.
    assert report["draw_ops"] > 40, (
        f"only {report['draw_ops']} drawing operators for a bar chart with "
        "axes, spines, ticks and error bars")
    assert report["text_ops"] > 0, "no text operators: the labels are not text"
    assert report["images"] == 0, (
        f"{report['images']} image XObject(s) in a figure that draws none — "
        "the page has been rasterised")


def test_the_detector_calls_a_rasterised_figure_a_bitmap(
        prefs, rasterised_twin, image_only_figure, tmp_path):
    """The negative control, without which the test above proves nothing.

    Both controls go through :func:`_export_vector_pdf` — the same writer, the
    same settings — so the only thing that differs is the content of the
    figure. If the detector could not tell these apart it would be measuring
    the writer, not the page.
    """
    from spacr.qt.widgets.figure_queue import _export_vector_pdf

    flattened = tmp_path / "flattened.pdf"
    assert _export_vector_pdf(rasterised_twin, flattened, 300, "#ffffff")
    flattened_report = _pdf_report(flattened)

    picture = tmp_path / "picture.pdf"
    assert _export_vector_pdf(image_only_figure, picture, 300, "#ffffff")
    picture_report = _pdf_report(picture)

    for name, report in (("set_rasterized", flattened_report),
                         ("imshow-only", picture_report)):
        assert report["header_ok"] and report["eof_ok"], name
        assert report["images"] >= 1, (name, report)
        assert report["text_ops"] == 0, (
            f"{name}: text operators found on a page with no live text — the "
            f"detector is matching noise inside the image stream. {report}")
        assert _verdict(report) == "bitmap", (
            f"{name}: the detector called a flattened page "
            f"{_verdict(report)!r} — it cannot tell vector from raster, so "
            f"the positive assertion is meaningless. {report}")


def test_text_is_embedded_as_truetype_so_it_stays_editable(
        prefs, spacr_figure, tmp_path):
    """"PDF (vector, editable)" — the second word used to be false.

    matplotlib defaults to ``pdf.fonttype`` 3, which emits every glyph as a
    Type 3 charproc. The file is vector either way; the difference is that
    Illustrator and Inkscape can select and retype text backed by an embedded
    TrueType face (``/FontFile2``) and cannot when it is Type 3 outlines.
    """
    from spacr.qt.widgets.figure_queue import render_figure_to_png, _sibling_pdf

    prefs.set_figure_format("pdf")
    png = tmp_path / "fig_00000.png"
    render_figure_to_png(spacr_figure, str(png))

    report = _pdf_report(_sibling_pdf(png))
    assert report["truetype_fonts"] >= 1, (
        "no /FontFile2: the text is not backed by an embedded TrueType face "
        "and will open as outlines")
    assert report["type3_fonts"] == 0, (
        "the page still carries Type 3 fonts — pdf.fonttype was not raised "
        "to 42 for this save")


def test_the_font_setting_does_not_leak_out_of_the_save(prefs, spacr_figure,
                                                        tmp_path):
    """``rc_context``, not a global assignment.

    A pipeline may have chosen ``pdf.fonttype`` for its own ``savefig`` calls,
    and rendering a preview for the Figures panel must not change what it then
    writes to the user's results directory.
    """
    from spacr.qt.widgets.figure_queue import render_figure_to_png

    before = matplotlib.rcParams["pdf.fonttype"]
    prefs.set_figure_format("pdf")
    render_figure_to_png(spacr_figure, str(tmp_path / "fig.png"))
    assert matplotlib.rcParams["pdf.fonttype"] == before


# ---------------------------------------------------------------------------
# resolution: what the DPI preference does and does not reach
# ---------------------------------------------------------------------------

def test_png_records_the_requested_dpi_when_the_cap_does_not_bite(
        prefs, spacr_figure, tmp_path):
    """300 DPI on spaCR's 10-inch canvas really is 300 DPI.

    Two independent readings, because either alone is weak. ``pHYs`` says what
    resolution the file claims; the pixel count says whether the claim was
    honoured. The pixel count is *not* ``10 x 300`` and that is not a bug:
    ``render_figure_to_png`` saves with ``bbox_inches="tight"``, which crops
    the canvas to its drawn extent plus a 0.1" pad before scaling. So the
    expectation asserted here is the tight box at 300 DPI, computed from
    matplotlib's own measurement of the figure rather than from the nominal
    canvas.
    """
    from spacr.qt.widgets.figure_queue import render_figure_to_png

    prefs.set_figure_format("png")
    prefs.set_figure_png_dpi(300)
    png = tmp_path / "fig.png"
    render_figure_to_png(spacr_figure, str(png))

    assert _png_dpi(png) == 300

    width, height = _png_size(png)
    assert tuple(spacr_figure.get_size_inches()) == (10.0, 10.0)
    assert (width, height) != (3000, 3000), (
        "the tight bounding box was not applied")

    spacr_figure.canvas.draw()
    tight = spacr_figure.get_tightbbox(spacr_figure.canvas.get_renderer())
    pad = matplotlib.rcParams["savefig.pad_inches"]
    expected = ((tight.width + 2 * pad) * 300, (tight.height + 2 * pad) * 300)
    # Older supported matplotlib releases round text extents a few pixels
    # differently from the savefig crop. Ten pixels is 0.033 inch at 300 DPI,
    # while still catching a missing tight box by hundreds of pixels.
    assert abs(width - expected[0]) <= 10 and abs(height - expected[1]) <= 10, (
        f"{width}x{height} px is not the tight box at 300 dpi "
        f"({expected[0]:.0f}x{expected[1]:.0f})")


def test_a_large_figure_is_displayed_below_the_requested_dpi(
        prefs, spacr_figure, tmp_path):
    """"300 dpi" silently means 250 for a 16x12" figure — on purpose.

    ``render_figure_to_png`` caps the *display* raster at
    ``min(dpi, 4000 / longest_side_inches)``: a 16x12" figure at a true 300 DPI
    is a 4800 px PNG that costs more to decode than any screen can show. The
    cap is asserted rather than hidden because it is a real gap between the
    label and the file, and because the sibling PDF is where it stops mattering
    — see the test below.

    The cap reaches further than 16 inches, too: at 600 DPI it engages for
    anything over 6.7", and at 1200 for anything over 3.3", so those two
    entries in the preference are unreachable for any figure spaCR draws.
    """
    from spacr.qt.widgets.figure_queue import render_figure_to_png

    prefs.set_figure_format("png")
    prefs.set_figure_png_dpi(300)
    spacr_figure.set_size_inches(16, 12)
    png = tmp_path / "big.png"
    render_figure_to_png(spacr_figure, str(png))

    assert prefs.get_figure_png_dpi() == 300
    assert _png_dpi(png) == 250, (
        "the display cap changed; this test records what it currently does")

    prefs.set_figure_png_dpi(600)
    tall = tmp_path / "tall.png"
    spacr_figure.set_size_inches(10, 10)
    render_figure_to_png(spacr_figure, str(tall))
    assert _png_dpi(tall) == 400, "600 dpi is unreachable at spaCR's 10\" canvas"


def test_the_display_cap_does_not_reach_the_pdf(prefs, tmp_path):
    """The export is written at the DPI the user asked for.

    A PDF page is resolution-independent, so this is invisible for line art —
    and decisive for the ``imshow`` panels that make up most spaCR figures.
    Before the fix the PDF was saved with no ``dpi`` at all, which meant
    matplotlib's own 100, so a "300 dpi, vector" export was a vector frame
    around a 100-DPI bitmap. Here the same figure is exported at three
    settings and the embedded raster has to grow with each of them, including
    past the 250 the display raster is capped to.
    """
    from spacr.qt.widgets.figure_queue import render_figure_to_png, _sibling_pdf

    prefs.set_figure_format("pdf")

    widths = {}
    for dpi in (100, 300, 600):
        figure, axes = plt.subplots(figsize=(16, 12))
        axes.imshow(np.random.default_rng(0).random((64, 64)))
        axes.set_title("montage")
        prefs.set_figure_png_dpi(dpi)
        png = tmp_path / f"montage_{dpi}.png"
        render_figure_to_png(figure, str(png))
        report = _pdf_report(_sibling_pdf(png))
        assert report["raster_widths"], f"no embedded raster at {dpi} dpi"
        widths[dpi] = max(report["raster_widths"])
        plt.close(figure)

    assert widths[100] < widths[300] < widths[600], widths
    # Ratios, not absolutes: the tight bbox decides how many inches the image
    # occupies, the DPI decides how many pixels each inch becomes.
    assert 2.8 < widths[300] / widths[100] < 3.2, widths
    assert 1.9 < widths[600] / widths[300] < 2.1, widths
    # And the on-screen raster for that same 16x12" figure was capped at 250,
    # so the export is strictly better than what the display could show.
    assert _png_dpi(tmp_path / "montage_600.png") == 250


# ---------------------------------------------------------------------------
# a failed export is no longer silent
# ---------------------------------------------------------------------------

class _ExplodingFigure:
    """Stands in for a Figure whose PDF save fails — a full disk, a read-only
    results directory, a backend that chokes on one artist. All of them used
    to be swallowed identically."""

    def savefig(self, *args, **kwargs):
        raise RuntimeError("no space left on device")


def test_a_failed_pdf_export_is_logged(prefs, tmp_path, caplog):
    from spacr.qt.widgets.figure_queue import _export_vector_pdf

    target = tmp_path / "doomed.pdf"
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert _export_vector_pdf(_ExplodingFigure(), target, 300, "#ffffff") is False

    assert any("no space left on device" in record.getMessage()
               for record in caplog.records), (
        "the export failed without a word — the old bare `except: pass`")
    assert not target.exists()


def test_a_failed_export_removes_the_page_left_from_the_previous_render(
        prefs, spacr_figure, tmp_path):
    """A stale page is worse than a missing one.

    The queue rasterises whatever ``.pdf`` sits beside the raster. If a
    re-render fails and the previous figure's page is left there, the user is
    shown the *wrong figure*, crisply, with nothing to indicate it.
    """
    from spacr.qt.widgets.figure_queue import _export_vector_pdf

    target = tmp_path / "slot.pdf"
    assert _export_vector_pdf(spacr_figure, target, 300, "#ffffff")
    assert target.is_file()

    assert _export_vector_pdf(_ExplodingFigure(), target, 300, "#ffffff") is False
    assert not target.exists(), "the previous figure's page survived the failure"


def test_a_failed_pdf_still_leaves_the_figure_visible(prefs, spacr_figure,
                                                      tmp_path, monkeypatch):
    """``render_figure_to_png`` reports the PNG, not the PDF.

    Returning False here would be a worse bug than the one being fixed: both
    callers turn False into "no pixmap, no thumbnail", so a figure would
    vanish from the gallery because a sibling export nothing had asked for yet
    could not be written.
    """
    import spacr.qt.widgets.figure_queue as figure_queue

    prefs.set_figure_format("pdf")
    monkeypatch.setattr(figure_queue, "_export_vector_pdf",
                        lambda *args, **kwargs: False)
    png = tmp_path / "fig.png"
    assert figure_queue.render_figure_to_png(spacr_figure, str(png)) is True
    assert png.is_file()


# ---------------------------------------------------------------------------
# the raster and its page have to agree on a name
# ---------------------------------------------------------------------------

def test_sibling_pdf_matches_the_writer_and_the_readers(tmp_path):
    """Nothing records the pairing; it is re-derived from the PNG's path."""
    from spacr.qt.widgets.figure_queue import _sibling_pdf

    assert _sibling_pdf("/tmp/fig_00007.png") == Path("/tmp/fig_00007.pdf")
    assert _sibling_pdf("/tmp/spacr_fig_1234_9.PNG") == Path("/tmp/spacr_fig_1234_9.pdf")
    # A dotted stem keeps its dots: `report.v2.png` is a normal name and
    # `with_suffix` handles it correctly.
    assert _sibling_pdf("/tmp/report.v2.png") == Path("/tmp/report.v2.pdf")


def test_sibling_pdf_cannot_collide_on_an_extensionless_name():
    """``Path.with_suffix`` alone would map two figures onto one page.

    It replaces everything after the last dot, so ``run_2.5`` and ``run_2.6``
    both become ``run_2.pdf``: two figures, one file, and whichever rendered
    last is what both slots display. Only a trailing ``.png`` is treated as an
    extension, so anything else keeps its whole name.
    """
    from spacr.qt.widgets.figure_queue import _sibling_pdf

    assert Path("/tmp/run_2.5").with_suffix(".pdf") == \
        Path("/tmp/run_2.6").with_suffix(".pdf")          # the trap
    assert _sibling_pdf("/tmp/run_2.5") != _sibling_pdf("/tmp/run_2.6")
    assert _sibling_pdf("/tmp/run_2.5") == Path("/tmp/run_2.5.pdf")
    assert _sibling_pdf("/tmp/plain") == Path("/tmp/plain.pdf")
