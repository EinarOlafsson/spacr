"""The figure-format and resolution preferences reach the files users keep.

They did not. ``render_figure_to_png`` honoured both, and it has exactly
two callers, both writing into a temp directory for on-screen display.
Every figure a pipeline saved into its results folder came from a
hard-coded ``savefig`` — thirteen in :mod:`spacr.plot` and around fifty
more across ``ml``, ``io``, ``measure``, ``submodules``, ``timelapse``,
``core``, ``toxo`` and ``deep_spacr`` — each with its own literal format
and its own literal DPI. Setting "PNG" in Preferences changed nothing
anyone kept; setting 600 DPI changed nothing at all.

:func:`spacr.plot.save_figure` is the one place that decision is made
now. Two things it must not undo, both already fixed once:

``pdf.fonttype = 42``
    matplotlib's default is 3, which draws every glyph as its own
    content stream — vector, but unselectable outlines in Illustrator
    and Inkscape, which is not what "PDF (vector, editable)" promises.

``dpi= is passed``
    a PDF page is resolution-independent, but ``imshow`` panels inside
    it are not. Without it 100, 300 and 600 DPI produced byte-identical
    files.

And one thing it must state rather than swallow: ``spacrGraph`` pins its
canvas to at least 10 inches square and grows it with the group count,
so the top of the DPI range is not deliverable for a large grouped
figure. :func:`spacr.plot.deliverable_dpi` says which DPI was used.
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from spacr import plot as P

#: Every module whose kept figures had to be routed.
ROUTED_MODULES = (
    "plot", "ml", "io", "measure", "submodules", "timelapse", "core",
    "toxo", "deep_spacr",
)

#: Call sites that are NOT a figure the user keeps, and must stay as they
#: are. Each is a deliberate exception with a reason.
ALLOWED_RAW_SAVEFIG = {
    # The helper itself.
    ("plot", "save_figure"),
}


def _figure(width=4.0, height=3.0, image=False):
    fig, ax = plt.subplots(figsize=(width, height))
    if image:
        import numpy as np
        ax.imshow(np.random.default_rng(0).random((32, 32)))
    else:
        ax.plot([0, 1], [0, 1])
    return fig


@pytest.fixture(autouse=True)
def _shipped_preferences(monkeypatch):
    """No Qt preference store in a bare pipeline test.

    ``figure_output_preferences`` must degrade rather than raise, and the
    tests below state the format they want explicitly where it matters.
    """
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# The format preference decides the file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["png", "pdf"])
def test_the_format_preference_decides_what_is_written(fmt, tmp_path,
                                                       monkeypatch):
    monkeypatch.setattr(P, "figure_output_preferences", lambda: (fmt, 200))
    written = P.save_figure(_figure(), str(tmp_path / "figure.pdf"))
    assert Path(written).suffix == f".{fmt}"
    assert Path(written).exists()
    head = Path(written).read_bytes()[:8]
    if fmt == "pdf":
        assert head.startswith(b"%PDF")
    else:
        assert head.startswith(b"\x89PNG")


def test_the_name_follows_the_format(tmp_path, monkeypatch):
    """A PNG written to ``figure.pdf`` is a file no viewer opens."""
    monkeypatch.setattr(P, "figure_output_preferences", lambda: ("png", 200))
    written = P.save_figure(_figure(), str(tmp_path / "hist.pdf"))
    assert written.endswith("hist.png")
    assert not (tmp_path / "hist.pdf").exists()


def test_a_dotted_stem_is_not_truncated(tmp_path, monkeypatch):
    """``plate_2.5_umap`` and ``plate_2.6_umap`` are two figures.

    Naive ``splitext`` turns both into ``plate_2`` — one file, silently
    overwritten, and the second result is the only one that survives.
    """
    monkeypatch.setattr(P, "figure_output_preferences", lambda: ("pdf", 200))
    first = P.save_figure(_figure(), str(tmp_path / "plate_2.5_umap"))
    second = P.save_figure(_figure(), str(tmp_path / "plate_2.6_umap"))
    assert first != second
    assert Path(first).exists() and Path(second).exists()


def test_an_explicit_format_still_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "figure_output_preferences", lambda: ("pdf", 200))
    written = P.save_figure(_figure(), str(tmp_path / "f.pdf"), fmt="png")
    assert written.endswith(".png")


def test_a_missing_directory_is_created(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "figure_output_preferences", lambda: ("pdf", 200))
    written = P.save_figure(_figure(), str(tmp_path / "deep" / "a" / "f.pdf"))
    assert Path(written).exists()


def test_no_preference_store_degrades_to_the_shipped_answer(monkeypatch):
    """Pipelines run from the CLI, where importing PySide6 would be absurd."""
    import builtins
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("spacr.qt") or "preferences" in name:
            raise ImportError("no Qt here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert P.figure_output_preferences() == (P.DEFAULT_FIGURE_FORMAT,
                                             P.DEFAULT_FIGURE_DPI)


# ---------------------------------------------------------------------------
# The two things already fixed, which must not be undone
# ---------------------------------------------------------------------------

def test_pdf_text_is_selectable_not_outlines(tmp_path, monkeypatch):
    """``pdf.fonttype`` 42, not matplotlib's default 3."""
    monkeypatch.setattr(P, "figure_output_preferences", lambda: ("pdf", 200))
    fig = _figure()
    fig.axes[0].set_title("selectable")
    written = Path(P.save_figure(fig, str(tmp_path / "text.pdf")))
    body = written.read_bytes()
    assert b"/TrueType" in body or b"/FontFile2" in body, (
        "the PDF holds no TrueType font, so pdf.fonttype fell back to 3 and "
        "the text is unselectable outlines")


def test_the_rc_setting_is_scoped_and_restored(tmp_path, monkeypatch):
    """A caller who deliberately chose fonttype 3 keeps it."""
    monkeypatch.setattr(P, "figure_output_preferences", lambda: ("pdf", 200))
    before = matplotlib.rcParams["pdf.fonttype"]
    P.save_figure(_figure(), str(tmp_path / "scoped.pdf"))
    assert matplotlib.rcParams["pdf.fonttype"] == before


@pytest.mark.parametrize("fmt", ["png", "pdf"])
def test_the_dpi_preference_actually_changes_the_file(fmt, tmp_path,
                                                      monkeypatch):
    """Without ``dpi=``, 100 / 300 / 600 produced identical files.

    A figure with an ``imshow`` panel, because that is the case a PDF
    page's resolution-independence does NOT cover — and spaCR figures are
    full of them.
    """
    sizes = {}
    for dpi in (100, 300):
        monkeypatch.setattr(P, "figure_output_preferences",
                            lambda dpi=dpi: (fmt, dpi))
        written = P.save_figure(_figure(image=True),
                                str(tmp_path / f"img_{dpi}.{fmt}"))
        sizes[dpi] = os.path.getsize(written)
    assert sizes[300] != sizes[100], (
        f"{fmt} at 100 and 300 DPI produced the same {sizes[100]} bytes, so "
        "dpi= is not reaching savefig")


# ---------------------------------------------------------------------------
# Say so rather than appearing to accept
# ---------------------------------------------------------------------------

def test_a_deliverable_dpi_is_returned_unchanged():
    assert P.deliverable_dpi(_figure(4, 3), 600) == 600


def test_an_undeliverable_dpi_is_reduced_and_reported(capsys):
    """The spacrGraph case: a 10-inch-plus canvas at the top of the range."""
    fig = _figure(60, 60)
    used = P.deliverable_dpi(fig, 1200, path="grouped.pdf")
    printed = capsys.readouterr().out
    assert used < 1200, "a 60-inch figure cannot be written at 1200 DPI"
    assert used >= 72
    assert "1200" in printed and str(used) in printed, (
        "the substitution has to be stated, or the setting only appears to "
        f"have been accepted. Printed: {printed!r}")
    assert "spacrGraph" in printed


def test_spacr_graph_canvases_are_the_case_this_covers():
    """``_standerdize_figure_format`` really does pin a >=10 inch canvas.

    If that floor ever moves, the warning above is describing something
    that no longer happens.
    """
    source = Path(P.__file__).read_text()
    body = source.split("def _standerdize_figure_format")[1][:2000]
    assert "fig_size = 10" in body.replace(" ", " "), (
        "the 10 inch floor is gone from _standerdize_figure_format")


def test_the_ceiling_is_never_below_a_usable_resolution():
    """Even an absurd canvas gets a real figure, not a 3 DPI thumbnail."""
    assert P.deliverable_dpi(_figure(5000, 5000), 1200) >= 72


def test_a_figure_that_cannot_report_its_size_is_not_second_guessed():
    class Odd:
        def get_size_inches(self):
            raise RuntimeError("no canvas")

    assert P.deliverable_dpi(Odd(), 600) == 600


def test_close_closes(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "figure_output_preferences", lambda: ("png", 100))
    fig = _figure()
    P.save_figure(fig, str(tmp_path / "closed.png"), close=True)
    assert not plt.fignum_exists(fig.number)


# ---------------------------------------------------------------------------
# Nothing kept still writes its own file
# ---------------------------------------------------------------------------

def _raw_savefig_sites(module_name):
    """Every ``*.savefig(...)`` call left in ``module_name``."""
    import importlib

    path = Path(importlib.import_module(f"spacr.{module_name}").__file__)
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "savefig":
            out.append(node.lineno)
    return out


@pytest.mark.parametrize("module_name", ROUTED_MODULES)
def test_every_kept_figure_goes_through_the_helper(module_name):
    """One helper, not sixty literals.

    ``spacr.plot.save_figure`` is allowed exactly one raw ``savefig`` —
    its own. Everything else in these modules has to come through it, or
    the preference reaches some figures and not others, which is harder
    to explain to a user than not working at all.
    """
    sites = _raw_savefig_sites(module_name)
    if module_name == "plot":
        assert len(sites) == 1, (
            f"spacr.plot has {len(sites)} raw savefig calls at lines "
            f"{sites}; only save_figure's own is allowed")
        return
    assert not sites, (
        f"spacr.{module_name} still writes figures directly at lines "
        f"{sites}, bypassing the format and DPI preferences")


#: Two modules import the helper at the CALL SITE rather than at module
#: scope. `spacr.plot` pulls in torch, cv2, seaborn, statsmodels and
#: pingouin, and both of these sit on the cold measure-worker spawn path,
#: where that cost is paid on every worker — see tests/test_measure_spawn.py,
#: which asserts spacr.plot is never imported there.
LAZY_IMPORTERS = ("measure", "io")


@pytest.mark.parametrize("module_name", ROUTED_MODULES)
def test_each_routed_module_imports_the_helper(module_name):
    import importlib

    module = importlib.import_module(f"spacr.{module_name}")
    if module_name in LAZY_IMPORTERS:
        source = Path(module.__file__).read_text()
        assert "from .plot import save_figure" in source, (
            f"spacr.{module_name} does not import save_figure anywhere")
        assert not re.search(r"(?m)^from \.plot import save_figure", source), (
            f"spacr.{module_name} imports save_figure at module scope, which "
            "puts torch and friends on the measure-worker spawn path")
        return
    assert hasattr(module, "save_figure"), (
        f"spacr.{module_name} does not import save_figure")


def test_the_worker_path_modules_do_not_drag_plot_in():
    """The reason `measure` and `io` import it late, stated as a test."""
    import subprocess

    code = (
        "import sys, spacr.measure, spacr.io;"
        "assert 'spacr.plot' not in sys.modules, "
        "'importing spacr.measure/io pulled spacr.plot onto the worker path';"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_the_helper_is_not_a_second_copy_of_the_qt_one():
    """``render_figure_to_png`` stays the screen path; this is the file path.

    They answer different questions — one caps resolution so a preview
    decodes quickly, the other writes what the user asked for — so this
    checks the file helper does NOT inherit the display cap.
    """
    fig = _figure(4, 3)
    assert P.deliverable_dpi(fig, 1200) == 1200, (
        "save_figure applied a display cap to a file the user is keeping")
