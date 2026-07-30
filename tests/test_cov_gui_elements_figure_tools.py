"""Coverage for the figure-tool tail of :mod:`spacr.gui_elements`.

Covers ``standardize_figure``, the error path of ``save_figure_as_format``,
the interactive ``modify_figure`` Toplevel (driven through real Tk widgets)
and ``generate_dna_matrix`` (GIF + video + font-fallback branches).

Everything runs CPU-only and offline. Tk widgets are created against the
shared hidden root from ``conftest.tk_root`` and immediately withdrawn so no
window is ever mapped on a developer's display; matplotlib is pinned to Agg
by the suite-wide MPLBACKEND setting.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from spacr import gui_elements as GE

FONT_PATH = os.path.join(
    os.path.dirname(GE.__file__),
    "resources", "font", "open_sans", "static", "OpenSans-Regular.ttf",
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _fig_with_line():
    """A one-axes figure carrying a line, title, axis labels and a legend."""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4], label="series")
    ax.set_title("t")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    return fig


# ---------------------------------------------------------------------------
# standardize_figure
# ---------------------------------------------------------------------------

def test_standardize_figure_applies_dark_palette(tk_root):
    fig = _fig_with_line()
    style_out = GE.set_dark_style(ttk.Style())
    bg = to_rgba(style_out["bg_color"])
    fg = to_rgba(style_out["fg_color"])
    font_size = style_out["font_size"]

    out = GE.standardize_figure(fig)

    assert out is fig
    ax = fig.get_axes()[0]

    # Spines: top/right hidden, left/bottom shown, 1 px, foreground colour.
    assert ax.spines["top"].get_visible() is False
    assert ax.spines["right"].get_visible() is False
    assert ax.spines["left"].get_visible() is True
    assert ax.spines["bottom"].get_visible() is True
    for spine in ax.spines.values():
        assert spine.get_linewidth() == 1
        assert to_rgba(spine.get_edgecolor()) == fg

    # Lines restyled to 1 px in the foreground colour.
    assert [ln.get_linewidth() for ln in ax.get_lines()] == [1]
    assert to_rgba(ax.get_lines()[0].get_color()) == fg

    # Backgrounds painted with the palette background.
    assert to_rgba(ax.get_facecolor()) == bg
    assert to_rgba(fig.patch.get_facecolor()) == bg

    # Typography: title / axis labels / ticks all at the palette size+colour.
    assert ax.title.get_fontsize() == font_size
    assert to_rgba(ax.title.get_color()) == fg
    assert ax.xaxis.label.get_fontsize() == font_size
    assert ax.yaxis.label.get_fontsize() == font_size
    assert to_rgba(ax.xaxis.label.get_color()) == fg
    assert to_rgba(ax.yaxis.label.get_color()) == fg
    tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
    assert tick_labels, "expected tick labels to restyle"
    for label in tick_labels:
        assert label.get_fontsize() == font_size
        assert to_rgba(label.get_color()) == fg
        assert label.get_fontproperties().get_file() == style_out["font_loader"].font_path


def test_standardize_figure_without_axes_only_paints_figure(tk_root):
    """No axes -> the per-axes loop body never runs but the patch is painted."""
    fig = plt.figure()
    assert fig.get_axes() == []
    bg = to_rgba(GE.set_dark_style(ttk.Style())["bg_color"])

    out = GE.standardize_figure(fig)

    assert out is fig
    assert to_rgba(fig.patch.get_facecolor()) == bg


# ---------------------------------------------------------------------------
# modify_figure_properties -- documented-but-dead `title` argument
# ---------------------------------------------------------------------------

def test_modify_figure_properties_sets_title():
    fig = _fig_with_line()
    GE.modify_figure_properties(fig, title="brand new title")
    assert fig.get_axes()[0].get_title() == "brand new title"


# ---------------------------------------------------------------------------
# save_figure_as_format -- failure path
# ---------------------------------------------------------------------------

def test_save_figure_as_format_reports_savefig_failure(tmp_path, monkeypatch, capsys):
    """An unsupported format makes savefig raise; the error is caught + printed."""
    target = tmp_path / "fig.bogusfmt"
    monkeypatch.setattr(GE.filedialog, "asksaveasfilename", lambda **kw: str(target))

    GE.save_figure_as_format(_fig_with_line(), "bogusfmt")

    captured = capsys.readouterr().out
    assert "Error saving figure:" in captured
    assert not target.exists()


# ---------------------------------------------------------------------------
# modify_figure -- the interactive Toplevel
# ---------------------------------------------------------------------------

@pytest.fixture
def display_figure_calls(monkeypatch):
    """Replace spacr.gui_core.display_figure with a recorder."""
    import spacr.gui_core as gui_core
    calls = []
    monkeypatch.setattr(gui_core, "display_figure", lambda f: calls.append(f))
    return calls


def _open_modify_window(tk_root, fig):
    """Run modify_figure and return the (withdrawn) Toplevel it created."""
    before = set(tk_root.winfo_children())
    GE.modify_figure(fig)
    new = [w for w in tk_root.winfo_children()
           if isinstance(w, tk.Toplevel) and w not in before]
    assert len(new) == 1, f"expected exactly one new Toplevel, got {new}"
    win = new[0]
    win.withdraw()
    return win


def _widgets(win):
    children = win.winfo_children()
    entries = [w for w in children if isinstance(w, tk.Entry)]
    checks = [w for w in children if isinstance(w, ttk.Checkbutton)]
    buttons = [w for w in children if isinstance(w, tk.Button)]
    return entries, checks, buttons


def test_modify_figure_builds_controls_and_applies(tk_root, display_figure_calls):
    fig = _fig_with_line()
    win = _open_modify_window(tk_root, fig)
    try:
        assert win.wm_title() == "Modify Figure Properties"
        entries, checks, buttons = _widgets(win)
        # 11 labelled entries, 4 checkbuttons, 1 Apply button.
        assert len(entries) == 11
        assert len(checks) == 4
        assert len(buttons) == 1
        assert buttons[0].cget("text") == "Apply"
        assert [c.cget("text") for c in checks] == [
            "Grid", "Legend", "Spleens", "Remove Axes"]

        values = ["1.5", "0.5", "3", "14", "(0, 3)", "(0, 5)",
                  "hello", "45", "#101010", "#ff0000", "#00ff00"]
        for entry, value in zip(entries, values):
            entry.delete(0, "end")
            entry.insert(0, value)

        buttons[0].invoke()

        ax = fig.get_axes()[0]
        assert ax.get_xlim() == (0.0, 3.0)
        assert ax.get_ylim() == (0.0, 5.0)
        assert ax.get_lines()[0].get_linewidth() == 3.0
        assert to_rgba(ax.get_lines()[0].get_color()) == to_rgba("#00ff00")
        assert to_rgba(ax.get_facecolor()) == to_rgba("#101010")
        assert to_rgba(fig.patch.get_facecolor()) == to_rgba("#101010")
        assert ax.title.get_fontsize() == 14
        assert to_rgba(ax.title.get_color()) == to_rgba("#ff0000")
        assert ax.get_xticklabels()[0].get_rotation() == 45.0
        # Apply pushed the figure back to the canvas exactly once.
        assert display_figure_calls == [fig]
    finally:
        win.destroy()


def test_modify_figure_apply_with_bad_number_is_reported(tk_root,
                                                         display_figure_calls,
                                                         capsys):
    fig = _fig_with_line()
    original_xlim = fig.get_axes()[0].get_xlim()
    win = _open_modify_window(tk_root, fig)
    try:
        entries, _checks, buttons = _widgets(win)
        # Non-numeric "Rescale X" -> float() raises ValueError before anything
        # is applied.
        entries[0].delete(0, "end")
        entries[0].insert(0, "not-a-number")
        # A valid x-limit that must NOT get applied because the error fires first.
        entries[4].delete(0, "end")
        entries[4].insert(0, "(0, 99)")

        buttons[0].invoke()

        assert "Invalid input; please enter numeric values." in capsys.readouterr().out
        assert fig.get_axes()[0].get_xlim() == original_xlim
        assert display_figure_calls == []
    finally:
        win.destroy()


def test_modify_figure_spleens_checkbox_toggles_spines(tk_root, display_figure_calls):
    fig = _fig_with_line()
    ax = fig.get_axes()[0]
    win = _open_modify_window(tk_root, fig)
    try:
        _entries, checks, _buttons = _widgets(win)
        spleens = checks[2]
        assert spleens.cget("text") == "Spleens"

        # First click -> True branch: hide top/right, keep left/bottom.
        spleens.invoke()
        assert ax.spines["top"].get_visible() is False
        assert ax.spines["right"].get_visible() is False
        assert ax.spines["left"].get_visible() is True
        assert ax.spines["bottom"].get_visible() is True
        assert ax.spines["top"].get_linewidth() == 2
        assert ax.spines["right"].get_linewidth() == 2
        assert len(display_figure_calls) == 1

        # Second click -> False branch: both spines come back.
        spleens.invoke()
        assert ax.spines["top"].get_visible() is True
        assert ax.spines["right"].get_visible() is True
        assert len(display_figure_calls) == 2
        assert display_figure_calls[-1] is fig
    finally:
        win.destroy()


def test_modify_figure_remove_axes_and_grid_checkboxes(tk_root, display_figure_calls):
    fig = _fig_with_line()
    ax = fig.get_axes()[0]
    win = _open_modify_window(tk_root, fig)
    try:
        entries, checks, buttons = _widgets(win)
        checks[0].invoke()   # Grid on
        checks[1].invoke()   # Legend on
        checks[3].invoke()   # Remove Axes on
        # Leave every entry blank so each optional value stays None.
        for entry in entries:
            entry.delete(0, "end")

        buttons[0].invoke()

        assert ax.xaxis.get_visible() is False
        assert ax.yaxis.get_visible() is False
        assert ax.xaxis._major_tick_kw.get("gridOn") is True
        assert display_figure_calls == [fig]
    finally:
        win.destroy()


# ---------------------------------------------------------------------------
# generate_dna_matrix
# ---------------------------------------------------------------------------

BASE_SIZE = 20
CANVAS_W = 40          # -> 2 columns
CANVAS_H = 24 * BASE_SIZE   # -> 24 rows, wide enough to reach every draw branch


@pytest.fixture
def deterministic_random(monkeypatch):
    """Pin the RNG generate_dna_matrix uses so every draw branch is reached.

    * string length 12 -> the 90 %-fade branch starts at row 10
    * initial base position 0 -> the first column reset happens on frame 12
    * white-run start 0 -> rows 0-7 take the highlighted branch, 8-9 the plain one
    """
    def fake_randint(a, b):
        if (a, b) == (10, 100):
            return 12
        if b == 0:
            return 0
        return 0

    monkeypatch.setattr(GE.random, "randint", fake_randint)
    monkeypatch.setattr(GE.random, "random", lambda: 0.0)


def _gen(tmp_path, name, **kwargs):
    out = tmp_path / name
    params = dict(
        output_path=str(out), canvas_width=CANVAS_W, canvas_height=CANVAS_H,
        duration=3, fps=10, base_size=BASE_SIZE, transition_frames=2,
        font_type=FONT_PATH, lowercase_prob=0.5,
    )
    params.update(kwargs)
    GE.generate_dna_matrix(**params)
    return out


def test_generate_dna_matrix_writes_animated_gif(tmp_path, deterministic_random):
    from PIL import Image

    out = _gen(tmp_path, "dna.gif")

    assert out.exists() and out.stat().st_size > 0
    with Image.open(out) as im:
        assert im.format == "GIF"
        assert im.size == (CANVAS_W, CANVAS_H)
        # transition_frames=2 blended frames are appended to the rendered ones.
        assert im.n_frames > 2
        im.seek(0)
        first = np.array(im.convert("RGB"))
    # Frames are only kept when they contain drawn glyphs, so the GIF is not
    # a stack of black rectangles.
    assert first.max() > 0


def test_generate_dna_matrix_default_enhance_brightens_frames(tmp_path,
                                                              deterministic_random):
    """enhance=None picks the [1.1, 1.5, 1.2, 1.5] default and is applied;
    a falsy enhance skips the enhancement block entirely."""
    from PIL import Image

    enhanced = _gen(tmp_path, "enhanced.gif")
    plain = _gen(tmp_path, "plain.gif", enhance=[])

    with Image.open(enhanced) as im:
        im.seek(0)
        a = np.asarray(im.convert("RGB"), dtype=np.int32)
        n_enhanced = im.n_frames
    with Image.open(plain) as im:
        im.seek(0)
        b = np.asarray(im.convert("RGB"), dtype=np.int32)
        n_plain = im.n_frames

    assert a.shape == b.shape
    assert n_enhanced == n_plain          # same render, different post-processing
    assert not np.array_equal(a, b)       # the enhancement actually changed pixels
    assert a.sum() > b.sum()              # brightness/contrast boost


def test_generate_dna_matrix_video_branch(tmp_path, monkeypatch, deterministic_random):
    """`.mp4` / `.avi` go through cv2.VideoWriter with the matching fourcc."""
    recorded = {}

    class _FakeWriter:
        def __init__(self, path, fourcc, fps, size):
            recorded["path"] = path
            recorded["fourcc"] = fourcc
            recorded["fps"] = fps
            recorded["size"] = size
            recorded["frames"] = []
            recorded["released"] = False

        def write(self, frame):
            recorded["frames"].append(frame)

        def release(self):
            recorded["released"] = True

    monkeypatch.setattr(GE.cv2, "VideoWriter", _FakeWriter)

    out = _gen(tmp_path, "dna.mp4")

    assert recorded["path"] == str(out)
    assert recorded["fps"] == 10
    assert recorded["size"] == (CANVAS_W, CANVAS_H)
    assert recorded["fourcc"] == GE.cv2.VideoWriter_fourcc(*"mp4v")
    assert recorded["released"] is True
    assert len(recorded["frames"]) > 2
    for frame in recorded["frames"]:
        assert frame.shape == (CANVAS_H, CANVAS_W, 3)
        assert frame.dtype == np.uint8
    mp4_frames = len(recorded["frames"])

    recorded.clear()
    _gen(tmp_path, "dna.avi")
    assert recorded["fourcc"] == GE.cv2.VideoWriter_fourcc(*"XVID")
    assert recorded["fourcc"] != GE.cv2.VideoWriter_fourcc(*"mp4v")
    assert len(recorded["frames"]) == mp4_frames


def test_generate_dna_matrix_unknown_extension_writes_nothing(tmp_path,
                                                              deterministic_random):
    """An extension that is neither gif nor video falls through save_output."""
    out = _gen(tmp_path, "dna.tiff")
    assert not out.exists()


def test_generate_dna_matrix_falls_back_to_default_font(tmp_path,
                                                        deterministic_random):
    """A font_type that cannot be opened hits the IOError fallback."""
    from PIL import Image, ImageFont

    with pytest.raises(OSError):
        ImageFont.truetype("definitely-not-a-real-font.ttf", BASE_SIZE)

    out = _gen(tmp_path, "fallback.gif", font_type="definitely-not-a-real-font.ttf")

    assert out.exists()
    with Image.open(out) as im:
        assert im.size == (CANVAS_W, CANVAS_H)
        assert im.n_frames > 2
