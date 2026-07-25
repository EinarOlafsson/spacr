"""
Coverage for the spacr.gui_elements widget block that spans
``spacrCheckbutton`` → ``spacrProgressBar`` → ``spacrSlider`` →
``spacrScrollbarStyle`` → ``spacrFrame`` (gui_elements.py ~1111-1555).

These are pure-Tk widgets, so every test here builds the real widget on a
hidden Tk root and then drives its public methods with synthesised events
(``<Configure>``/``<B1-Motion>``/``<ButtonRelease-1>`` payloads) instead of a
live mainloop. Nothing blocks, nothing is mapped on screen.

Assertions are on observable state: knob geometry in canvas pixels, the
value<->position round trip, canvas item fill colours, label text, and the
child-widget composition of ``spacrFrame``.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import pytest

try:
    import spacr.gui_elements as ge
except Exception as e:  # pragma: no cover - env without a usable display
    pytest.skip(f"spacr.gui_elements unavailable in this env: {e}",
                allow_module_level=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _MotionEvent:
    """Stand-in for a Tk <Button-1>/<B1-Motion> event (only ``x`` is read)."""

    def __init__(self, x):
        self.x = x
        self.y = 0


class _ConfigureEvent:
    """Stand-in for a Tk <Configure> event (only ``width`` is read)."""

    def __init__(self, width):
        self.width = width
        self.height = 20


@pytest.fixture(autouse=True)
def _no_lingering_figures():
    """Belt-and-braces: nothing here plots, but never leak a figure."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except Exception:
        pass


# ---------------------------------------------------------------------------
# spacrCheckbutton
# ---------------------------------------------------------------------------

def test_checkbutton_defaults_create_own_booleanvar(tk_root):
    """With no ``variable`` the widget owns a fresh BooleanVar defaulting False."""
    cb = ge.spacrCheckbutton(tk_root, text="enable")
    tk_root.update_idletasks()

    assert isinstance(cb, ttk.Checkbutton)
    assert isinstance(cb.variable, tk.BooleanVar)
    assert cb.variable.get() is False
    assert cb.text == "enable"
    assert cb.command is None
    assert str(cb.cget("text")) == "enable"
    assert str(cb.cget("style")) == "Spacr.TCheckbutton"


def test_checkbutton_invoke_toggles_variable_and_fires_command(tk_root):
    """invoke() flips the bound variable and calls the registered command."""
    var = tk.BooleanVar(value=False)
    calls = []
    cb = ge.spacrCheckbutton(tk_root, text="t", variable=var,
                             command=lambda: calls.append(var.get()))
    tk_root.update_idletasks()

    assert cb.variable is var
    cb.invoke()
    assert var.get() is True
    cb.invoke()
    assert var.get() is False
    assert calls == [True, False]


# ---------------------------------------------------------------------------
# spacrProgressBar
# ---------------------------------------------------------------------------

def test_progressbar_builds_label_and_palette(tk_root, dark_style):
    """label=True builds the companion tk.Label using the shared palette."""
    bar = ge.spacrProgressBar(tk_root, label=True)
    tk_root.update_idletasks()

    assert isinstance(bar, ttk.Progressbar)
    assert bar['value'] == 0
    assert str(bar.cget("style")) == "spacr.Horizontal.TProgressbar"
    assert bar.label is True
    assert isinstance(bar.progress_label, tk.Label)
    # grid_forget() was called in __init__ -> not managed yet.
    assert bar.progress_label.grid_info() == {}
    assert str(bar.progress_label.cget("text")) == "Processing: 0/0"

    assert bar.fg_color == dark_style['fg_color']
    assert bar.bg_color == dark_style['bg_color']
    assert bar.active_color == dark_style['active_color']
    assert bar.inactive_color == dark_style['inactive_color']
    assert bar.font_size == dark_style['font_size']
    assert bar.operation_type is None
    assert bar.additional_info is None

    # The custom style really carries the palette colours.
    style = ttk.Style()
    assert style.lookup("spacr.Horizontal.TProgressbar",
                        "troughcolor") == bar.inactive_color
    assert style.lookup("spacr.Horizontal.TProgressbar",
                        "background") == bar.active_color


def test_progressbar_without_label_has_no_label_attribute(tk_root):
    """label=False skips the tk.Label entirely and the updaters no-op."""
    bar = ge.spacrProgressBar(tk_root, label=False)
    tk_root.update_idletasks()

    assert bar.label is False
    assert not hasattr(bar, "progress_label")
    # Both label helpers must be safe to call and must not blow up.
    assert bar.update_label() is None
    assert bar.set_label_position() is None


def test_progressbar_set_label_position_maps_the_label(tk_root):
    """set_label_position() grids the previously-forgotten label."""
    bar = ge.spacrProgressBar(tk_root, label=True)
    bar.grid(row=0, column=0)
    tk_root.update_idletasks()

    assert bar.progress_label.grid_info() == {}
    bar.set_label_position()
    tk_root.update_idletasks()

    info = bar.progress_label.grid_info()
    assert info != {}
    assert int(info['padx']) == 5
    assert int(info['pady']) == 5
    assert str(info['sticky']) == 'ew'
    # columnspan is the one geometry key read from a real grid_info key.
    assert int(info['columnspan']) == int(bar.grid_info()['columnspan'])


@pytest.mark.xfail(
    strict=True,
    reason="BUG: spacrProgressBar.set_label_position reads grid_info()['rowID'] "
           "/ ['columnID'], which are never keys of Tk's grid_info() (they are "
           "'row'/'column'), so the label is always pinned to row 1/column 0 "
           "instead of directly beneath the progress bar.",
)
def test_progressbar_label_sits_directly_beneath_the_bar(tk_root):
    """The label must land one row below the bar, in the bar's column."""
    bar = ge.spacrProgressBar(tk_root, label=True)
    bar.grid(row=3, column=2, columnspan=2)
    tk_root.update_idletasks()

    bar.set_label_position()
    tk_root.update_idletasks()

    bar_info = bar.grid_info()
    lab_info = bar.progress_label.grid_info()
    assert int(lab_info['row']) == int(bar_info['row']) + 1
    assert int(lab_info['column']) == int(bar_info['column'])


def test_progressbar_update_label_base_text(tk_root):
    """Plain progress text when neither operation nor extra info is set."""
    bar = ge.spacrProgressBar(tk_root, label=True)
    bar['maximum'] = 20
    bar['value'] = 7
    bar.update_label()
    tk_root.update_idletasks()

    assert str(bar.progress_label.cget("text")) == "Processing: 7/20"


def test_progressbar_update_label_with_operation_type(tk_root):
    """operation_type is appended after a comma."""
    bar = ge.spacrProgressBar(tk_root, label=True)
    bar['maximum'] = 4
    bar['value'] = 2
    bar.operation_type = "segmentation"
    bar.update_label()

    assert str(bar.progress_label.cget("text")) == "Processing: 2/4, segmentation"


def test_progressbar_update_label_flattens_additional_info(tk_root):
    """Comma-separated additional_info is re-joined with single spaces."""
    bar = ge.spacrProgressBar(tk_root, label=True)
    bar['maximum'] = 10
    bar['value'] = 3
    bar.operation_type = "measure"
    bar.additional_info = "batch 1, 12.5 s/img, mem 2GB"
    bar.update_label()

    text = str(bar.progress_label.cget("text"))
    assert text == "Processing: 3/10, measure batch 1 12.5 s/img mem 2GB"
    assert "\n" not in text


def test_progressbar_update_label_ignores_empty_additional_info(tk_root):
    """An empty additional_info string must not add a trailing separator."""
    bar = ge.spacrProgressBar(tk_root, label=True)
    bar['maximum'] = 5
    bar['value'] = 5
    bar.additional_info = ""
    bar.update_label()

    assert str(bar.progress_label.cget("text")) == "Processing: 5/5"


# ---------------------------------------------------------------------------
# spacrSlider — construction
# ---------------------------------------------------------------------------

def test_slider_constructs_with_fixed_length_and_no_entry(tk_root, dark_style):
    """A fixed-length slider parks the knob at ``from_`` and draws 2 items."""
    s = ge.spacrSlider(tk_root, from_=0, to=100, length=200, knob_radius=10)
    tk_root.update_idletasks()

    assert isinstance(s, tk.Frame)
    assert s.specified_length == 200
    assert s.length == 200
    assert s.offset == 0
    assert s.show_index is False
    assert not hasattr(s, "index_var")
    assert s.value == 0
    assert s.knob_position == pytest.approx(10.0)
    assert s.fg_color == dark_style['fg_color']
    # draw_slider() drew exactly the track line + the knob oval.
    assert len(s.canvas.find_all()) == 2
    assert s.canvas.type(s.slider_line) == "line"
    assert s.canvas.type(s.knob) == "oval"
    # Built inactive -> knob wears the inactive colour.
    assert s.canvas.itemcget(s.knob, "fill") == s.inactive_color


def test_slider_without_length_falls_back_to_canvas_reqwidth(tk_root):
    """length=None means the initial length comes from the canvas request."""
    s = ge.spacrSlider(tk_root, from_=0, to=10)
    tk_root.update_idletasks()

    assert s.specified_length is None
    assert s.length == s.canvas.winfo_reqwidth()
    assert s.length > 0


def test_slider_show_index_builds_entry_bound_to_value(tk_root):
    """show_index=True adds a tk.Entry mirroring the current value."""
    s = ge.spacrSlider(tk_root, from_=0, to=100, value=42, length=200,
                       show_index=True)
    tk_root.update_idletasks()

    assert isinstance(s.index_entry, tk.Entry)
    assert s.index_var.get() == "42"
    assert s.index_entry.get() == "42"
    assert int(s.index_entry.grid_info()['column']) == 0
    assert int(s.canvas.grid_info()['column']) == 1


# ---------------------------------------------------------------------------
# spacrSlider — value/position mapping
# ---------------------------------------------------------------------------

def test_slider_value_position_round_trip(tk_root):
    """value_to_position and position_to_value are exact inverses."""
    s = ge.spacrSlider(tk_root, from_=0, to=100, length=200, knob_radius=10)

    assert s.value_to_position(0) == pytest.approx(10.0)
    assert s.value_to_position(100) == pytest.approx(190.0)
    assert s.value_to_position(50) == pytest.approx(100.0)

    assert s.position_to_value(10) == pytest.approx(0.0)
    assert s.position_to_value(190) == pytest.approx(100.0)
    assert s.position_to_value(100) == pytest.approx(50.0)

    for v in (0, 13, 50, 87.5, 100):
        assert s.position_to_value(s.value_to_position(v)) == pytest.approx(v)


def test_slider_degenerate_range_is_not_a_zero_division(tk_root):
    """from_ == to short-circuits both mappings instead of dividing by zero."""
    s = ge.spacrSlider(tk_root, from_=5, to=5, length=200, knob_radius=10)
    tk_root.update_idletasks()

    assert s.value == 5
    assert s.value_to_position(5) == 10          # == knob_radius
    assert s.value_to_position(999) == 10
    assert s.position_to_value(0) == 5
    assert s.position_to_value(190) == 5
    assert s.knob_position == 10


# ---------------------------------------------------------------------------
# spacrSlider — resize
# ---------------------------------------------------------------------------

def test_slider_resize_keeps_fixed_length_and_centres_it(tk_root):
    """position='center' with a fixed length splits the slack evenly."""
    s = ge.spacrSlider(tk_root, from_=0, to=100, value=50, length=200,
                       knob_radius=10, position="center")
    s.resize_slider(_ConfigureEvent(width=400))
    tk_root.update_idletasks()

    assert s.length == 200
    assert s.offset == 100
    assert s.knob_position == pytest.approx(100.0)
    assert s.canvas.itemcget(s.knob, "fill") == s.inactive_color


def test_slider_resize_right_aligns(tk_root):
    """position='right' pushes the whole track to the far edge."""
    s = ge.spacrSlider(tk_root, from_=0, to=100, length=200, position="RIGHT")
    assert s.position == "right"           # constructor lower()s it
    s.resize_slider(_ConfigureEvent(width=500))

    assert s.length == 200
    assert s.offset == 300


def test_slider_resize_left_uses_ninety_percent_of_width(tk_root):
    """Dynamic length = 90% of the container, flush left."""
    s = ge.spacrSlider(tk_root, from_=0, to=100, value=50, knob_radius=10,
                       position="left")
    s.resize_slider(_ConfigureEvent(width=400))
    tk_root.update_idletasks()

    assert s.length == 360
    assert s.offset == 0
    # knob recomputed against the NEW length: 10 + 0.5 * (360 - 20)
    assert s.knob_position == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# spacrSlider — pointer interaction
# ---------------------------------------------------------------------------

def test_slider_move_knob_updates_value_and_geometry(tk_root):
    """Dragging to the mid-point yields the mid value and re-places the oval."""
    s = ge.spacrSlider(tk_root, from_=0, to=100, length=200, knob_radius=10,
                       show_index=True)
    tk_root.update_idletasks()

    s.move_knob(_MotionEvent(x=100))

    assert s.knob_position == 100
    assert s.get() == pytest.approx(50.0)
    assert s.index_var.get() == "50"
    x1, y1, x2, y2 = s.canvas.coords(s.knob)
    assert (x1, y1, x2, y2) == pytest.approx((90.0, 0.0, 110.0, 20.0))


def test_slider_move_knob_clamps_at_both_ends(tk_root):
    """Pointer positions outside the track clamp to from_/to."""
    s = ge.spacrSlider(tk_root, from_=0, to=100, length=200, knob_radius=10)

    s.move_knob(_MotionEvent(x=-500))
    assert s.knob_position == 10
    assert s.get() == pytest.approx(0.0)

    s.move_knob(_MotionEvent(x=9999))
    assert s.knob_position == 190
    assert s.get() == pytest.approx(100.0)


def test_slider_activate_and_release_swap_knob_colour(tk_root):
    """activate_knob paints the accent colour; release_knob restores it."""
    released = []
    s = ge.spacrSlider(tk_root, from_=0, to=100, length=200, knob_radius=10,
                       command=released.append)
    tk_root.update_idletasks()

    s.activate_knob(_MotionEvent(x=190))
    assert s.canvas.itemcget(s.knob, "fill") == s.active_color
    assert s.get() == pytest.approx(100.0)
    assert released == []          # command only fires on release

    s.release_knob(_MotionEvent(x=190))
    assert s.canvas.itemcget(s.knob, "fill") == s.inactive_color
    assert released == [pytest.approx(100.0)]


def test_slider_release_without_command_is_a_noop(tk_root):
    """No command registered -> release just repaints the knob."""
    s = ge.spacrSlider(tk_root, from_=0, to=100, length=200)
    s.activate_knob(_MotionEvent(x=100))
    assert s.canvas.itemcget(s.knob, "fill") == s.active_color

    assert s.release_knob(_MotionEvent(x=100)) is None
    assert s.canvas.itemcget(s.knob, "fill") == s.inactive_color


def test_slider_jump_to_click_delegates_to_activate(tk_root):
    """jump_to_click is the click-anywhere shortcut onto activate_knob."""
    s = ge.spacrSlider(tk_root, from_=0, to=100, length=200, knob_radius=10)

    s.jump_to_click(_MotionEvent(x=55))

    assert s.knob_position == 55
    assert s.get() == pytest.approx(25.0)
    assert s.canvas.itemcget(s.knob, "fill") == s.active_color


# ---------------------------------------------------------------------------
# spacrSlider — programmatic API
# ---------------------------------------------------------------------------

def test_slider_set_clamps_into_range_and_syncs_entry(tk_root):
    """set() clamps to [from_, to] and mirrors into the index entry."""
    s = ge.spacrSlider(tk_root, from_=10, to=20, value=10, length=200,
                       knob_radius=10, show_index=True)
    tk_root.update_idletasks()

    s.set(15)
    assert s.get() == 15
    assert s.index_var.get() == "15"
    assert s.knob_position == pytest.approx(100.0)
    assert s.canvas.itemcget(s.knob, "fill") == s.active_color

    s.set(999)
    assert s.get() == 20
    assert s.index_var.get() == "20"
    assert s.knob_position == pytest.approx(190.0)

    s.set(-999)
    assert s.get() == 10
    assert s.index_var.get() == "10"
    assert s.knob_position == pytest.approx(10.0)


def test_slider_set_to_rescales_the_track(tk_root):
    """set_to() changes the upper bound and re-places the knob for it."""
    s = ge.spacrSlider(tk_root, from_=0, to=100, value=50, length=200,
                       knob_radius=10)
    assert s.knob_position == pytest.approx(100.0)

    s.set_to(50)

    assert s.to == 50
    assert s.get() == 50
    # value 50 is now the maximum -> knob at the far right of the track.
    assert s.knob_position == pytest.approx(190.0)
    assert s.canvas.itemcget(s.knob, "fill") == s.active_color


def test_slider_update_from_entry_sets_value_and_fires_command(tk_root):
    """A valid integer typed in the entry drives the slider + command."""
    seen = []
    s = ge.spacrSlider(tk_root, from_=0, to=100, length=200, knob_radius=10,
                       show_index=True, command=seen.append)
    tk_root.update_idletasks()

    s.index_var.set("75")
    s.update_slider_from_entry(None)

    assert s.get() == 75
    assert s.knob_position == pytest.approx(145.0)
    assert seen == [75]


def test_slider_update_from_entry_swallows_bad_input(tk_root):
    """Non-numeric entry text is ignored: value and command are untouched."""
    seen = []
    s = ge.spacrSlider(tk_root, from_=0, to=100, value=30, length=200,
                       show_index=True, command=seen.append)
    tk_root.update_idletasks()

    s.index_var.set("not-a-number")
    assert s.update_slider_from_entry(None) is None

    assert s.get() == 30
    assert seen == []
    assert s.index_var.get() == "not-a-number"   # left exactly as typed


# ---------------------------------------------------------------------------
# spacrScrollbarStyle
# ---------------------------------------------------------------------------

def test_scrollbar_style_is_idempotent_and_applies_colors(tk_root):
    """Calling it twice must not re-create the 'from clam' elements."""
    style = ttk.Style()
    ge.spacrScrollbarStyle(style, "#111111", "#222222")
    ge.spacrScrollbarStyle(style, "#111111", "#222222")   # second call: guarded

    assert style.lookup("Custom.Vertical.TScrollbar", "troughcolor") == "#111111"
    assert style.lookup("Custom.Vertical.TScrollbar", "background") == "#111111"
    layout = style.layout("Custom.Vertical.TScrollbar")
    assert layout[0][0] == "Vertical.Scrollbar.trough"
    assert layout[0][1]["children"][0][0] == "Vertical.Scrollbar.thumb"
    names = style.element_names()
    assert names.count("custom.Vertical.Scrollbar.trough") == 1
    assert names.count("custom.Vertical.Scrollbar.thumb") == 1


# ---------------------------------------------------------------------------
# spacrFrame
# ---------------------------------------------------------------------------

def _frame_canvas(frame):
    return [c for c in frame.winfo_children() if isinstance(c, tk.Canvas)][0]


def test_frame_default_builds_canvas_scrollbar_and_inner_frame(tk_root, dark_style):
    """Defaults: a scrolled ttk.Frame inside a canvas, with the themed bar."""
    f = ge.spacrFrame(tk_root, width=300)
    tk_root.update_idletasks()

    assert isinstance(f, ttk.Frame)
    canvas = _frame_canvas(f)
    assert int(canvas.cget("width")) == 300
    assert int(canvas.cget("highlightthickness")) == 0

    bars = [c for c in f.winfo_children() if isinstance(c, ttk.Scrollbar)]
    assert len(bars) == 1
    assert str(bars[0].cget("style")) == "Custom.Vertical.TScrollbar"
    assert int(bars[0].grid_info()['column']) == 1
    assert int(canvas.grid_info()['column']) == 0

    assert isinstance(f.scrollable_frame, ttk.Frame)
    assert f.inactive_color == dark_style['inactive_color']
    assert f.active_color == dark_style['active_color']
    assert f.fg_color == dark_style['fg_color']
    # The rounded backdrop polygon plus the embedded window.
    assert len(canvas.find_all()) >= 1
    assert "polygon" in {canvas.type(i) for i in canvas.find_all()}


def test_frame_without_width_uses_a_quarter_of_the_screen(tk_root):
    """width=None derives the canvas width from the screen width."""
    f = ge.spacrFrame(tk_root)
    tk_root.update_idletasks()

    canvas = _frame_canvas(f)
    assert int(canvas.cget("width")) == tk_root.winfo_screenwidth() // 4


def test_frame_textbox_mode_uses_a_tk_text(tk_root, dark_style):
    """textbox=True swaps the inner ttk.Frame for a word-wrapped tk.Text."""
    f = ge.spacrFrame(tk_root, width=250, textbox=True)
    tk_root.update_idletasks()

    assert isinstance(f.scrollable_frame, tk.Text)
    assert str(f.scrollable_frame.cget("wrap")) == "word"
    assert str(f.scrollable_frame.cget("fg")) == dark_style['fg_color']

    f.scrollable_frame.insert("1.0", "hello spacr")
    assert f.scrollable_frame.get("1.0", "end").strip() == "hello spacr"


def test_frame_scrollbar_false_omits_the_scrollbar(tk_root):
    """scrollbar=False leaves the canvas unbound and adds no ttk.Scrollbar."""
    f = ge.spacrFrame(tk_root, width=200, scrollbar=False)
    tk_root.update_idletasks()

    assert [c for c in f.winfo_children() if isinstance(c, ttk.Scrollbar)] == []
    canvas = _frame_canvas(f)
    assert str(canvas.cget("yscrollcommand")) == ""
    assert isinstance(f.scrollable_frame, ttk.Frame)


def test_frame_scroll_region_follows_inner_frame_configure(tk_root):
    """The <Configure> binding wires the canvas scrollregion to the bbox."""
    f = ge.spacrFrame(tk_root, width=200)
    f.grid(row=0, column=0)
    tk.Label(f.scrollable_frame, text="x" * 40).grid(row=0, column=0)
    tk_root.update_idletasks()
    f.scrollable_frame.event_generate("<Configure>")
    tk_root.update_idletasks()

    canvas = _frame_canvas(f)
    region = str(canvas.cget("scrollregion")).split()
    assert len(region) == 4
    assert [float(v) for v in region] == pytest.approx(
        [float(v) for v in canvas.bbox("all")]
    )


def test_frame_rounded_rectangle_returns_a_smooth_polygon(tk_root):
    """rounded_rectangle draws a single smoothed polygon of 17 points."""
    f = ge.spacrFrame(tk_root, width=120)
    canvas = _frame_canvas(f)
    before = set(canvas.find_all())

    item = f.rounded_rectangle(canvas, 0, 0, 100, 60, radius=15, fill="#123456")
    tk_root.update_idletasks()

    assert item not in before
    assert canvas.type(item) == "polygon"
    assert canvas.itemcget(item, "fill") == "#123456"
    assert canvas.itemcget(item, "smooth") in ("1", "true", "bezier")
    coords = canvas.coords(item)
    assert len(coords) == 34          # 17 (x, y) pairs
    assert min(coords) >= 0
    assert max(coords) <= 100
