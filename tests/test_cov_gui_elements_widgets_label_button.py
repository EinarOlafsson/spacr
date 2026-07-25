"""Branch coverage for the spacr.gui_elements label / button / switch widgets.

Targets ``spacrLabel``, ``spacrButton`` and ``spacrSwitch`` (gui_elements.py
lines ~1557-1994).  Every alternative branch is exercised for real against a
hidden Tk root:

* ``spacrLabel``   - explicit vs screen-derived height, right vs centre
  alignment, ttk-style rendering vs canvas text, font-loader present vs
  absent, non-OpenSans font family with and without an explicit font, and
  both ``set_text`` code paths.
* ``spacrButton``  - text/no-text geometry, outline on/off, font-loader
  fallbacks, all three icon-loading outcomes (direct hit, space->underscore
  retry, default icon), the hover colour-fade state machine including its
  three failure-injection paths, the icon zoom animation, click dispatch and
  the description walk-up over a parent that does / does not provide one.
* ``spacrSwitch``  - default and supplied variables, both ``update_switch``
  states, both animation directions, ``get``/``set`` and the rounded-rect
  geometry helper.

No Tk main loop is entered; timer-driven animations are driven by explicitly
pumping the event loop with a bounded timeout.
"""
from __future__ import annotations

import os
import time
import tkinter as tk
import tkinter.font as tkFont
from tkinter import ttk

import pytest

try:
    import spacr.gui_elements as ge
except Exception as e:  # pragma: no cover - environment without a display
    pytest.skip(f"spacr.gui_elements unavailable: {e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_stray_figures():
    """Never let a matplotlib window survive a test in this module."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


def _pump(root, predicate=None, timeout=5.0):
    """Drive the Tk event loop until ``predicate()`` is true or time runs out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        root.update_idletasks()
        root.update()
        if predicate is None or predicate():
            return True
        time.sleep(0.005)
    return bool(predicate()) if predicate is not None else True


def _patch_style(monkeypatch, **overrides):
    """Make every set_dark_style() call inside gui_elements return overrides.

    The real implementation still runs (so widgets are really themed); only
    the returned dict is amended, and a *copy* is amended so the module-level
    cache is never mutated.
    """
    real = ge.set_dark_style

    def wrapper(*args, **kwargs):
        out = dict(real(*args, **kwargs))
        out.update(overrides)
        return out

    monkeypatch.setattr(ge, "set_dark_style", wrapper)
    return wrapper


def _boom(*args, **kwargs):
    raise RuntimeError("injected failure")


class _DescriptionHost(tk.Frame):
    """A parent frame that implements the show/clear description protocol."""

    def __init__(self, master):
        super().__init__(master)
        self.main_buttons = {}
        self.additional_buttons = {}
        self.shown = []
        self.cleared = 0

    def show_description(self, text):
        self.shown.append(text)

    def clear_description(self):
        self.cleared += 1


# ---------------------------------------------------------------------------
# spacrLabel
# ---------------------------------------------------------------------------

def test_label_explicit_height_right_align_geometry(tk_root):
    """height= is honoured verbatim and width is 10x height; text is right-anchored."""
    w = ge.spacrLabel(tk_root, text="hello", height=30)
    tk_root.update_idletasks()

    assert isinstance(w, tk.Frame)
    assert w.canvas.winfo_reqheight() == 30
    assert w.canvas.winfo_reqwidth() == 300
    # default align -> anchor 'e', drawn 5px in from the right edge
    assert w.align == "right"
    assert w.canvas.itemcget(w.label_text, "anchor") == tk.E
    assert w.canvas.coords(w.label_text) == [300 - 5, 15]
    assert w.canvas.itemcget(w.label_text, "text") == "hello"


def test_label_default_height_derives_from_screen(tk_root):
    """With height=None the canvas is screen_height // 50 tall."""
    expected_h = tk_root.winfo_screenheight() // 50
    w = ge.spacrLabel(tk_root, text="auto")
    tk_root.update_idletasks()

    assert w.canvas.winfo_reqheight() == expected_h
    assert w.canvas.winfo_reqwidth() == expected_h * 10


def test_label_center_alignment_anchor_and_coords(tk_root):
    """align='center' anchors the text item at the canvas mid-point."""
    w = ge.spacrLabel(tk_root, text="mid", align="center", height=20)
    tk_root.update_idletasks()

    assert w.align == "center"
    assert w.canvas.itemcget(w.label_text, "anchor") == tk.CENTER
    assert w.canvas.coords(w.label_text) == [200 // 2, 10]


def test_label_drops_conflicting_kwargs_but_keeps_frame_options(tk_root):
    """foreground/background/anchor/justify/wraplength are filtered out of **kwargs."""
    w = ge.spacrLabel(
        tk_root, text="x", height=20, bd=3,
        foreground="red", background="blue", anchor="w",
        justify="left", wraplength=10,
    )
    tk_root.update_idletasks()

    assert int(w.cget("bd")) == 3
    # the filtered keys never reached the Frame: a Frame has no 'foreground'
    with pytest.raises(tk.TclError):
        w.cget("foreground")


def _spy_on_style_configure(monkeypatch, style_name, sink):
    """Record every ttk.Style.configure() call made against ``style_name``."""
    real_configure = ttk.Style.configure

    def spy(self, style, query_opt=None, **kw):
        if style == style_name and kw:
            sink.append(dict(kw))
        return real_configure(self, style, query_opt, **kw)

    monkeypatch.setattr(ttk.Style, "configure", spy)


def test_label_ttk_style_branch_with_font_loader(tk_root, monkeypatch):
    """style= renders a real ttk.Label styled with the loaded OpenSans font."""
    recorded = []
    _spy_on_style_configure(monkeypatch, "CovA.TLabel", recorded)
    w = ge.spacrLabel(tk_root, text="styled", style="CovA.TLabel", height=24)
    tk_root.update_idletasks()

    assert isinstance(w.label_text, ttk.Label)
    assert w.label_text.cget("text") == "styled"
    assert str(w.label_text.cget("style")) == "CovA.TLabel"
    assert str(w.label_text.cget("anchor")) == "e"
    # the ttk style was configured from the palette, using the font loader
    assert len(recorded) == 1
    assert recorded[0]["background"] == w.style_out["bg_color"]
    assert recorded[0]["foreground"] == w.style_out["fg_color"]
    assert isinstance(recorded[0]["font"], tkFont.Font)
    assert recorded[0]["font"].actual("size") == w.style_out["font_size"]


def test_label_ttk_style_branch_without_font_loader(tk_root, monkeypatch):
    """font_loader=None takes the fallback ttk_style.configure(font=font_style) arm."""
    recorded = []
    _spy_on_style_configure(monkeypatch, "CovB.TLabel", recorded)
    _patch_style(monkeypatch, font_loader=None)
    w = ge.spacrLabel(tk_root, text="nofont", style="CovB.TLabel",
                      align="center", height=24)
    tk_root.update_idletasks()

    assert isinstance(w.label_text, ttk.Label)
    assert w.font_loader is None
    assert str(w.label_text.cget("anchor")) == "center"
    assert len(recorded) == 1
    assert recorded[0]["background"] == w.style_out["bg_color"]
    assert recorded[0]["foreground"] == w.style_out["fg_color"]
    # no loader -> the raw self.font_style is handed to ttk instead of a Font
    assert recorded[0]["font"] == w.font_style == "OpenSans"


def test_label_canvas_text_without_font_loader_uses_font_style(tk_root, monkeypatch):
    """No style, no font_loader -> canvas text drawn with self.font_style."""
    _patch_style(monkeypatch, font_loader=None)
    w = ge.spacrLabel(tk_root, text="plain", height=20)
    tk_root.update_idletasks()

    assert w.font_loader is None
    assert isinstance(w.label_text, int)          # a canvas item id
    assert w.canvas.itemcget(w.label_text, "text") == "plain"
    assert w.canvas.itemcget(w.label_text, "fill") == w.style_out["fg_color"]


def test_label_non_opensans_family_builds_a_tkfont(tk_root, monkeypatch):
    """A non-OpenSans family with font=None builds a tkFont.Font from the palette."""
    _patch_style(monkeypatch, font_family="Helvetica", font_loader=None)
    w = ge.spacrLabel(tk_root, text="helv", height=20)
    tk_root.update_idletasks()

    assert isinstance(w.font_style, tkFont.Font)
    assert w.font_style.actual("size") == w.style_out["font_size"]
    assert w.font_style.cget("weight") == tkFont.NORMAL


def test_label_non_opensans_family_honours_explicit_font(tk_root, monkeypatch):
    """A non-OpenSans family with an explicit font= keeps that font object."""
    _patch_style(monkeypatch, font_family="Helvetica", font_loader=None)
    w = ge.spacrLabel(tk_root, text="explicit", font=("Arial", 9), height=20)
    tk_root.update_idletasks()

    assert w.font_style == ("Arial", 9)
    assert w.canvas.itemcget(w.label_text, "text") == "explicit"


def test_label_set_text_canvas_branch(tk_root):
    w = ge.spacrLabel(tk_root, text="before", height=20)
    tk_root.update_idletasks()
    w.set_text("after")
    assert w.canvas.itemcget(w.label_text, "text") == "after"


def test_label_set_text_ttk_style_branch(tk_root):
    w = ge.spacrLabel(tk_root, text="before", style="CovC.TLabel", height=20)
    tk_root.update_idletasks()
    w.set_text("after")
    assert w.label_text.cget("text") == "after"


# ---------------------------------------------------------------------------
# spacrButton - construction
# ---------------------------------------------------------------------------

def test_button_show_text_geometry_and_caption(tk_root):
    """show_text=True widens the button to 3x size and draws a capitalized caption."""
    btn = ge.spacrButton(tk_root, text="run masks", size=40, show_text=True,
                         animation=False)
    tk_root.update_idletasks()

    assert btn.button_width == 120
    assert btn.canvas.winfo_reqwidth() == 124
    assert btn.canvas.winfo_reqheight() == 44
    assert btn.text == "Run masks"                      # only first letter upper
    assert btn.canvas.itemcget(btn.button_text, "text") == "Run masks"
    assert btn.canvas.itemcget(btn.button_text, "anchor") == "w"
    assert btn.canvas.coords(btn.button_text) == [50, 22]


def test_button_without_text_is_square_and_has_no_caption(tk_root):
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=False)
    tk_root.update_idletasks()

    assert btn.button_width == 40
    assert btn.canvas.winfo_reqwidth() == 44
    assert not hasattr(btn, "button_text")


def test_button_outline_uses_fg_color_stroke(tk_root):
    with_outline = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                                  outline=True, animation=False)
    without = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                             outline=False, animation=False)
    tk_root.update_idletasks()

    assert with_outline.canvas.itemcget(with_outline.button_bg, "outline") == \
        with_outline.fg_color
    assert without.canvas.itemcget(without.button_bg, "outline") == \
        without.inactive_color
    assert with_outline.canvas.itemcget(with_outline.button_bg, "fill") == \
        with_outline.inactive_color


def test_button_rounded_rect_has_seventeen_points(tk_root):
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=False)
    tk_root.update_idletasks()
    item = btn.create_rounded_rectangle(0, 0, 100, 50, radius=10, fill="#123456")
    coords = btn.canvas.coords(item)

    assert btn.canvas.type(item) == "polygon"
    assert len(coords) == 34                       # 17 (x, y) pairs
    assert coords[0] == 10 and coords[1] == 0
    assert coords[-2] == 0 and coords[-1] == 0
    assert btn.canvas.itemcget(item, "fill") == "#123456"


def test_button_font_fallback_without_loader_and_without_font(tk_root, monkeypatch):
    _patch_style(monkeypatch, font_loader=None)
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=True,
                         animation=False)
    tk_root.update_idletasks()

    assert btn.font_loader is None
    assert btn.font_style == ("Arial", 12)


def test_button_font_fallback_without_loader_uses_explicit_font(tk_root, monkeypatch):
    _patch_style(monkeypatch, font_loader=None)
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=True,
                         font=("Courier", 8), animation=False)
    tk_root.update_idletasks()

    assert btn.font_style == ("Courier", 8)


def test_button_font_loader_is_used_when_available(tk_root):
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=True,
                         animation=False)
    tk_root.update_idletasks()

    assert btn.font_loader is not None
    assert btn.font_style is not None
    assert btn.font_style != ("Arial", 12)


# ---------------------------------------------------------------------------
# spacrButton - icon loading
# ---------------------------------------------------------------------------

def test_button_get_icon_path_points_into_the_package_resources(tk_root):
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=False)
    p = btn.get_icon_path("default")

    assert p.endswith(os.path.join("resources", "icons", "default.png"))
    assert os.path.isfile(p)


def test_button_icon_loads_directly_and_is_scaled_to_65_percent(tk_root):
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=False)
    tk_root.update_idletasks()

    assert btn.original_icon_image.size == (26, 26)      # int(40 * 0.65)
    assert btn.icon_photo.width() == 26
    assert btn.canvas.type(btn.button_icon) == "image"
    assert btn.canvas.coords(btn.button_icon) == [22, 22]


def test_button_icon_space_is_retried_as_underscore(tk_root):
    """'cellpose all' has no icon, but 'cellpose_all.png' does - the retry hits."""
    btn = ge.spacrButton(tk_root, text="cellpose all", size=40, show_text=False,
                         animation=False)
    tk_root.update_idletasks()

    assert btn.icon_name == "cellpose all"
    assert not os.path.exists(btn.get_icon_path("cellpose all"))
    assert os.path.isfile(btn.get_icon_path("cellpose_all"))
    assert btn.original_icon_image.size == (26, 26)


def test_button_missing_icon_falls_back_to_default(tk_root, capsys):
    btn = ge.spacrButton(tk_root, text="definitely missing icon", size=40,
                         show_text=False, animation=False)
    tk_root.update_idletasks()
    out = capsys.readouterr().out

    assert "Icon not found" in out
    assert "definitely_missing_icon.png" in out
    from PIL import Image
    expected = Image.open(btn.get_icon_path("default"))
    assert btn.original_icon_image.size == (26, 26)
    assert expected.size[0] > 0


def test_button_unreadable_icon_file_falls_back_to_default(tk_root, tmp_path,
                                                           monkeypatch, capsys):
    """A file that exists but is not an image raises UnidentifiedImageError twice."""
    bad = tmp_path / "not_an_image.png"
    bad.write_bytes(b"this is definitely not a PNG")

    real_get = ge.spacrButton.get_icon_path

    def fake_get(self, icon_name):
        if icon_name == "default":
            return real_get(self, icon_name)
        return str(bad)

    monkeypatch.setattr(ge.spacrButton, "get_icon_path", fake_get)
    btn = ge.spacrButton(tk_root, text="broken", size=40, show_text=False,
                         animation=False)
    tk_root.update_idletasks()
    out = capsys.readouterr().out

    assert "Icon not found" in out
    assert btn.original_icon_image.size == (26, 26)
    assert btn.icon_photo.width() == 26


# ---------------------------------------------------------------------------
# spacrButton - click + description protocol
# ---------------------------------------------------------------------------

def test_button_click_invokes_command(tk_root):
    calls = []
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         command=lambda: calls.append("hit"), animation=False)
    btn.on_click(None)
    btn.on_click()
    assert calls == ["hit", "hit"]


def test_button_click_without_command_is_a_noop(tk_root):
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=False)
    assert btn.command is None
    assert btn.on_click(None) is None


def test_button_description_prefers_main_then_additional_then_default(tk_root):
    host = _DescriptionHost(tk_root)
    host.pack()
    main_btn = ge.spacrButton(host, text="run", size=40, show_text=False,
                              animation=False)
    extra_btn = ge.spacrButton(host, text="mask", size=40, show_text=False,
                               animation=False)
    unknown_btn = ge.spacrButton(host, text="umap", size=40, show_text=False,
                                 animation=False)
    host.main_buttons[main_btn] = "runs the pipeline"
    host.additional_buttons[extra_btn] = "makes masks"
    tk_root.update_idletasks()

    main_btn.update_description(None)
    extra_btn.update_description(None)
    unknown_btn.update_description(None)

    assert host.shown == ["runs the pipeline", "makes masks",
                          "No description available."]

    extra_btn.clear_description(None)
    assert host.cleared == 1


def test_button_description_walks_up_through_intermediate_frames(tk_root):
    """The host may be several frames above the button."""
    host = _DescriptionHost(tk_root)
    host.pack()
    middle = tk.Frame(host)
    middle.pack()
    inner = tk.Frame(middle)
    inner.pack()
    btn = ge.spacrButton(inner, text="run", size=40, show_text=False,
                         animation=False)
    host.main_buttons[btn] = "nested description"
    tk_root.update_idletasks()

    btn.update_description(None)
    btn.clear_description(None)

    assert host.shown == ["nested description"]
    assert host.cleared == 1


def test_button_description_walk_terminates_without_a_host(tk_root):
    """No ancestor implements the protocol -> the walk ends silently at the root."""
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=False)
    tk_root.update_idletasks()

    assert btn.update_description(None) is None
    assert btn.clear_description(None) is None


# ---------------------------------------------------------------------------
# spacrButton - the hover colour fade
# ---------------------------------------------------------------------------

def test_fade_returns_immediately_when_already_at_target(tk_root):
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=False)
    tk_root.update_idletasks()
    before = btn.canvas.itemcget(btn.button_bg, "fill")

    btn._fade_bg_to(btn.inactive_color)

    assert btn.canvas.itemcget(btn.button_bg, "fill") == before
    assert btn._fade_after_id is None


def test_fade_runs_to_completion_and_lands_exactly_on_target(tk_root):
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=False)
    tk_root.update_idletasks()

    btn._fade_bg_to(btn.active_color)
    # the first tick already ran synchronously and scheduled the next one
    assert btn._fade_after_id is not None

    assert _pump(tk_root, lambda: btn._fade_after_id is None, timeout=5.0)
    assert btn.canvas.itemcget(btn.button_bg, "fill").lower() == \
        btn.active_color.lower()


def test_hover_in_then_out_cancels_the_in_flight_fade(tk_root):
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=False)
    tk_root.update_idletasks()

    btn.on_enter()
    first_id = btn._fade_after_id
    assert first_id is not None
    btn.on_leave()                     # cancels the fade-in, starts a fade-out
    assert btn._fade_after_id != first_id

    assert _pump(tk_root, lambda: btn._fade_after_id is None, timeout=5.0)
    assert btn.canvas.itemcget(btn.button_bg, "fill").lower() == \
        btn.inactive_color.lower()


def test_fade_survives_an_after_cancel_that_raises(tk_root, monkeypatch):
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=False)
    tk_root.update_idletasks()
    monkeypatch.setattr(btn, "after_cancel", _boom)
    btn._fade_after_id = "not-a-real-after-id"

    btn._fade_bg_to(btn.active_color)

    # the broken cancel was swallowed and a brand-new fade took over
    assert btn._fade_after_id not in (None, "not-a-real-after-id")
    assert _pump(tk_root, lambda: btn._fade_after_id is None, timeout=5.0)
    assert btn.canvas.itemcget(btn.button_bg, "fill").lower() == \
        btn.active_color.lower()


def test_fade_falls_back_to_inactive_when_itemcget_raises(tk_root, monkeypatch):
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=False)
    tk_root.update_idletasks()
    monkeypatch.setattr(btn.canvas, "itemcget", _boom)

    btn._fade_bg_to(btn.active_color)
    assert _pump(tk_root, lambda: btn._fade_after_id is None, timeout=5.0)

    monkeypatch.undo()
    assert btn.canvas.itemcget(btn.button_bg, "fill").lower() == \
        btn.active_color.lower()


def test_fade_does_an_instant_swap_when_colour_parsing_fails(tk_root, monkeypatch):
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=False)
    tk_root.update_idletasks()
    scheduled = []
    monkeypatch.setattr(btn, "winfo_rgb", _boom)
    monkeypatch.setattr(btn, "after", lambda *a, **k: scheduled.append(a))

    btn._fade_bg_to(btn.active_color)

    assert btn.canvas.itemcget(btn.button_bg, "fill") == btn.active_color
    assert scheduled == []                    # no interpolation was scheduled
    assert btn._fade_after_id is None


def test_fade_tick_aborts_when_itemconfig_raises(tk_root, monkeypatch):
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=False)
    tk_root.update_idletasks()
    calls = []

    def failing_itemconfig(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("canvas gone")

    monkeypatch.setattr(btn.canvas, "itemconfig", failing_itemconfig)
    btn._fade_bg_to(btn.active_color)

    assert len(calls) == 1                    # aborted after the very first tick
    assert btn._fade_after_id is None
    _pump(tk_root, lambda: False, timeout=0.2)
    assert len(calls) == 1                    # and nothing was rescheduled
    monkeypatch.undo()
    assert btn.canvas.itemcget(btn.button_bg, "fill").lower() == \
        btn.inactive_color.lower()


# ---------------------------------------------------------------------------
# spacrButton - the icon zoom animation
# ---------------------------------------------------------------------------

def test_zoom_icon_resizes_the_photo_image(tk_root):
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=False)
    tk_root.update_idletasks()

    btn.zoom_icon(0.5)
    assert btn.icon_photo.width() == 20
    assert btn.icon_photo.height() == 20
    assert btn.canvas.image is btn.icon_photo
    # the original is untouched so repeated zooms stay crisp
    assert btn.original_icon_image.size == (26, 26)


def test_animate_zoom_in_then_out_flips_the_zoom_flag(tk_root):
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=True)
    tk_root.update_idletasks()
    assert btn.is_zoomed_in is False

    btn.animate_zoom(0.85)
    assert _pump(tk_root, lambda: btn.is_zoomed_in is True, timeout=5.0)
    zoomed_in = btn.icon_photo.width()
    assert zoomed_in == int(40 * 0.85)

    btn.animate_zoom(0.65)
    assert _pump(tk_root, lambda: btn.is_zoomed_in is False, timeout=5.0)
    # ten float steps of -0.02 land on 0.6499999..., i.e. 1px shy of the
    # original 26px render - assert the shrink happened, within that slack.
    assert btn.icon_photo.width() == pytest.approx(int(40 * 0.65), abs=1)
    assert btn.icon_photo.width() < zoomed_in


def test_on_enter_and_on_leave_drive_the_zoom_when_animation_is_on(tk_root):
    host = _DescriptionHost(tk_root)
    host.pack()
    btn = ge.spacrButton(host, text="run", size=40, show_text=False,
                         animation=True)
    host.main_buttons[btn] = "zoomy"
    tk_root.update_idletasks()

    btn.on_enter()
    assert host.shown == ["zoomy"]
    assert _pump(tk_root, lambda: btn.is_zoomed_in is True, timeout=5.0)
    zoomed_in = btn.icon_photo.width()
    assert zoomed_in == int(40 * 0.85)

    btn.on_leave()
    assert host.cleared == 1
    assert _pump(tk_root, lambda: btn.is_zoomed_in is False, timeout=5.0)
    assert btn.icon_photo.width() == pytest.approx(int(40 * 0.65), abs=1)
    assert btn.icon_photo.width() < zoomed_in
    # the hover-out fade started by on_leave also settles back to inactive
    assert _pump(tk_root, lambda: btn._fade_after_id is None, timeout=5.0)
    assert btn.canvas.itemcget(btn.button_bg, "fill").lower() == \
        btn.inactive_color.lower()


def test_hover_skips_the_zoom_when_already_in_the_target_state(tk_root):
    """on_enter while zoomed-in (and on_leave while zoomed-out) must not animate."""
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=True)
    tk_root.update_idletasks()
    zooms = []
    btn.animate_zoom = lambda *a, **k: zooms.append(a)

    btn.on_leave()                      # is_zoomed_in is False -> no zoom
    assert zooms == []
    btn.is_zoomed_in = True
    btn.on_enter()                      # already zoomed in -> no zoom
    assert zooms == []
    _pump(tk_root, lambda: btn._fade_after_id is None, timeout=5.0)


def test_animation_disabled_never_touches_the_zoom(tk_root):
    btn = ge.spacrButton(tk_root, text="run", size=40, show_text=False,
                         animation=False)
    tk_root.update_idletasks()
    zooms = []
    btn.animate_zoom = lambda *a, **k: zooms.append(a)

    btn.on_enter()
    btn.on_leave()
    _pump(tk_root, lambda: btn._fade_after_id is None, timeout=5.0)

    assert zooms == []
    assert btn.is_zoomed_in is False


# ---------------------------------------------------------------------------
# spacrSwitch
# ---------------------------------------------------------------------------

def test_switch_defaults_to_a_fresh_false_boolean_var(tk_root):
    sw = ge.spacrSwitch(tk_root, text="power")
    tk_root.update_idletasks()

    assert isinstance(sw.variable, tk.BooleanVar)
    assert sw.get() is False
    assert isinstance(sw.label, ge.spacrLabel)
    assert sw.canvas.coords(sw.switch) == [4.0, 4.0, 16.0, 16.0]
    assert sw.canvas.itemcget(sw.switch, "fill") == "#800080"
    assert sw.canvas.winfo_reqwidth() == 40
    assert sw.canvas.winfo_reqheight() == 20


def test_switch_uses_the_supplied_variable_and_renders_its_state(tk_root):
    var = tk.BooleanVar(value=True)
    sw = ge.spacrSwitch(tk_root, text="on", variable=var)
    tk_root.update_idletasks()

    assert sw.variable is var
    assert sw.get() is True
    assert sw.canvas.coords(sw.switch) == [24.0, 4.0, 36.0, 16.0]
    assert sw.canvas.itemcget(sw.switch, "fill") == "#008080"


def test_switch_set_moves_the_knob_without_animating(tk_root):
    sw = ge.spacrSwitch(tk_root, text="power")
    tk_root.update_idletasks()

    sw.set(True)
    assert sw.get() is True
    assert sw.canvas.coords(sw.switch) == [24.0, 4.0, 36.0, 16.0]
    assert sw.canvas.itemcget(sw.switch, "fill") == "#008080"

    sw.set(False)
    assert sw.get() is False
    assert sw.canvas.coords(sw.switch) == [4.0, 4.0, 16.0, 16.0]
    assert sw.canvas.itemcget(sw.switch, "fill") == "#800080"


def test_switch_toggle_animates_both_directions_and_fires_the_command(tk_root):
    calls = []
    var = tk.BooleanVar(value=False)
    sw = ge.spacrSwitch(tk_root, text="power", variable=var,
                        command=lambda: calls.append(var.get()))
    tk_root.update_idletasks()

    sw.toggle()
    assert var.get() is True
    assert sw.canvas.itemcget(sw.switch, "fill") == "#008080"
    # animate_movement stops one step short of end_x, then recolours
    assert sw.canvas.coords(sw.switch)[0] == 23.0

    sw.toggle(None)
    assert var.get() is False
    assert sw.canvas.itemcget(sw.switch, "fill") == "#800080"
    assert sw.canvas.coords(sw.switch)[0] == 5.0

    assert calls == [True, False]


def test_switch_toggle_without_command_still_flips(tk_root):
    sw = ge.spacrSwitch(tk_root, text="power")
    tk_root.update_idletasks()
    assert sw.command is None

    sw.toggle()
    assert sw.get() is True
    assert sw.canvas.itemcget(sw.switch, "fill") == "#008080"


def test_switch_rounded_rectangle_has_twenty_points(tk_root):
    sw = ge.spacrSwitch(tk_root, text="power")
    tk_root.update_idletasks()
    item = sw.create_rounded_rectangle(0, 0, 60, 30, radius=5, fill="#abcdef",
                                       outline="")
    coords = sw.canvas.coords(item)

    assert sw.canvas.type(item) == "polygon"
    assert len(coords) == 40                    # 20 (x, y) pairs
    assert coords[0] == 5 and coords[1] == 0
    assert coords[-2] == 0 and coords[-1] == 0
    assert sw.canvas.itemcget(item, "fill") == "#abcdef"
    # the background pill created in __init__ uses the same helper
    assert sw.canvas.type(sw.switch_bg) == "polygon"
    assert len(sw.canvas.coords(sw.switch_bg)) == 40
