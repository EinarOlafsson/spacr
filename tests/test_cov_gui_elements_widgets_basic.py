"""CPU-only coverage tests for the basic spaCR GUI widgets.

Covers ``spacr.gui_elements`` lines ~460-1130:

    spacrFont, spacrContainer, spacrEntry, spacrCheck, spacrCombo,
    spacrDropdownMenu and spacrCheckbutton.

Everything runs headless against a hidden Tk root (the ``tk_root`` fixture
from ``tests/conftest.py`` skips cleanly when no display is available).
No network, no CUDA, no matplotlib windows.

Branches that only fire on failure (e.g. ``spacrFont.load_font``'s
``tk.TclError`` fallback, or the ``font_loader is None`` fallbacks in every
widget) are reached by injecting the failure rather than by pinning a
happy path.
"""
from __future__ import annotations

import types

import pytest

import tkinter as tk
from tkinter import ttk


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Never let a stray figure window accumulate across these tests."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


def _patch_no_font_loader(monkeypatch):
    """Force every widget built afterwards down the ``font_loader is None`` path.

    ``set_dark_style`` memoizes its return dict in a module global, so the
    wrapper must hand back a *copy* — mutating the real dict would poison
    the cache for every other test in the session.
    """
    import spacr.gui_elements as ge

    real = ge.set_dark_style

    def fake(*args, **kwargs):
        out = dict(real(*args, **kwargs))
        out["font_loader"] = None
        return out

    monkeypatch.setattr(ge, "set_dark_style", fake)
    return fake


def _n_items(widget):
    """Number of canvas items currently drawn on ``widget.canvas``."""
    return len(widget.canvas.find_all())


# ---------------------------------------------------------------------------
# spacrFont
# ---------------------------------------------------------------------------

def test_spacrfont_resolves_every_bundled_style(tk_root):
    """Regular/Bold/Italic each map onto their own bundled .ttf."""
    from spacr.gui_elements import spacrFont

    f = spacrFont("OpenSans", "Regular", font_size=12)

    assert f.font_path.endswith("OpenSans-Regular.ttf")
    assert f.get_font_path("OpenSans", "Bold").endswith("OpenSans-Bold.ttf")
    assert f.get_font_path("OpenSans", "Italic").endswith("OpenSans-Italic.ttf")
    # all three live in the same bundled directory
    assert "resources" in f.get_font_path("OpenSans", "Bold").replace("\\", "/")


@pytest.mark.parametrize(
    "name,style",
    [
        ("OpenSans", "SemiBoldCondensed"),  # known family, unknown style
        ("Comic Sans MS", "Regular"),       # unknown family entirely
        ("OpenSans", ""),                   # empty style
    ],
)
def test_spacrfont_unknown_combination_raises_valueerror(tk_root, name, style):
    from spacr.gui_elements import spacrFont

    f = spacrFont("OpenSans", "Regular")
    with pytest.raises(ValueError) as exc:
        f.get_font_path(name, style)
    assert name in str(exc.value)
    assert style in str(exc.value)


def test_spacrfont_constructor_propagates_valueerror(tk_root):
    """A bad style is rejected during __init__, before any Tk work happens."""
    from spacr.gui_elements import spacrFont

    with pytest.raises(ValueError):
        spacrFont("OpenSans", "Nope")


def test_spacrfont_load_font_falls_back_to_file_registration(tk_root, monkeypatch):
    """When Tk does not know the family, the .ttf path is registered by name."""
    import spacr.gui_elements as ge

    calls = []
    sentinel = object()

    def fake_font(*args, **kwargs):
        calls.append(kwargs)
        if "file" in kwargs:
            return sentinel
        raise tk.TclError("simulated: font family not registered")

    monkeypatch.setattr(ge.font, "Font", fake_font)

    f = ge.spacrFont("OpenSans", "Bold", font_size=17)

    assert f.tk_font is sentinel
    assert len(calls) == 2
    # first probe: plain family lookup that we made fail
    assert calls[0] == {"family": "OpenSans", "size": 17}
    # fallback: register the resolved on-disk file under the family name
    assert calls[1]["name"] == "OpenSans"
    assert calls[1]["size"] == 17
    assert calls[1]["file"].endswith("OpenSans-Bold.ttf")


def test_spacrfont_load_font_no_fallback_when_family_known(tk_root):
    """The happy path must not create the file-backed fallback font."""
    from spacr.gui_elements import spacrFont

    f = spacrFont("OpenSans", "Regular", font_size=12)
    assert not hasattr(f, "tk_font")


def test_spacrfont_get_font_size_defaults_and_overrides(tk_root):
    from spacr.gui_elements import spacrFont

    f = spacrFont("OpenSans", "Regular", font_size=13)

    default = f.get_font()
    assert default.cget("size") == 13
    assert default.cget("family") == "OpenSans"

    override = f.get_font(size=31)
    assert override.cget("size") == 31


# ---------------------------------------------------------------------------
# spacrContainer
# ---------------------------------------------------------------------------

def _container_with_panes(parent, orient, n=2):
    from spacr.gui_elements import spacrContainer

    c = spacrContainer(parent, orient=orient)
    widgets = [tk.Frame(c) for _ in range(n)]
    for w in widgets:
        c.add(w)
    return c, widgets


def test_container_defaults_and_empty_reposition_is_a_noop(tk_root):
    from spacr.gui_elements import spacrContainer

    c = spacrContainer(tk_root)
    assert c.orient == tk.VERTICAL
    assert c.bg == "lightgrey"
    assert c.sash_thickness == 10
    assert c.panes == [] and c.sashes == []

    # early-return guard: no panes -> nothing to lay out, no exception
    assert c.reposition_panes() is None
    assert c.panes == []


def test_container_add_vertical_grids_panes_and_creates_sashes(tk_root):
    c, widgets = _container_with_panes(tk_root, tk.VERTICAL, n=3)

    assert len(c.panes) == 3
    # one sash between each adjacent pair
    assert len(c.sashes) == 2

    # each widget was re-parented into its own pane frame
    for (pane, widget), expected in zip(c.panes, widgets):
        assert widget is expected
        assert widget.grid_info()["in"] is pane

    # panes occupy even rows, sashes the odd rows in between
    assert [int(p.grid_info()["row"]) for p, _ in c.panes] == [0, 2, 4]
    assert [int(s.grid_info()["row"]) for s in c.sashes] == [1, 3]
    # all panes live in column 0 for a vertical split
    assert {int(p.grid_info()["column"]) for p, _ in c.panes} == {0}


def test_container_add_horizontal_grids_panes_across_columns(tk_root):
    c, _ = _container_with_panes(tk_root, tk.HORIZONTAL, n=3)

    assert [int(p.grid_info()["column"]) for p, _ in c.panes] == [0, 2, 4]
    assert [int(s.grid_info()["column"]) for s in c.sashes] == [1, 3]
    assert {int(p.grid_info()["row"]) for p, _ in c.panes} == {0}


def test_container_single_pane_creates_no_sash(tk_root):
    c, _ = _container_with_panes(tk_root, tk.VERTICAL, n=1)
    assert len(c.panes) == 1
    assert c.sashes == []


@pytest.mark.parametrize(
    "orient,cursor",
    [(tk.VERTICAL, "sb_v_double_arrow"), (tk.HORIZONTAL, "sb_h_double_arrow")],
)
def test_container_sash_cursor_matches_orientation(tk_root, orient, cursor):
    c, _ = _container_with_panes(tk_root, orient, n=2)
    sash = c.sashes[0]
    assert sash.cget("cursor") == cursor
    assert sash.cget("bg") == c.bg
    assert int(sash.cget("height")) == c.sash_thickness
    assert int(sash.cget("width")) == c.sash_thickness


def test_container_on_configure_relays_out_panes(tk_root):
    c, _ = _container_with_panes(tk_root, tk.VERTICAL, n=2)
    # forget the layout, then let the <Configure> handler restore it
    for pane, _ in c.panes:
        pane.grid_forget()
    assert c.panes[1][0].grid_info() == {}

    c.on_configure(types.SimpleNamespace(width=400, height=300))

    assert int(c.panes[0][0].grid_info()["row"]) == 0
    assert int(c.panes[1][0].grid_info()["row"]) == 2


def test_container_sash_hover_highlight_round_trip(tk_root):
    c, _ = _container_with_panes(tk_root, tk.VERTICAL, n=2)
    sash = c.sashes[0]
    event = types.SimpleNamespace(widget=sash)

    c.on_enter_sash(event)
    assert sash.cget("bg") == "blue"

    c.on_leave_sash(event)
    assert sash.cget("bg") == c.bg


def test_container_start_resize_records_origin_and_binds_motion(tk_root):
    c, _ = _container_with_panes(tk_root, tk.VERTICAL, n=2)
    sash = c.sashes[0]
    assert sash.bind("<B1-Motion>") == ""

    c.start_resize(types.SimpleNamespace(widget=sash, y_root=412, x_root=99))

    assert c.start_pos == 412               # vertical -> tracks y_root
    assert c.start_size == sash.winfo_y()
    assert sash.bind("<B1-Motion>") != ""   # drag handler is now live


def test_container_start_resize_horizontal_tracks_x(tk_root):
    c, _ = _container_with_panes(tk_root, tk.HORIZONTAL, n=2)
    sash = c.sashes[0]

    c.start_resize(types.SimpleNamespace(widget=sash, y_root=412, x_root=99))

    assert c.start_pos == 99                # horizontal -> tracks x_root
    assert c.start_size == sash.winfo_x()


def test_container_perform_resize_vertical_pushes_previous_pane(tk_root, monkeypatch):
    """A downward drag re-rows the pane *above* the sash being dragged."""
    c, _ = _container_with_panes(tk_root, tk.VERTICAL, n=2)
    sash = c.sashes[0]
    c.start_resize(types.SimpleNamespace(widget=sash, y_root=0, x_root=0))

    relayouts = []
    monkeypatch.setattr(c, "reposition_panes", lambda: relayouts.append(1))

    c.perform_resize(types.SimpleNamespace(widget=sash, y_root=100, x_root=0))

    new_size = c.start_size + 100
    expected = max(0, (new_size - c.panes[1][0].winfo_height()) // c.sash_thickness)
    assert int(c.panes[0][0].grid_info()["row"]) == expected
    assert expected > 0                      # the drag actually moved something
    assert relayouts == [1]                  # and a relayout was requested


def test_container_perform_resize_vertical_negative_drag_collapses_rows(tk_root, monkeypatch):
    """Dragging above the origin clamps every pane back to row 0."""
    c, _ = _container_with_panes(tk_root, tk.VERTICAL, n=2)
    sash = c.sashes[0]
    c.start_resize(types.SimpleNamespace(widget=sash, y_root=500, x_root=0))
    monkeypatch.setattr(c, "reposition_panes", lambda: None)

    c.perform_resize(types.SimpleNamespace(widget=sash, y_root=0, x_root=0))

    assert [int(p.grid_info()["row"]) for p, _ in c.panes] == [0, 0]


def test_container_perform_resize_horizontal_pushes_previous_pane(tk_root, monkeypatch):
    c, _ = _container_with_panes(tk_root, tk.HORIZONTAL, n=2)
    sash = c.sashes[0]
    c.start_resize(types.SimpleNamespace(widget=sash, y_root=0, x_root=0))
    monkeypatch.setattr(c, "reposition_panes", lambda: None)

    c.perform_resize(types.SimpleNamespace(widget=sash, y_root=0, x_root=80))

    new_size = c.start_size + 80
    expected = max(0, (new_size - c.panes[1][0].winfo_width()) // c.sash_thickness)
    assert int(c.panes[0][0].grid_info()["column"]) == expected
    assert expected > 0


def test_container_perform_resize_horizontal_negative_drag_collapses_columns(tk_root, monkeypatch):
    c, _ = _container_with_panes(tk_root, tk.HORIZONTAL, n=2)
    sash = c.sashes[0]
    c.start_resize(types.SimpleNamespace(widget=sash, y_root=0, x_root=400))
    monkeypatch.setattr(c, "reposition_panes", lambda: None)

    c.perform_resize(types.SimpleNamespace(widget=sash, y_root=0, x_root=0))

    assert [int(p.grid_info()["column"]) for p, _ in c.panes] == [0, 0]


def test_container_perform_resize_relayouts_for_real(tk_root):
    """Un-spied: the trailing reposition_panes() restores the canonical grid."""
    c, _ = _container_with_panes(tk_root, tk.VERTICAL, n=2)
    sash = c.sashes[0]
    c.start_resize(types.SimpleNamespace(widget=sash, y_root=0, x_root=0))

    c.perform_resize(types.SimpleNamespace(widget=sash, y_root=250, x_root=0))

    assert [int(p.grid_info()["row"]) for p, _ in c.panes] == [0, 2]
    assert [int(s.grid_info()["row"]) for s in c.sashes] == [1]


# ---------------------------------------------------------------------------
# spacrEntry
# ---------------------------------------------------------------------------

def test_spacr_entry_builds_pill_and_binds_textvariable(tk_root):
    from spacr.gui_elements import spacrEntry

    var = tk.StringVar(value="seed")
    e = spacrEntry(tk_root, textvariable=var)

    assert isinstance(e.entry, tk.Entry)
    assert e.entry.get() == "seed"
    var.set("changed")
    assert e.entry.get() == "changed"

    # 4 corner arcs + 2 body rectangles, no focus ring yet
    assert _n_items(e) == 6
    assert e.canvas_height == 40
    assert e.entry.cget("bg") == e.bg_color
    assert e.entry.cget("fg") == e.fg_color
    assert e.font_loader is not None


def test_spacr_entry_focus_ring_added_and_removed(tk_root):
    from spacr.gui_elements import spacrEntry

    e = spacrEntry(tk_root)
    baseline = _n_items(e)

    e.on_focus_in(None)
    focused = _n_items(e)
    # ring = 4 extra arcs + 4 straight edges
    assert focused == baseline + 8
    ring_colors = {
        e.canvas.itemcget(i, "outline") for i in e.canvas.find_all()
        if e.canvas.type(i) == "arc"
    }
    assert e.active_color in ring_colors

    e.on_focus_out(None)
    assert _n_items(e) == baseline
    assert e.entry.cget("bg") == e.bg_color


def test_spacr_entry_resize_redraws_without_ring(tk_root):
    from spacr.gui_elements import spacrEntry

    e = spacrEntry(tk_root)
    e.on_focus_in(None)
    assert _n_items(e) == 14

    e._on_resize(types.SimpleNamespace(width=300, height=40))

    assert _n_items(e) == 6
    assert e.active_color not in {
        e.canvas.itemcget(i, "outline") for i in e.canvas.find_all()
    }


def test_spacr_entry_falls_back_to_tuple_font_without_loader(tk_root, monkeypatch):
    from spacr.gui_elements import spacrEntry

    _patch_no_font_loader(monkeypatch)
    e = spacrEntry(tk_root, outline=True)

    assert e.font_loader is None
    assert e.outline is True
    spec = str(e.entry.cget("font"))
    assert e.font_family in spec
    assert str(e.font_size) in spec


# ---------------------------------------------------------------------------
# spacrCheck
# ---------------------------------------------------------------------------

def test_spacr_check_reflects_and_toggles_its_variable(tk_root):
    from spacr.gui_elements import spacrCheck

    var = tk.BooleanVar(value=False)
    chk = spacrCheck(tk_root, text="on?", variable=var)

    # 4 arcs + 2 rectangles + 4 border lines
    assert _n_items(chk) == 10

    def _body_fill():
        rects = [i for i in chk.canvas.find_all() if chk.canvas.type(i) == "rectangle"]
        return chk.canvas.itemcget(rects[0], "fill")

    assert _body_fill() == chk.inactive_color

    chk.toggle_variable(None)
    assert var.get() is True
    assert _body_fill() == chk.active_color          # trace -> update_check redrew

    var.set(False)                                    # external write also redraws
    assert _body_fill() == chk.inactive_color


# ---------------------------------------------------------------------------
# spacrCombo
# ---------------------------------------------------------------------------

def test_spacr_combo_initial_state(tk_root):
    from spacr.gui_elements import spacrCombo

    var = tk.StringVar(value="beta")
    combo = spacrCombo(tk_root, textvariable=var, values=["alpha", "beta"])

    assert combo.values == ["alpha", "beta"]
    assert combo.selected_value == "beta"
    assert combo.label.cget("text") == "beta"
    assert combo.dropdown_menu is None
    assert _n_items(combo) == 6
    assert combo.label.cget("bg") == combo.inactive_color


def test_spacr_combo_defaults_to_own_stringvar_and_empty_values(tk_root):
    from spacr.gui_elements import spacrCombo

    combo = spacrCombo(tk_root)
    assert isinstance(combo.var, tk.StringVar)
    assert combo.values == []
    assert combo.selected_value == ""


def test_spacr_combo_click_opens_then_closes_dropdown(tk_root):
    from spacr.gui_elements import spacrCombo

    combo = spacrCombo(tk_root, values=["a", "b", None])

    combo.on_click(None)
    assert isinstance(combo.dropdown_menu, tk.Toplevel)
    labels = combo.dropdown_menu.winfo_children()
    assert [w.cget("text") for w in labels] == ["a", "b", "None"]
    # ring drawn while the popup is open
    assert _n_items(combo) == 14

    combo.on_click(None)
    assert combo.dropdown_menu is None
    assert _n_items(combo) == 6


def test_spacr_combo_resize_keeps_ring_only_while_open(tk_root):
    from spacr.gui_elements import spacrCombo

    combo = spacrCombo(tk_root, values=["a"])

    combo._on_resize(types.SimpleNamespace(width=250, height=40))
    assert _n_items(combo) == 6            # closed -> no ring

    combo.open_dropdown()
    combo._on_resize(types.SimpleNamespace(width=250, height=40))
    assert _n_items(combo) == 14           # open -> ring survives the redraw

    combo.close_dropdown()


def test_spacr_combo_close_dropdown_is_idempotent(tk_root):
    from spacr.gui_elements import spacrCombo

    combo = spacrCombo(tk_root, values=["a"])
    assert combo.dropdown_menu is None

    combo.close_dropdown()               # no popup -> guard skips the destroy
    assert combo.dropdown_menu is None
    assert _n_items(combo) == 6


def test_spacr_combo_on_select_commits_value_and_closes(tk_root):
    from spacr.gui_elements import spacrCombo

    var = tk.StringVar(value="a")
    combo = spacrCombo(tk_root, textvariable=var, values=["a", "b"])
    combo.open_dropdown()

    combo.on_select("b")

    assert combo.selected_value == "b"
    assert combo.label.cget("text") == "b"
    assert var.get() == "b"
    assert combo.dropdown_menu is None
    assert _n_items(combo) == 6


def test_spacr_combo_on_select_none_displays_the_word_none(tk_root):
    from spacr.gui_elements import spacrCombo

    combo = spacrCombo(tk_root, values=["a", None])
    combo.open_dropdown()

    combo.on_select(None)

    assert combo.selected_value is None
    assert combo.label.cget("text") == "None"
    assert combo.dropdown_menu is None


def test_spacr_combo_set_updates_without_opening(tk_root):
    from spacr.gui_elements import spacrCombo

    var = tk.StringVar(value="a")
    combo = spacrCombo(tk_root, textvariable=var, values=["a", "b"])

    combo.set("b")
    assert combo.selected_value == "b"
    assert combo.label.cget("text") == "b"
    assert var.get() == "b"
    assert combo.dropdown_menu is None

    combo.set(None)
    assert combo.selected_value is None
    assert combo.label.cget("text") == "None"
    assert combo.dropdown_menu is None


def test_spacr_combo_falls_back_to_tuple_font_without_loader(tk_root, monkeypatch):
    from spacr.gui_elements import spacrCombo

    _patch_no_font_loader(monkeypatch)
    combo = spacrCombo(tk_root, values=["a", None])

    assert combo.font_loader is None
    label_spec = str(combo.label.cget("font"))
    assert combo.font_family in label_spec
    assert str(combo.font_size) in label_spec

    combo.open_dropdown()
    items = combo.dropdown_menu.winfo_children()
    assert [w.cget("text") for w in items] == ["a", "None"]
    for item in items:
        item_spec = str(item.cget("font"))
        assert combo.font_family in item_spec
        assert str(combo.font_size) in item_spec
        assert item.cget("anchor") == "w"
    combo.close_dropdown()


def test_spacr_combo_dropdown_geometry_matches_requested_height(tk_root):
    from spacr.gui_elements import spacrCombo

    combo = spacrCombo(tk_root, values=["a", "b", "c"])
    combo.open_dropdown()

    geom = combo.dropdown_menu.geometry()
    size, x, y = geom.split("+")[0], geom.split("+")[1], geom.split("+")[2]
    _w, h = (int(v) for v in size.split("x"))
    # tall enough to show every value, anchored directly under the combo
    assert h == combo.dropdown_menu.winfo_reqheight()
    assert h >= len(combo.values)
    assert int(x) == combo.winfo_rootx()
    assert int(y) == combo.winfo_rooty() + combo.winfo_height()
    # borderless popup, not a decorated window
    assert combo.dropdown_menu.wm_overrideredirect() in (1, True)

    combo.close_dropdown()


# ---------------------------------------------------------------------------
# spacrDropdownMenu
# ---------------------------------------------------------------------------

def _dropdown(parent, **kw):
    from spacr.gui_elements import spacrDropdownMenu

    kw.setdefault("options", ["Alpha", "Beta", "Gamma"])
    kw.setdefault("variable", tk.StringVar(value="Alpha"))
    return spacrDropdownMenu(parent, **kw)


def test_spacr_dropdown_menu_builds_button_and_entries(tk_root):
    dd = _dropdown(tk_root, size=30)

    assert dd.text == "Settings"
    assert dd.button_width == 90
    assert dd.canvas_width == 94
    assert dd.canvas_height == 34
    assert int(dd.canvas.cget("width")) == 94
    assert int(dd.canvas.cget("height")) == 34
    # one menu command per option
    assert dd.menu.index("end") == len(dd.options) - 1
    assert [dd.menu.entrycget(i, "label") for i in range(3)] == ["Alpha", "Beta", "Gamma"]
    # button body starts in the inactive color
    assert dd.canvas.itemcget(dd.button_bg, "fill") == dd.inactive_color
    assert dd.canvas.type(dd.button_bg) == "polygon"
    assert dd.canvas.itemcget(dd.button_text, "text") == "Settings"


def test_spacr_dropdown_menu_hover_swaps_button_fill(tk_root):
    dd = _dropdown(tk_root)

    dd.on_enter()
    assert dd.canvas.itemcget(dd.button_bg, "fill") == dd.active_color

    dd.on_leave()
    assert dd.canvas.itemcget(dd.button_bg, "fill") == dd.inactive_color


def test_spacr_dropdown_menu_click_posts_menu_below_button(tk_root, monkeypatch):
    dd = _dropdown(tk_root)
    posted = []
    monkeypatch.setattr(dd.menu, "post", lambda x, y: posted.append((x, y)))

    dd.on_click(None)

    assert posted == [(dd.winfo_rootx(), dd.winfo_rooty() + dd.winfo_height())]


def test_spacr_dropdown_menu_on_select_invokes_command(tk_root):
    seen = []
    dd = _dropdown(tk_root, command=seen.append)

    dd.on_select("Beta")
    assert seen == ["Beta"]

    # and the registered menu entries route through the same path
    dd.menu.invoke(2)
    assert seen == ["Beta", "Gamma"]


def test_spacr_dropdown_menu_on_select_without_command_is_silent(tk_root):
    dd = _dropdown(tk_root, command=None)
    assert dd.command is None
    assert dd.on_select("Alpha") is None


def test_spacr_dropdown_menu_uses_supplied_font_without_loader(tk_root, monkeypatch):
    _patch_no_font_loader(monkeypatch)

    dd = _dropdown(tk_root, font=("Courier", 9))
    assert dd.font_loader is None
    assert dd.font_style == ("Courier", 9)

    dd2 = _dropdown(tk_root, font=None)
    assert dd2.font_style == ("Arial", 12)


def test_spacr_dropdown_menu_update_styles_marks_active_categories(tk_root):
    from spacr.gui_elements import set_dark_style

    dd = _dropdown(tk_root)
    style_out = set_dark_style(ttk.Style())

    dd.update_styles(active_categories=["Alpha", "Gamma"])

    bg = [str(dd.menu.entrycget(i, "background")) for i in range(3)]
    assert bg == [style_out["active_color"],
                  style_out["bg_color"],
                  style_out["active_color"]]
    for i in range(3):
        assert str(dd.menu.entrycget(i, "foreground")) == style_out["fg_color"]


def test_spacr_dropdown_menu_update_styles_none_leaves_entries_alone(tk_root):
    dd = _dropdown(tk_root)
    before = [str(dd.menu.entrycget(i, "background")) for i in range(3)]

    dd.update_styles()               # active_categories is None -> skip the loop

    assert [str(dd.menu.entrycget(i, "background")) for i in range(3)] == before
    # entries were never explicitly colored, so they stay at the Tk default
    assert before == ["", "", ""]


def test_spacr_dropdown_menu_create_rounded_rectangle_bounds(tk_root):
    dd = _dropdown(tk_root, size=40)
    item = dd.create_rounded_rectangle(10, 20, 60, 70, radius=5, fill="#123456")

    assert dd.canvas.type(item) == "polygon"
    assert dd.canvas.itemcget(item, "fill") == "#123456"
    coords = dd.canvas.coords(item)
    assert len(coords) == 34                        # 17 points x/y
    xs, ys = coords[0::2], coords[1::2]
    assert min(xs) == 10 and max(xs) == 60
    assert min(ys) == 20 and max(ys) == 70


# ---------------------------------------------------------------------------
# spacrCheckbutton
# ---------------------------------------------------------------------------

def test_spacr_checkbutton_binds_variable_and_command(tk_root):
    from spacr.gui_elements import spacrCheckbutton

    var = tk.BooleanVar(value=False)
    fired = []
    cb = spacrCheckbutton(tk_root, text="Enable", variable=var,
                          command=lambda: fired.append(var.get()))

    assert cb.cget("text") == "Enable"
    assert cb.variable is var
    assert "Spacr.TCheckbutton" in str(cb.cget("style"))

    cb.invoke()
    assert var.get() is True
    assert fired == [True]

    cb.invoke()
    assert var.get() is False
    assert fired == [True, False]


def test_spacr_checkbutton_creates_its_own_variable_when_omitted(tk_root):
    from spacr.gui_elements import spacrCheckbutton

    cb = spacrCheckbutton(tk_root)
    assert isinstance(cb.variable, tk.BooleanVar)
    assert cb.variable.get() is False
    assert cb.command is None
    assert cb.cget("text") == ""

    cb.invoke()
    assert cb.variable.get() is True
