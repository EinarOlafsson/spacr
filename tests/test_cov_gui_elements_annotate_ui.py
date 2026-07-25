"""
Coverage tests for ``spacr.gui_elements.AnnotateApp`` construction, the
settings/UMAP Toplevels and the grid-rebuild helpers (gui_elements.py
lines ~3339-4400).

Everything runs against a real (but tiny) sqlite ``png_list`` table and a
real Tk display.  Background work is neutralised rather than mocked away:

* the DB-writer thread started by ``__init__`` is always drained and joined,
* every pending ``after`` callback is cancelled before the Toplevel dies,
* the UMAP runners execute synchronously through a stub ``threading.Thread``
  so the Tk calls they make stay on the main thread,
* ``spacr.core`` is swapped for a stub module so no heavy import (or real
  UMAP) ever happens.

Defensive branches are reached by injection, never by pragma: the macOS
button path via ``platform.system``, the "matplotlib missing" path by
blocking the import, the label-lookup failure paths by feeding widgets
whose ``master``/``cget`` raise.
"""
from __future__ import annotations

import builtins
import contextlib
import os
import sqlite3
import sys
import threading
import types

import pandas as pd
import pytest

pytestmark = pytest.mark.gui  # needs a display


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate across tests."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _collect_only_on_the_main_thread():
    """Keep the cyclic collector off the ThreadPoolExecutor workers.

    ``load_images`` decodes tiles in worker threads while the main thread
    holds ``ImageTk.PhotoImage`` objects.  If a generational collection
    happens to fire inside a worker it can finalise a PhotoImage there,
    which calls into Tk from the wrong thread and aborts the interpreter.
    Collect explicitly on the main thread instead.
    """
    import gc
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        if was_enabled:
            gc.enable()


def _make_db(db_path, n=3, extra_cols=None, png_dir=None):
    """Write a minimal ``png_list`` table and return the list of png paths."""
    png_dir = png_dir or os.path.join(os.path.dirname(db_path), "images")
    paths = [os.path.join(png_dir, f"cell_{i}.png") for i in range(n)]
    data = {
        "png_path": paths,
        "prc": [f"plate1_A01_{i:03d}" for i in range(n)],
        "cell_id": list(range(1, n + 1)),
    }
    if extra_cols:
        data.update(extra_cols)
    con = sqlite3.connect(db_path)
    try:
        pd.DataFrame(data).to_sql("png_list", con, index=False)
    finally:
        con.close()
    return paths


@pytest.fixture
def annotate_env(tmp_path):
    """A spacr-shaped src dir with measurements/measurements.db + png_list."""
    (tmp_path / "measurements").mkdir()
    (tmp_path / "images").mkdir()
    db_path = tmp_path / "measurements" / "measurements.db"
    paths = _make_db(str(db_path), n=3)
    return {"src": str(tmp_path), "db_path": str(db_path), "paths": paths,
            "root": tmp_path}


def _cancel_all_after(widget):
    """Cancel every pending Tcl ``after`` callback in this interpreter."""
    try:
        for aid in widget.tk.splitlist(widget.tk.eval("after info")):
            try:
                widget.after_cancel(aid)
            except Exception:
                pass
    except Exception:
        pass


@contextlib.contextmanager
def _app(tk_root, env, **kwargs):
    """Build an AnnotateApp on a Toplevel and always tear the worker down."""
    import tkinter as tk
    from spacr.gui_elements import AnnotateApp

    kwargs.setdefault("db_path", env["db_path"])
    kwargs.setdefault("src", env["src"])
    kwargs.setdefault("image_size", 1000)

    top = tk.Toplevel(tk_root)
    app = None
    try:
        app = AnnotateApp(root=top, **kwargs)
        tk_root.update_idletasks()
        yield app
    finally:
        if app is not None:
            app.terminate = True
            try:
                app.update_queue.put(app.SENTINEL)
            except Exception:
                pass
            try:
                if app.db_update_thread.is_alive():
                    app.db_update_thread.join(timeout=10)
            except Exception:
                pass
            # release the PhotoImages on the main thread
            try:
                app.images.clear()
            except Exception:
                pass
        _cancel_all_after(top)
        try:
            top.destroy()
        except Exception:
            pass


def _walk(widget):
    """Yield every descendant widget of ``widget`` (depth first)."""
    for child in widget.winfo_children():
        yield child
        yield from _walk(child)


def _toplevels(widget):
    import tkinter as tk
    return [w for w in _walk(widget) if isinstance(w, tk.Toplevel)]


def _label_texts(widget):
    import tkinter as tk
    out = []
    for w in _walk(widget):
        if isinstance(w, tk.Label):
            try:
                out.append(str(w.cget("text")))
            except Exception:
                pass
    return out


def _scratch_toplevel(app):
    """A throw-away Toplevel: AnnotateApp's own root is grid-managed, so we
    cannot pack test widgets straight into it."""
    import tkinter as tk
    return tk.Toplevel(app.root)


def _stub_prefilter(app):
    """Replace prefilter_paths_annotations with a recorder.

    The measurement/threshold branch of the real prefilter needs a full
    measurement schema (cell/cytoplasm/nucleus/pathogen); it belongs to a
    different region of the file, so keep it out of these tests but assert
    it was invoked.
    """
    calls = []

    def record():
        calls.append(True)

    app.prefilter_paths_annotations = record
    return calls


class _SyncThread:
    """Drop-in for ``threading.Thread`` that runs the target inline."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, **_kw):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return False


# ---------------------------------------------------------------------------
# __init__ branches
# ---------------------------------------------------------------------------

def test_init_image_size_as_list_makes_square_tuple(tk_root, annotate_env):
    """image_size given as a list is coerced to a tuple (line 3379)."""
    with _app(tk_root, annotate_env, image_size=[1000, 1000]) as app:
        assert app.image_size == (1000, 1000)
        assert app.grid_rows >= 1 and app.grid_cols >= 1
        assert len(app.labels) == app.grid_rows * app.grid_cols


@pytest.mark.xfail(
    strict=True,
    reason="BUG: AnnotateApp(image_size=[w, h]) uses image_size[0] for both "
           "dimensions, silently discarding the height",
)
def test_init_image_size_list_keeps_width_and_height(tk_root, annotate_env):
    """A non-square ``[w, h]`` must survive as ``(w, h)``."""
    with _app(tk_root, annotate_env, image_size=[1200, 600]) as app:
        assert app.image_size == (1200, 600)


def test_init_rejects_non_int_non_list_image_size(tk_root, annotate_env):
    """A string image_size raises before any thread is started (line 3383)."""
    import tkinter as tk
    from spacr.gui_elements import AnnotateApp

    top = tk.Toplevel(tk_root)
    try:
        with pytest.raises(ValueError, match="Invalid image size"):
            AnnotateApp(root=top, db_path=annotate_env["db_path"],
                        src=annotate_env["src"], image_size="1000")
    finally:
        top.destroy()
    # No worker thread should have been left running by the aborted build.
    assert not any(t.name.startswith("Thread-") and "update_database_worker" in repr(t)
                   for t in threading.enumerate())


def test_init_falls_back_to_arial_when_font_loader_missing(
    tk_root, annotate_env, monkeypatch
):
    """No font loader -> the hard-coded ("Arial", 12) fallback (line 3434)."""
    import spacr.gui_elements as ge

    real = ge.set_dark_style

    def no_font_loader(style, *a, **kw):
        out = dict(real(style, *a, **kw))
        out["font_loader"] = None
        return out

    monkeypatch.setattr(ge, "set_dark_style", no_font_loader)
    with _app(tk_root, annotate_env) as app:
        assert app.font_loader is None
        assert app.font_style == ("Arial", 12)


def test_init_macos_buttons_are_labels_and_react_to_clicks(
    tk_root, annotate_env, monkeypatch
):
    """On 'Darwin' the buttons become Labels with press/release handlers
    (lines 3469-3486)."""
    import tkinter as tk
    import spacr.gui_elements as ge

    monkeypatch.setattr(ge.platform, "system", lambda: "Darwin")

    with _app(tk_root, annotate_env) as app:
        assert isinstance(app.next_button, tk.Label)
        assert not isinstance(app.next_button, tk.Button)
        assert app.count_button.cget("bg") == "#1a1a1a"
        assert app.count_button.cget("fg") == "white"
        assert str(app.count_button.cget("cursor")) == "hand2"

        before = len(_toplevels(app.root))
        app.count_button.event_generate("<ButtonPress-1>", when="now")
        assert app.count_button.cget("bg") == "#333333"   # on_press

        app.count_button.event_generate("<ButtonRelease-1>", when="now")
        assert app.count_button.cget("bg") == "#1a1a1a"   # on_release restored
        # on_release also fires the wrapped command -> a counts window opened.
        after = _toplevels(app.root)
        assert len(after) == before + 1
        assert any("Class counts" in str(w.title()) for w in after)


# ---------------------------------------------------------------------------
# _int_to_color / _label_to_color
# ---------------------------------------------------------------------------

def test_int_to_color_is_deterministic_hex(tk_root, annotate_env):
    """_int_to_color returns stable, distinct #rrggbb strings (3554-3564)."""
    with _app(tk_root, annotate_env) as app:
        c0 = app._int_to_color(0)
        assert c0 == app._int_to_color(0)
        assert c0.startswith("#") and len(c0) == 7
        int(c0[1:], 16)  # parses as hex
        colors = {app._int_to_color(k) for k in range(12)}
        assert len(colors) == 12, "golden-ratio hues must not collide early"
        # s/v are honoured: s=0 -> a pure gray at v*255
        assert app._int_to_color(3, s=0.0, v=1.0) == "#ffffff"


@pytest.mark.parametrize(
    "val, expected",
    [
        (None, None),
        (0, None),
        (-4, None),
        ("nope", None),
        (1, "#1f77b4"),
        (2, "#d62728"),
        ("2", "#d62728"),
    ],
)
def test_label_to_color_fixed_and_invalid_values(tk_root, annotate_env, val, expected):
    """Fixed starters + the three None-returning guards (3575-3593)."""
    with _app(tk_root, annotate_env) as app:
        assert app._label_to_color(val) == expected


def test_label_to_color_generates_and_caches_beyond_two(tk_root, annotate_env):
    """Class 3+ delegates to _int_to_color and is memoised (3595-3601)."""
    with _app(tk_root, annotate_env) as app:
        assert not hasattr(app, "_class_color_cache")
        c3 = app._label_to_color(3)
        assert c3 == app._int_to_color(0)
        c7 = app._label_to_color(7)
        assert c7 == app._int_to_color(4)
        assert c3 != c7
        # cache populated and reused (cache-hit branch)
        assert app._class_color_cache[3] == c3
        app._class_color_cache[3] = "#000000"
        assert app._label_to_color(3) == "#000000"


# ---------------------------------------------------------------------------
# _embed_figure_in
# ---------------------------------------------------------------------------

def test_embed_figure_replaces_children_with_a_canvas(tk_root, annotate_env):
    """The parent is emptied and a FigureCanvasTkAgg is packed (3606-3619)."""
    import tkinter as tk
    from matplotlib.figure import Figure

    with _app(tk_root, annotate_env) as app:
        holder = _scratch_toplevel(app)
        parent = tk.Frame(holder)
        parent.pack()
        stale = tk.Label(parent, text="stale")
        stale.pack()
        assert parent.winfo_children() == [stale]

        fig = Figure(figsize=(2, 2))
        fig.add_subplot(111).plot([0, 1], [1, 0])
        canvas = app._embed_figure_in(parent, fig)

        assert canvas is not None
        assert canvas.figure is fig
        assert not stale.winfo_exists()
        kids = parent.winfo_children()
        assert len(kids) == 1 and isinstance(kids[0], tk.Canvas)
        holder.destroy()


def test_embed_figure_survives_undestroyable_child_and_missing_backend(
    tk_root, annotate_env, monkeypatch
):
    """A child whose destroy() raises is skipped, and a missing tkagg backend
    degrades to a red error Label returning None (3607-3614)."""
    import tkinter as tk
    from matplotlib.figure import Figure

    class _Undestroyable:
        def destroy(self):
            raise RuntimeError("cannot destroy")

    class _FlakyParent:
        """Real Frame proxy that also reports an undestroyable child."""

        def __init__(self, real, bad):
            object.__setattr__(self, "_real", real)
            object.__setattr__(self, "_bad", bad)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_real"), name)

        def winfo_children(self):
            return [object.__getattribute__(self, "_bad")] + \
                object.__getattribute__(self, "_real").winfo_children()

    real_import = builtins.__import__

    def blocked(name, *a, **kw):
        if name == "matplotlib.backends.backend_tkagg":
            raise ImportError("no tkagg here")
        return real_import(name, *a, **kw)

    with _app(tk_root, annotate_env) as app:
        holder = _scratch_toplevel(app)
        frame = tk.Frame(holder)
        frame.pack()
        doomed = tk.Label(frame, text="doomed")
        doomed.pack()
        parent = _FlakyParent(frame, _Undestroyable())

        monkeypatch.setattr(builtins, "__import__", blocked)
        result = app._embed_figure_in(parent, Figure())
        monkeypatch.undo()

        assert result is None
        assert not doomed.winfo_exists(), "real children must still be destroyed"
        kids = frame.winfo_children()
        assert len(kids) == 1
        assert "Matplotlib not available" in str(kids[0].cget("text"))
        assert str(kids[0].cget("fg")) == "red"
        holder.destroy()


# ---------------------------------------------------------------------------
# open_umap_window
# ---------------------------------------------------------------------------

_UMAP_ENTRY_ORDER = [
    "src", "tables", "row_limit", "n_neighbors", "min_dist", "metric",
    "clustering", "eps", "min_samples", "kmeans_k", "color_by", "dot_size",
    "fig_size", "img_nr", "red_grid", "dbscan_grid", "kmeans_grid",
]


def _open_umap(app):
    """Open the UMAP window; return (win, entries-by-name, buttons-by-text)."""
    import tkinter as tk
    from tkinter import ttk

    before = set(_toplevels(app.root))
    app.open_umap_window()
    app.root.update_idletasks()
    win = [w for w in _toplevels(app.root) if w not in before][0]

    outer = win.winfo_children()[0]
    left = outer.winfo_children()[0]
    entry_widgets = [w for w in left.winfo_children() if isinstance(w, tk.Entry)]
    assert len(entry_widgets) == len(_UMAP_ENTRY_ORDER)
    entries = dict(zip(_UMAP_ENTRY_ORDER, entry_widgets))
    buttons = {str(w.cget("text")): w for w in _walk(win)
               if isinstance(w, ttk.Button)}
    return win, entries, buttons


def _set(entry, value):
    entry.delete(0, "end")
    entry.insert(0, value)


def _fake_core(monkeypatch, umap=None, search=None):
    """Install a stub ``spacr.core`` so the runners never import the real one."""
    mod = types.ModuleType("spacr.core")
    if umap is not None:
        mod.generate_image_umap = umap
    if search is not None:
        mod.reducer_hyperparameter_search = search
    monkeypatch.setitem(sys.modules, "spacr.core", mod)
    return mod


def test_umap_window_defaults_and_run_umap(tk_root, annotate_env, monkeypatch):
    """Opening the window builds the whole form; Run UMAP collects settings,
    embeds the returned figure and reports Done (3623-3794)."""
    import tkinter as tk
    from matplotlib.figure import Figure

    seen = {}

    def gen(settings=None, return_fig=False):
        seen["settings"] = settings
        seen["return_fig"] = return_fig
        return Figure(figsize=(2, 2))

    with _app(tk_root, annotate_env) as app:
        _fake_core(monkeypatch, umap=gen)
        monkeypatch.setattr(threading, "Thread", _SyncThread)

        win, entries, buttons = _open_umap(app)
        assert str(win.title()) == "Image UMAP & Hyperparameter Search"
        assert entries["src"].get() == annotate_env["src"]
        assert entries["tables"].get() == "cell,cytoplasm,nucleus,pathogen"
        assert set(buttons) == {"Run UMAP", "Run Hyperparam Search"}
        # every _row() call produced a caption label
        texts = _label_texts(win)
        for cap in ("src", "tables (csv)", "UMAP n_neighbors", "dot_size",
                    "KMeans grid (JSON list of dicts)"):
            assert cap in texts

        # exercise the blank / unparsable fallbacks in _collect_common_settings
        _set(entries["tables"], "   ")
        _set(entries["row_limit"], "")
        _set(entries["n_neighbors"], "")
        _set(entries["min_dist"], "not-a-float")
        _set(entries["color_by"], "  ")
        _set(entries["fig_size"], "12.5")

        buttons["Run UMAP"].invoke()

        s = seen["settings"]
        assert seen["return_fig"] is True
        assert s["tables"] == ["cell", "cytoplasm", "nucleus", "pathogen"]
        assert s["row_limit"] is None
        assert s["n_neighbors"] == 15          # blank -> `or 15`
        assert s["min_dist"] == 0.1            # unparsable -> default
        assert s["metric"] == "euclidean"
        assert s["clustering"] == "DBSCAN"
        assert s["eps"] == 0.5
        assert s["min_samples"] == 5
        assert s["image_nr"] == 200
        assert s["dot_size"] == 6
        assert s["figuresize"] == 12.5
        assert s["plot_images"] is False
        assert s["color_by"] is None
        assert s["kmeans_k"] == 8
        assert s["reduction_method"] == "umap"
        assert s["n_jobs"] >= 1
        assert "Done." in _label_texts(win)

        # the figure got embedded into the right-hand pane
        right = win.winfo_children()[0].winfo_children()[1]
        assert any(isinstance(w, tk.Canvas) for w in right.winfo_children())


def test_umap_window_parses_explicit_values(tk_root, annotate_env, monkeypatch):
    """Non-default entries flow through _collect_common_settings verbatim."""
    import tkinter as tk
    from matplotlib.figure import Figure

    seen = {}

    def gen(settings=None, return_fig=False):
        seen["settings"] = settings
        return Figure()

    with _app(tk_root, annotate_env) as app:
        _fake_core(monkeypatch, umap=gen)
        monkeypatch.setattr(threading, "Thread", _SyncThread)

        win, entries, buttons = _open_umap(app)
        _set(entries["tables"], " cell , nucleus , ")
        _set(entries["row_limit"], "10.0")
        _set(entries["n_neighbors"], "20")
        _set(entries["min_dist"], "0.25")
        _set(entries["metric"], "  cosine  ")
        _set(entries["eps"], "0.9")
        _set(entries["min_samples"], "3")
        _set(entries["kmeans_k"], "4")
        _set(entries["color_by"], "columnID")
        _set(entries["dot_size"], "9")
        _set(entries["img_nr"], "50")
        entries["clustering"].set("kmeans")

        checkbuttons = [w for w in _walk(win) if isinstance(w, tk.Checkbutton)]
        assert len(checkbuttons) == 1
        checkbuttons[0].invoke()  # plot_images -> True

        buttons["Run UMAP"].invoke()

        s = seen["settings"]
        assert s["tables"] == ["cell", "nucleus"]
        assert s["row_limit"] == 10
        assert s["n_neighbors"] == 20
        assert s["min_dist"] == 0.25
        assert s["metric"] == "cosine"
        assert s["clustering"] == "KMEANS"
        assert s["eps"] == 0.9
        assert s["min_samples"] == 3
        assert s["kmeans_k"] == 4
        assert s["color_by"] == "columnID"
        assert s["dot_size"] == 9
        assert s["image_nr"] == 50
        assert s["plot_images"] is True


def test_umap_hyperparam_search_uses_entered_grids(tk_root, annotate_env, monkeypatch):
    """Run Hyperparam Search literal-evals the three grid entries (3796-3823)."""
    from matplotlib.figure import Figure

    seen = {}

    def search(settings=None, reduction_params=None, dbscan_params=None,
               kmeans_params=None, show=None, return_fig=None):
        seen.update(dict(settings=settings, reduction_params=reduction_params,
                         dbscan_params=dbscan_params, kmeans_params=kmeans_params,
                         show=show, return_fig=return_fig))
        return Figure()

    with _app(tk_root, annotate_env) as app:
        _fake_core(monkeypatch, search=search)
        monkeypatch.setattr(threading, "Thread", _SyncThread)

        win, entries, buttons = _open_umap(app)
        buttons["Run Hyperparam Search"].invoke()

        assert seen["show"] is False and seen["return_fig"] is True
        assert seen["reduction_params"] == [
            {"n_neighbors": 10, "min_dist": 0.05},
            {"n_neighbors": 15, "min_dist": 0.1},
            {"n_neighbors": 30, "min_dist": 0.3},
        ]
        assert seen["dbscan_params"] == [
            {"eps": 0.3, "min_samples": 5},
            {"eps": 0.5, "min_samples": 5},
            {"eps": 0.7, "min_samples": 3},
        ]
        assert seen["kmeans_params"] == [{"n_clusters": 6}, {"n_clusters": 8},
                                         {"n_clusters": 10}]
        assert "Done." in _label_texts(win)


def test_umap_hyperparam_search_falls_back_on_bad_grids(
    tk_root, annotate_env, monkeypatch
):
    """Blank / unparsable grid entries fall back to the built-in defaults."""
    from matplotlib.figure import Figure

    seen = {}

    def search(settings=None, reduction_params=None, dbscan_params=None,
               kmeans_params=None, show=None, return_fig=None):
        seen.update(dict(reduction_params=reduction_params,
                         dbscan_params=dbscan_params,
                         kmeans_params=kmeans_params))
        return Figure()

    with _app(tk_root, annotate_env) as app:
        _fake_core(monkeypatch, search=search)
        monkeypatch.setattr(threading, "Thread", _SyncThread)

        win, entries, buttons = _open_umap(app)
        _set(entries["red_grid"], "   ")            # blank  -> []
        _set(entries["dbscan_grid"], "{{not python")  # unparsable -> []
        _set(entries["kmeans_grid"], "")            # blank  -> []
        _set(entries["kmeans_k"], "11")
        buttons["Run Hyperparam Search"].invoke()

        assert seen["reduction_params"] == [{"n_neighbors": 15, "min_dist": 0.1}]
        assert seen["dbscan_params"] == [{"eps": 0.5, "min_samples": 5}]
        assert seen["kmeans_params"] == [{"n_clusters": 11}]


def test_umap_runners_report_errors_in_the_status_label(
    tk_root, annotate_env, monkeypatch
):
    """Exceptions raised inside either worker land in the status label
    instead of escaping (3792-3793 / 3824-3825)."""
    def boom(*a, **kw):
        raise RuntimeError("umap exploded")

    with _app(tk_root, annotate_env) as app:
        _fake_core(monkeypatch, umap=boom, search=boom)
        monkeypatch.setattr(threading, "Thread", _SyncThread)

        win, entries, buttons = _open_umap(app)
        buttons["Run UMAP"].invoke()
        assert "Error: umap exploded" in _label_texts(win)

        buttons["Run Hyperparam Search"].invoke()
        assert "Error: umap exploded" in _label_texts(win)


# ---------------------------------------------------------------------------
# _poll_save_status
# ---------------------------------------------------------------------------

def test_poll_save_status_spinner_then_saved_then_idle(tk_root, annotate_env):
    """The status text cycles spinner -> saved -> blank (3833-3848)."""
    with _app(tk_root, annotate_env) as app:
        app._unsaved_batches = 2
        app._poll_save_status()
        text = str(app.status_label.cget("text"))
        assert "Saving" in text and "pending=2" in text
        assert text[0] in app._spinner_frames
        first_idx = app._spinner_idx

        app._poll_save_status()
        assert app._spinner_idx == (first_idx + 1) % len(app._spinner_frames)
        assert str(app.status_label.cget("text"))[0] == \
            app._spinner_frames[app._spinner_idx]

        # pending_updates alone also counts as "saving"
        app._unsaved_batches = 0
        app.pending_updates = {"a.png": 1}
        app._poll_save_status()
        assert "pending=0" in str(app.status_label.cget("text"))

        # nothing outstanding but something was saved before -> tick message
        app.pending_updates = {}
        app._last_save_ts = 123.0
        app._poll_save_status()
        assert str(app.status_label.cget("text")) == "✓ All changes saved"

        # never saved anything -> empty label
        app._last_save_ts = None
        app._poll_save_status()
        assert str(app.status_label.cget("text")) == ""


# ---------------------------------------------------------------------------
# open_settings_window
# ---------------------------------------------------------------------------

def _open_settings(app):
    """Open the settings Toplevel; return (win, entries-by-key, apply_button)."""
    import tkinter as tk
    from tkinter import ttk
    from spacr.gui_elements import spacrButton

    before = set(_toplevels(app.root))
    app.open_settings_window()
    app.root.update_idletasks()
    win = [w for w in _toplevels(app.root) if w not in before][0]

    frame = win.winfo_children()[0]
    labels_by_row = {}
    for w in frame.winfo_children():
        info = w.grid_info()
        if isinstance(w, tk.Label) and info:
            labels_by_row[int(info["row"])] = str(w.cget("text"))
    entries = {}
    for w in frame.winfo_children():
        info = w.grid_info()
        if isinstance(w, ttk.Entry) and info:
            key = labels_by_row.get(int(info["row"]), "")
            key = key.rstrip(":").strip().lower().replace(" ", "_")
            entries[key] = w
    apply_button = [w for w in _walk(win) if isinstance(w, spacrButton)][0]
    return win, entries, apply_button


def test_settings_window_prefills_current_state(tk_root, annotate_env):
    """Entries are seeded from the live attributes (4008-4041)."""
    with _app(tk_root, annotate_env, image_type="cell_png",
              channels=["r", "g"], percentiles=(2, 98),
              normalize_channels=["r", "", None], outline=["b"],
              annotation_column="annotate") as app:
        win, entries, _ = _open_settings(app)
        assert str(win.title()) == "Modify Annotation Settings"
        assert entries["image_type"].get() == "cell_png"
        assert entries["channels"].get() == "r,g"
        assert entries["img_size"].get() == "1000,1000"
        assert entries["annotation_column"].get() == "annotate"
        assert entries["percentiles"].get() == "2,98"
        assert entries["normalize_channels"].get() == "r"
        assert entries["outline"].get() == "b"
        assert entries["src"].get() == annotate_env["src"]
        assert entries["measurement"].get() == ""
        assert entries["threshold"].get() == ""
        assert entries["threshold_direction"].get() == "higher"
        assert entries["object_size"].get() == "0,0"
        win.destroy()


@pytest.mark.parametrize(
    "measurement, threshold, direction, exp_m, exp_t, exp_d",
    [
        (None, None, "", "", "", ""),
        ("cell_area", 5, "higher", "cell_area", "5", "higher"),
        (["a", "b"], [1, 2], ["lower", "higher"], "a,b", "[1, 2]", "lower,higher"),
        ([["a"], ["b", "c"]], ("q3",), None,
         '[["a"], ["b", "c"]]', '["q3"]', ""),
        (17, 2.5, "lower", "17", "2.5", "lower"),
    ],
)
def test_settings_window_serializes_every_filter_shape(
    tk_root, annotate_env, measurement, threshold, direction, exp_m, exp_t, exp_d
):
    """_serialize_measurement / _threshold / _direction cover all shapes
    (3983-4006)."""
    with _app(tk_root, annotate_env) as app:
        app.measurement = measurement
        app.threshold = threshold
        app.threshold_direction = direction
        win, entries, _ = _open_settings(app)
        assert entries["measurement"].get() == exp_m
        assert entries["threshold"].get() == exp_t
        assert entries["threshold_direction"].get() == exp_d
        win.destroy()


def test_settings_window_handles_empty_channel_lists(tk_root, annotate_env):
    """The `else ''` arms of the channels/outline joins (4010, 4020)."""
    with _app(tk_root, annotate_env, channels=None, outline=None,
              image_type=None, normalize_channels=None) as app:
        win, entries, _ = _open_settings(app)
        assert entries["channels"].get() == ""
        assert entries["outline"].get() == ""
        assert entries["image_type"].get() == ""
        assert entries["normalize_channels"].get() == ""
        win.destroy()


def test_settings_window_adds_missing_threshold_direction_row(
    tk_root, annotate_env, monkeypatch
):
    """When generate_annotate_fields omits threshold_direction the window
    builds the row itself (3967-3980)."""
    import tkinter as tk
    from tkinter import ttk
    from spacr import gui_utils

    def packed_fields(frame):
        """Same contract as generate_annotate_fields but pack-managed and
        without a threshold_direction entry."""
        d = {}
        for key in ("src", "image_type", "measurement"):
            row = tk.Frame(frame)
            row.pack(fill="x")
            tk.Label(row, text=f"{key}:").pack(side="left")
            entry = ttk.Entry(row)
            entry.pack(side="left")
            d[key] = {"entry": entry, "value": ""}
        return d

    monkeypatch.setattr(gui_utils, "generate_annotate_fields", packed_fields)

    with _app(tk_root, annotate_env) as app:
        app.threshold_direction = "lower"
        before = set(_toplevels(app.root))
        app.open_settings_window()
        app.root.update_idletasks()
        win = [w for w in _toplevels(app.root) if w not in before][0]

        assert "threshold_direction" in _label_texts(win)
        rows = [w for w in _walk(win)
                if isinstance(w, tk.Frame)
                and any(isinstance(c, tk.Label)
                        and str(c.cget("text")) == "threshold_direction"
                        for c in w.winfo_children())]
        assert len(rows) == 1
        entry = [c for c in rows[0].winfo_children() if isinstance(c, tk.Entry)]
        assert len(entry) == 1
        # the fill loop seeded it from the live attribute
        assert entry[0].get() == "lower"
        win.destroy()


@pytest.mark.xfail(
    strict=True,
    reason="BUG: the fallback threshold_direction row is pack()ed into "
           "settings_frame, whose other children are grid()ed, so the "
           "fallback raises TclError instead of adding the row",
)
def test_settings_window_fallback_row_works_with_the_real_field_builder(
    tk_root, annotate_env, monkeypatch
):
    """The defensive 'threshold_direction missing' arm must work with the
    real (grid-managed) generate_annotate_fields."""
    from spacr import gui_utils

    real = gui_utils.generate_annotate_fields

    def without_direction(frame):
        d = real(frame)
        d.pop("threshold_direction", None)
        return d

    monkeypatch.setattr(gui_utils, "generate_annotate_fields", without_direction)

    with _app(tk_root, annotate_env) as app:
        before = set(_toplevels(app.root))
        app.open_settings_window()          # must not raise
        win = [w for w in _toplevels(app.root) if w not in before][0]
        assert "threshold_direction" in _label_texts(win)
        win.destroy()


def test_settings_window_label_lookup_failure_paths(
    tk_root, annotate_env, monkeypatch
):
    """_find_label_for: no siblings label, raising master, raising cget and
    the multi-label preference rules (3934-3950)."""
    import tkinter as tk
    from tkinter import ttk
    from spacr import gui_utils

    class _BadLabel(tk.Label):
        """Looks like a Label to isinstance() but blows up on cget()."""

        def __init__(self):  # deliberately skips widget creation
            pass

        def cget(self, key):
            raise RuntimeError("no such option")

    class _FakeMaster:
        def __init__(self, kids):
            self._kids = kids

        def winfo_children(self):
            return list(self._kids)

    class _ProxyEntry:
        def __init__(self, real, master=None, raise_on_master=False):
            object.__setattr__(self, "_real", real)
            object.__setattr__(self, "_master", master)
            object.__setattr__(self, "_raise", raise_on_master)

        def __getattr__(self, name):
            if name == "master":
                if object.__getattribute__(self, "_raise"):
                    raise RuntimeError("detached widget")
                return object.__getattribute__(self, "_master")
            return getattr(object.__getattribute__(self, "_real"), name)

    holder = {}

    def crafted_fields(frame):
        d = {}

        # 1. entry alone in a label-less frame -> _find_label_for -> None
        f1 = tk.Frame(frame)
        f1.pack()
        d["db_path"] = {"entry": ttk.Entry(f1), "value": ""}

        # 2. entry whose .master access raises -> except branch -> None
        f2 = tk.Frame(frame)
        f2.pack()
        d["src"] = {"entry": _ProxyEntry(ttk.Entry(f2), raise_on_master=True),
                    "value": ""}

        # 3. exactly one sibling label -> returned directly
        f3 = tk.Frame(frame)
        f3.pack()
        only = tk.Label(f3, text="Outline:")
        only.pack()
        d["outline"] = {"entry": ttk.Entry(f3), "value": ""}
        holder["only"] = only

        # 4. several labels, one raising on cget, one matching the key
        f4 = tk.Frame(frame)
        f4.pack()
        good = tk.Label(f4, text="Measurement:")
        good.pack()
        d["measurement"] = {
            "entry": _ProxyEntry(ttk.Entry(f4),
                                 master=_FakeMaster([_BadLabel(), good])),
            "value": "",
        }
        holder["good"] = good

        # 5. several labels, none matching -> first one wins
        f5 = tk.Frame(frame)
        f5.pack()
        first = tk.Label(f5, text="zzz")
        first.pack()
        tk.Label(f5, text="yyy").pack()
        d["channels"] = {"entry": ttk.Entry(f5), "value": ""}
        holder["first"] = first

        # 6. a field with neither a current value nor a tooltip -> skipped
        f6 = tk.Frame(frame)
        f6.pack()
        untipped = ttk.Entry(f6)
        untipped.insert(0, "untouched")
        d["rows"] = {"entry": untipped, "value": ""}
        holder["untipped"] = untipped
        return d

    monkeypatch.setattr(gui_utils, "generate_annotate_fields", crafted_fields)

    with _app(tk_root, annotate_env, channels=["r"]) as app:
        before = set(_toplevels(app.root))
        app.open_settings_window()
        app.root.update_idletasks()
        win = [w for w in _toplevels(app.root) if w not in before][0]

        # every entry was still pre-filled from current_settings
        frame = win.winfo_children()[0]
        entries = [w for w in _walk(frame) if isinstance(w, ttk.Entry)]
        vals = {e.get() for e in entries}
        assert annotate_env["db_path"] in vals
        assert "r" in vals

        # a tooltip was attached to the resolvable labels (bind list grew)
        assert holder["only"].bind("<Enter>")
        assert holder["good"].bind("<Enter>")
        assert holder["first"].bind("<Enter>")
        # an unknown key has no tooltip and no current value -> left alone
        assert holder["untipped"].get() == "untouched"
        assert not holder["untipped"].bind("<Enter>")
        win.destroy()


def test_settings_window_local_tooltip_class_show_and_hide(tk_root, annotate_env):
    """The window's inline _ToolTip helper creates and destroys its Toplevel
    (3856-3889).  Captured through __build_class__ because the class is
    local to open_settings_window."""
    import tkinter as tk

    captured = {}
    real_build_class = builtins.__build_class__

    def spy(func, name, *bases, **kwds):
        cls = real_build_class(func, name, *bases, **kwds)
        if name == "_ToolTip":
            captured["cls"] = cls
        return cls

    with _app(tk_root, annotate_env) as app:
        builtins.__build_class__ = spy
        try:
            app.open_settings_window()
        finally:
            builtins.__build_class__ = real_build_class
        app.root.update_idletasks()

        ToolTip = captured["cls"]
        holder = _scratch_toplevel(app)
        host = tk.Label(holder, text="host")
        host.pack()
        app.root.update_idletasks()

        tip = ToolTip(host, "helpful text")
        assert tip.tipwindow is None
        tip._show()
        assert isinstance(tip.tipwindow, tk.Toplevel)
        made = tip.tipwindow
        assert made.winfo_children()[0].cget("text") == "helpful text"

        tip._show()                      # already shown -> early return
        assert tip.tipwindow is made

        tip._hide()
        assert tip.tipwindow is None
        assert not made.winfo_exists()
        tip._hide()                      # nothing to hide -> no-op
        assert tip.tipwindow is None

        blank = ToolTip(host, "")
        blank._show()                    # empty text -> early return
        assert blank.tipwindow is None
        holder.destroy()


def test_settings_apply_parses_and_pushes_every_field(tk_root, annotate_env):
    """apply_new_settings parses the whole form and hands it to
    update_settings, then closes the window (4091-4204)."""
    with _app(tk_root, annotate_env) as app:
        win, entries, apply_button = _open_settings(app)
        refiltered = _stub_prefilter(app)

        _set(entries["channels"], " R , b , ")
        _set(entries["img_size"], "900,900")
        _set(entries["percentiles"], "3,97")
        _set(entries["normalize_channels"], "r,x,G")     # x dropped
        _set(entries["outline"], "b,zzz")                # zzz dropped
        _set(entries["object_size"], "500;20")           # swapped -> (20, 500)
        _set(entries["outline_threshold_factor"], "1,5")  # comma decimal
        _set(entries["outline_sigma"], "2,5")
        _set(entries["edge_thickness"], "0,5")
        _set(entries["edge_transparency"], "250")        # clamped to 100
        _set(entries["edge_image"], "yes")
        _set(entries["measurement"], "cell_area")
        _set(entries["threshold"], "q3")
        _set(entries["threshold_direction"], "Lower")

        apply_button.command()

        assert app.channels == ["r", "b"]
        assert app.image_size == (900, 900)
        assert app.percentiles == [3, 97]
        assert app.normalize_channels == ["r", "g"]
        assert app.outline == ["b"]
        assert app.object_size == (20, 500)
        assert app.outline_threshold_factor == 1.5
        assert app.outline_sigma == 2.5
        assert app.edge_thickness == 0.5
        assert app.edge_transparency == 100.0
        assert app.edge_image is True
        assert app.measurement == "cell_area"
        assert app.threshold == "q3"
        assert app.db_path == os.path.join(
            annotate_env["src"], "measurements", "measurements.db")
        assert not win.winfo_exists(), "the settings window must close"
        # image_size changed -> the grid was rebuilt to the new tile size
        assert len(app.labels) == app.grid_rows * app.grid_cols
        assert refiltered, "update_settings must re-run the prefilter"


@pytest.mark.xfail(
    strict=True,
    reason="BUG: 'threshold_direction' is missing from update_settings' "
           "allowed_attributes, so the settings window silently discards it",
)
def test_settings_apply_persists_threshold_direction(tk_root, annotate_env):
    """Editing threshold_direction in the settings window must take effect."""
    with _app(tk_root, annotate_env, threshold_direction="higher") as app:
        win, entries, apply_button = _open_settings(app)
        _stub_prefilter(app)
        _set(entries["measurement"], "cell_area")
        _set(entries["threshold"], "5")
        _set(entries["threshold_direction"], "lower")
        apply_button.command()
        assert app.threshold_direction == "lower"


def test_settings_apply_blank_fields_use_documented_defaults(tk_root, annotate_env):
    """Blank entries fall back to their defaults (4103, 4109, 4116, 4136,
    4140, 4144, 4148)."""
    with _app(tk_root, annotate_env) as app:
        win, entries, apply_button = _open_settings(app)

        _set(entries["percentiles"], "")
        _set(entries["normalize_channels"], "")
        _set(entries["outline"], "")
        _set(entries["object_size"], "")
        _set(entries["outline_threshold_factor"], "")
        _set(entries["outline_sigma"], "")
        _set(entries["edge_thickness"], "")
        _set(entries["edge_transparency"], "")
        _set(entries["edge_image"], "no")
        _set(entries["measurement"], "")
        _set(entries["threshold"], "none")
        _set(entries["threshold_direction"], "")

        apply_button.command()

        assert app.percentiles == [1, 99]
        assert app.outline is None            # [] -> ignored by update_settings
        assert app.object_size == (0, 0)
        assert app.outline_threshold_factor == 1.0
        assert app.outline_sigma == 1.0
        assert app.edge_thickness == 1.0
        assert app.edge_transparency == 0.0
        assert app.edge_image is False
        assert not win.winfo_exists()


def test_settings_apply_object_size_and_transparency_garbage(tk_root, annotate_env):
    """Unparsable object_size parts become 0, a single value is padded, and a
    non-numeric transparency falls back to 0 (4123-4126, 4152-4153)."""
    with _app(tk_root, annotate_env) as app:
        win, entries, apply_button = _open_settings(app)
        _set(entries["object_size"], "abc")
        _set(entries["edge_transparency"], "not-a-number")
        apply_button.command()
        assert app.object_size == (0, 0)
        assert app.edge_transparency == 0.0

    with _app(tk_root, annotate_env) as app:
        win, entries, apply_button = _open_settings(app)
        _set(entries["object_size"], "42")     # single value -> padded
        apply_button.command()
        assert app.object_size == (42, 0)


def _record_update_settings(app):
    """Capture the kwargs apply_new_settings forwards to update_settings."""
    seen = {}

    def recorder(**kwargs):
        seen.update(kwargs)

    app.update_settings = recorder
    return seen


@pytest.mark.parametrize(
    "m_txt, t_txt, d_txt, exp_m, exp_t, exp_d",
    [
        ("cell_area", "100", "higher", "cell_area", 100, "higher"),
        ("a, b ", "q2,3.5", "Lower, Higher", ["a", "b"], ["q2", 3.5],
         ["lower", "higher"]),
        ('[["a"],["b","c"]]', '[100,"q5"]', '["lower","higher"]',
         [["a"], ["b", "c"]], [100, "q5"], ["lower", "higher"]),
        (",,", ",,", ",,", None, [], None),
        ("area", "2,5", "higher", "area", [2, 5], "higher"),
        ("", "none", "", None, None, None),
    ],
)
def test_settings_apply_filter_parsers(
    tk_root, annotate_env, m_txt, t_txt, d_txt, exp_m, exp_t, exp_d
):
    """_parse_measurement / _parse_threshold / _parse_direction across every
    accepted shape (4046-4089)."""
    with _app(tk_root, annotate_env) as app:
        win, entries, apply_button = _open_settings(app)
        seen = _record_update_settings(app)
        _set(entries["measurement"], m_txt)
        _set(entries["threshold"], t_txt)
        _set(entries["threshold_direction"], d_txt)
        apply_button.command()
        assert seen["measurement"] == exp_m
        assert seen["threshold"] == exp_t
        assert seen["threshold_direction"] == exp_d
        assert not win.winfo_exists()


def test_settings_apply_unparsable_threshold_disables_filtering(
    tk_root, annotate_env, capsys
):
    """A garbage threshold is caught and turns filtering off (4163-4167)."""
    with _app(tk_root, annotate_env) as app:
        win, entries, apply_button = _open_settings(app)
        seen = _record_update_settings(app)
        _set(entries["measurement"], "cell_area")
        _set(entries["threshold"], "not-a-threshold")
        _set(entries["threshold_direction"], "higher")
        apply_button.command()
        out = capsys.readouterr().out
        assert "could not parse measurement/threshold" in out
        assert seen["measurement"] is None
        assert seen["threshold"] is None
        assert seen["threshold_direction"] is None
        assert not win.winfo_exists()


def test_settings_apply_bad_json_measurement_is_caught(tk_root, annotate_env, capsys):
    """A malformed JSON measurement hits the JSONDecodeError arm."""
    with _app(tk_root, annotate_env) as app:
        win, entries, apply_button = _open_settings(app)
        seen = _record_update_settings(app)
        _set(entries["measurement"], '[[unquoted')
        _set(entries["threshold"], "1")
        apply_button.command()
        assert "could not parse measurement/threshold" in capsys.readouterr().out
        assert seen["measurement"] is None
        assert seen["threshold"] is None
        assert not win.winfo_exists()


# ---------------------------------------------------------------------------
# _ensure_annotation_column
# ---------------------------------------------------------------------------

def test_ensure_annotation_column_noop_without_a_column(tk_root, annotate_env):
    """A falsy annotation_column short-circuits (line 4211)."""
    with _app(tk_root, annotate_env, annotation_column="annotate") as app:
        with sqlite3.connect(annotate_env["db_path"]) as con:
            before = [r[1] for r in con.execute(
                'PRAGMA table_info("png_list")').fetchall()]
        app.annotation_column = None
        assert app._ensure_annotation_column() is None
        with sqlite3.connect(annotate_env["db_path"]) as con:
            after = [r[1] for r in con.execute(
                'PRAGMA table_info("png_list")').fetchall()]
        assert after == before
        app.annotation_column = "annotate"


def test_ensure_annotation_column_swallows_duplicate_column_error(tk_root, tmp_path):
    """PRAGMA is case-sensitive but ALTER TABLE is not: the resulting
    OperationalError is swallowed (4223-4224)."""
    (tmp_path / "measurements").mkdir()
    (tmp_path / "images").mkdir()
    db_path = tmp_path / "measurements" / "measurements.db"
    _make_db(str(db_path), n=2, extra_cols={"Annotate": [None, None]})

    env = {"src": str(tmp_path), "db_path": str(db_path)}
    with sqlite3.connect(str(db_path)) as con:
        cols_before = [r[1] for r in con.execute(
            'PRAGMA table_info("png_list")').fetchall()]
    assert "Annotate" in cols_before and "annotate" not in cols_before

    # Constructing with the lowercase name makes ALTER TABLE raise
    # "duplicate column name" - which must be swallowed, not propagated.
    with _app(tk_root, env, annotation_column="annotate") as app:
        assert app.annotation_column == "annotate"

    with sqlite3.connect(str(db_path)) as con:
        cols_after = [r[1] for r in con.execute(
            'PRAGMA table_info("png_list")').fetchall()]
    assert cols_after == cols_before, "no column may have been added"


# ---------------------------------------------------------------------------
# update_settings
# ---------------------------------------------------------------------------

def test_update_settings_ignores_unknown_and_none(tk_root, annotate_env):
    """Only whitelisted, non-None keys are applied (4251-4252)."""
    with _app(tk_root, annotate_env) as app:
        app.update_settings(not_a_setting=123, image_type=None)
        assert not hasattr(app, "not_a_setting")
        assert app.image_type is None


@pytest.mark.parametrize(
    "value, expected",
    [
        (["R", " g ", None, "zz"], ["r", "g"]),
        (("b",), ["b"]),
        ("r, B , q", ["r", "b"]),
        ("qq", None),
        ([], None),
        (17, None),
    ],
)
def test_update_settings_normalize_channels_coercion(
    tk_root, annotate_env, value, expected
):
    """normalize_channels accepts lists, tuples, strings, junk (4254-4264)."""
    with _app(tk_root, annotate_env) as app:
        app.update_settings(normalize_channels=value)
        assert app.normalize_channels == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (["R", " g ", None, "zz"], ["r", "g"]),
        (("b",), ["b"]),
        ("r, B , q", ["r", "b"]),
        ("qq", None),
        (17, None),
    ],
)
def test_update_settings_outline_coercion(tk_root, annotate_env, value, expected):
    """outline accepts lists, tuples, strings, junk (4266-4274)."""
    with _app(tk_root, annotate_env) as app:
        app.update_settings(outline=value)
        assert app.outline == expected


def test_update_settings_numeric_coercions(tk_root, annotate_env):
    """floats, the transparency clamp and the bool cast (4276-4292)."""
    with _app(tk_root, annotate_env) as app:
        app.update_settings(outline_threshold_factor="2.5", outline_sigma=3,
                            edge_thickness="0.25", edge_transparency="60",
                            edge_image=1)
        assert app.outline_threshold_factor == 2.5
        assert app.outline_sigma == 3.0
        assert app.edge_thickness == 0.25
        assert app.edge_transparency == 60.0
        assert app.edge_image is True

        app.update_settings(edge_transparency=1000)
        assert app.edge_transparency == 100.0
        app.update_settings(edge_transparency=-5)
        assert app.edge_transparency == 0.0
        app.update_settings(edge_transparency="nonsense")
        assert app.edge_transparency == 0.0


@pytest.mark.parametrize(
    "value, expected",
    [
        ("10;500", (10, 500)),
        ("500,10", (10, 500)),      # swapped
        ("bad,7", (0, 7)),
        ("42", (42, 0)),
        ([5, 9], (5, 9)),
        ((9, 5), (5, 9)),
        ((7,), (7, 0)),
        (17, (0, 0)),
    ],
)
def test_update_settings_object_size_shapes(tk_root, annotate_env, value, expected):
    """object_size normalisation for every accepted input (4294-4319)."""
    with _app(tk_root, annotate_env) as app:
        app.update_settings(object_size=value)
        assert app.object_size == expected


@pytest.mark.xfail(
    strict=True,
    reason="BUG: update_settings(object_size='' | []) takes the "
           "`v in (None, '', [])` branch, which sets v but never mn/mx, so "
           "the following `if mn and mx` raises UnboundLocalError",
)
@pytest.mark.parametrize("value", ["", []])
def test_update_settings_object_size_empty_means_no_bounds(
    tk_root, annotate_env, value
):
    """An empty object_size documents 'no bounds' and must yield (0, 0)."""
    with _app(tk_root, annotate_env) as app:
        app.update_settings(object_size=value)
        assert app.object_size == (0, 0)


def test_update_settings_annotation_column_creates_it(tk_root, annotate_env):
    """A new annotation_column is added to png_list (4324-4325)."""
    with _app(tk_root, annotate_env, annotation_column="annotate") as app:
        app.update_settings(annotation_column="second_pass")
        assert app.annotation_column == "second_pass"
    with sqlite3.connect(annotate_env["db_path"]) as con:
        cols = [r[1] for r in con.execute(
            'PRAGMA table_info("png_list")').fetchall()]
    assert "second_pass" in cols


@pytest.mark.parametrize("value, expected", [
    ([700, 700], (700, 700)),
    (700, (700, 700)),
    ((700, 800), (700, 800)),
])
def test_update_settings_image_size_shapes_rebuild_grid(
    tk_root, annotate_env, value, expected
):
    """image_size list/int/tuple all rebuild the grid (4327-4338, 4373-4387)."""
    with _app(tk_root, annotate_env) as app:
        old_labels = list(app.labels)
        app.update_settings(image_size=value)
        assert app.image_size == expected
        assert len(app.labels) == app.grid_rows * app.grid_cols
        assert all(not lab.winfo_exists() for lab in old_labels)
        assert all(lab.winfo_exists() for lab in app.labels)


def test_update_settings_rejects_bad_image_size(tk_root, annotate_env):
    """A non list/int/tuple image_size raises (line 4335)."""
    with _app(tk_root, annotate_env) as app:
        with pytest.raises(ValueError, match="Invalid image size"):
            app.update_settings(image_size="700")
        app.image_size = (1000, 1000)  # restore for teardown


def test_update_settings_src_change_resets_pagination(tk_root, annotate_env, tmp_path):
    """A new src clears the path map and rewinds to page 0 (4340-4342)."""
    other = tmp_path / "other_src"
    other.mkdir()
    with _app(tk_root, annotate_env) as app:
        app.index = 12
        app.adjusted_to_original_paths["x"] = "y"
        app.update_settings(src=str(other))
        assert app.src == str(other)
        assert app.adjusted_to_original_paths == {}
        assert app.index == 0


def test_update_settings_db_path_change_restarts_worker(
    tk_root, annotate_env, tmp_path
):
    """Switching DB flushes pending updates and respawns the writer thread
    (4344-4361)."""
    second = tmp_path / "second"
    (second / "measurements").mkdir(parents=True)
    (second / "images").mkdir()
    db2 = second / "measurements" / "measurements.db"
    _make_db(str(db2), n=2)

    with _app(tk_root, annotate_env, annotation_column="annotate") as app:
        first_thread = app.db_update_thread
        target = annotate_env["paths"][0]
        app.pending_updates[target] = 2
        app._last_save_ts = 999.0

        app.update_settings(db_path=str(db2))

        assert app.db_path == str(db2)
        assert app.pending_updates == {}
        assert app.db_update_thread is not first_thread
        assert app.db_update_thread.is_alive()
        assert not first_thread.is_alive()
        assert app.terminate is False
        assert app.worker_busy is False
        assert app._last_save_ts is None
        # the pending annotation was written to the ORIGINAL database
        with sqlite3.connect(annotate_env["db_path"]) as con:
            got = con.execute(
                'SELECT "annotate" FROM "png_list" WHERE png_path = ?',
                (target,)).fetchone()
        assert got == (2,)
        # and the new DB got its annotation column created
        with sqlite3.connect(str(db2)) as con:
            cols = [r[1] for r in con.execute(
                'PRAGMA table_info("png_list")').fetchall()]
        assert "annotate" in cols


def test_update_settings_db_switch_survives_unjoinable_thread(
    tk_root, annotate_env, tmp_path
):
    """A writer thread that refuses to join must not abort the DB switch
    (4352-4356)."""
    third = tmp_path / "third"
    (third / "measurements").mkdir(parents=True)
    db3 = third / "measurements" / "measurements.db"
    _make_db(str(db3), n=1)

    class _UnjoinableThread:
        def join(self, timeout=None):
            raise RuntimeError("cannot join this one")

        def is_alive(self):
            return True

    with _app(tk_root, annotate_env, annotation_column="annotate") as app:
        real_thread = app.db_update_thread
        app.db_update_thread = _UnjoinableThread()

        app.update_settings(db_path=str(db3))

        assert app.db_path == str(db3)
        assert isinstance(app.db_update_thread, threading.Thread)
        assert app.db_update_thread.is_alive()
        # the original worker still shut down cleanly on the sentinel
        real_thread.join(timeout=10)
        assert not real_thread.is_alive()


def test_update_settings_reloads_page_and_clamps_index(tk_root, annotate_env):
    """After a real change the page is re-filtered and reloaded (4363-4368)."""
    with _app(tk_root, annotate_env) as app:
        app.filtered_paths_annotations = []
        app.images = {}
        app.update_settings(image_type="cell")
        assert app.image_type == "cell"
        assert len(app.filtered_paths_annotations) == 3
        assert app.index == 0
        assert len(app.images) == 3, "load_images must have painted the tiles"

        # nothing matches the substring -> an empty, but valid, page
        app.update_settings(image_type="no_such_prefix")
        assert app.filtered_paths_annotations == []
        assert app.images == {}


# ---------------------------------------------------------------------------
# recreate_image_grid / update_display / swich_back_annotation_column
# ---------------------------------------------------------------------------

def test_recreate_image_grid_replaces_all_labels(tk_root, annotate_env):
    """recreate_image_grid destroys the old labels and lays out new ones."""
    with _app(tk_root, annotate_env) as app:
        old = list(app.labels)
        app.grid_rows, app.grid_cols = 2, 4
        app.recreate_image_grid()
        assert len(app.labels) == 8
        assert all(not lab.winfo_exists() for lab in old)
        positions = {(int(lab.grid_info()["row"]), int(lab.grid_info()["column"]))
                     for lab in app.labels}
        assert positions == {(r, c) for r in range(2) for c in range(4)}


def test_update_display_refilters_and_repaints(tk_root, annotate_env):
    """update_display re-runs the prefilter and reloads the grid (4391-4392)."""
    with _app(tk_root, annotate_env) as app:
        app.filtered_paths_annotations = []
        app.images = {}
        app.update_display()
        assert len(app.filtered_paths_annotations) == 3
        assert len(app.images) == 3


def test_swich_back_annotation_column_restores_original(tk_root, annotate_env):
    """swich_back_annotation_column restores the configured column, makes
    sure it exists and repaints (4396-4399)."""
    with _app(tk_root, annotate_env, annotation_column="annotate") as app:
        app.annotation_column = "temporary_col"
        app.filtered_paths_annotations = []
        app.swich_back_annotation_column()
        assert app.annotation_column == "annotate"
        assert app.orig_annotation_columns == "annotate"
        assert len(app.filtered_paths_annotations) == 3
    with sqlite3.connect(annotate_env["db_path"]) as con:
        cols = [r[1] for r in con.execute(
            'PRAGMA table_info("png_list")').fetchall()]
    assert "annotate" in cols
    assert "temporary_col" not in cols
