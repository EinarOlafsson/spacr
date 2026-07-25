"""Coverage for ``AnnotateApp.open_deep_spacr_window`` (spacr/gui_elements.py).

The Deep-SPACR window is a Tk ``Toplevel`` full of form widgets whose "Run"
button assembles a settings dict and hands it to ``spacr.deep_spacr.deep_spacr``
on a daemon thread. These tests

  * build the window against a real (but withdrawn) Tk toplevel,
  * locate widgets structurally (by label text / Tk class), never by index,
  * drive the three ``dataset_mode`` panels, the header tab toggles and the
    Run/Cancel buttons, and
  * assert on the settings dict that actually reaches ``deep_spacr``.

Everything is CPU-only and offline: ``deep_spacr`` itself is replaced by a
recorder, and every ``messagebox`` call is captured so no modal dialog can
block the run. Tests skip cleanly when no display is available (``tk_root``).
"""
from __future__ import annotations

import builtins
import json
import sqlite3
import time

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _hide_toplevels(monkeypatch):
    """Keep every Toplevel the code under test opens unmapped."""
    import tkinter as tk

    real_toplevel = tk.Toplevel

    class _HiddenToplevel(real_toplevel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            try:
                self.withdraw()
            except Exception:
                pass

    monkeypatch.setattr(tk, "Toplevel", _HiddenToplevel)
    return _HiddenToplevel


@pytest.fixture(autouse=True)
def dialogs(monkeypatch):
    """Capture messagebox calls instead of opening modal dialogs."""
    from tkinter import messagebox

    calls = []

    def _record(kind):
        def _fn(*args, **kwargs):
            calls.append((kind, args[0] if args else None,
                          args[1] if len(args) > 1 else None))
            return True
        return _fn

    for name in ("showwarning", "showerror", "showinfo"):
        monkeypatch.setattr(messagebox, name, _record(name))
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: True)
    return calls


@pytest.fixture
def png_db(tmp_path):
    """A measurements DB whose png_list carries a few annotation columns."""
    db_dir = tmp_path / "measurements"
    db_dir.mkdir()
    db_path = db_dir / "measurements.db"
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            'CREATE TABLE png_list ('
            ' png_path TEXT, prcfo TEXT, test INTEGER, parasite INTEGER,'
            ' annotate INTEGER)'
        )
        con.executemany(
            'INSERT INTO png_list VALUES (?,?,?,?,?)',
            [(f"/img/cell_{i}.png", f"plate1_A01_1_o{i}", i % 2, i % 3, None)
             for i in range(4)],
        )
        con.commit()
    finally:
        con.close()
    return db_path


def _make_app(root, db_path, src="/tmp/spacr_src", annotation_column="annotate"):
    """A real AnnotateApp instance with only the attributes the window needs.

    ``__init__`` builds an entire annotation grid and starts a DB worker
    thread; ``open_deep_spacr_window`` needs none of that, so the object is
    allocated directly and given the exact attribute surface the method
    touches.
    """
    from tkinter import ttk

    from spacr.gui_elements import AnnotateApp, set_dark_style

    style = set_dark_style(ttk.Style())
    app = AnnotateApp.__new__(AnnotateApp)
    app.root = root
    app.db_path = str(db_path)
    app.src = src
    app.annotation_column = annotation_column
    app.bg_color = style["bg_color"]
    app.fg_color = style["fg_color"]
    app.active_color = style["active_color"]
    app.inactive_color = style["inactive_color"]
    app.font_style = ("Arial", 12)

    app.gui_texts = []
    app.update_gui_text = app.gui_texts.append

    app.ensure_calls = []

    def _ensure(source_columns, target_column="class_column", force_rebuild=True):
        app.ensure_calls.append((list(source_columns), target_column,
                                 bool(force_rebuild)))
        return f"{target_column}__built"

    app.ensure_multi_annot_from_selection = _ensure
    return app


@pytest.fixture
def deep_app(tk_root, png_db):
    """AnnotateApp stand-in wired to the synthetic png_list DB."""
    return _make_app(tk_root, png_db)


@pytest.fixture
def fake_deep_spacr(monkeypatch):
    """Replace spacr.deep_spacr.deep_spacr with a recorder."""
    state = {"calls": [], "exc": None}

    def _fake(settings):
        state["calls"].append(dict(settings))
        if state["exc"] is not None:
            raise state["exc"]

    monkeypatch.setattr("spacr.deep_spacr.deep_spacr", _fake)
    return state


# ---------------------------------------------------------------------------
# Widget locators
# ---------------------------------------------------------------------------

def _descendants(widget):
    out = []
    for child in widget.winfo_children():
        out.append(child)
        out.extend(_descendants(child))
    return out


def _by_class(widget, tk_class):
    return [w for w in _descendants(widget) if w.winfo_class() == tk_class]


def _labelframe(widget, text):
    for frame in _by_class(widget, "Labelframe"):
        if str(frame.cget("text")) == text:
            return frame
    raise AssertionError(f"no LabelFrame titled {text!r}")


def _form(container):
    """Map ``label text at column 0`` -> ``widget at column 1`` in a grid form."""
    labels, widgets = {}, {}
    for child in container.winfo_children():
        info = child.grid_info()
        if not info:
            continue
        row, col = int(info["row"]), int(info["column"])
        if col == 0 and child.winfo_class() == "Label":
            labels[row] = str(child.cget("text"))
        elif col == 1:
            widgets[row] = child
    return {labels.get(row, ""): w for row, w in widgets.items() if labels.get(row, "")}


class DeepWindow:
    """Structural facade over the Deep-SPACR window's widget tree."""

    def __init__(self, root):
        tops = [w for w in root.winfo_children()
                if w.winfo_class() == "Toplevel"
                and str(w.title()).startswith("Deep SPACR")]
        assert len(tops) == 1, f"expected 1 Deep SPACR window, got {len(tops)}"
        self.win = tops[0]
        self.notebook = _by_class(self.win, "TNotebook")[0]
        paned = _by_class(self.win, "Panedwindow")[0]
        self.gen = _form(paned.winfo_children()[0])
        self.ann_panel = _labelframe(self.win, "Annotation columns")
        self.meta_panel = _labelframe(self.win, "Metadata rules (JSON)")
        self.meas_panel = _labelframe(self.win, "Measurement selection")
        self.listbox = _by_class(self.ann_panel, "Listbox")[0]
        self.meta_entry = _by_class(self.meta_panel, "Entry")[0]
        self.meas = _form(self.meas_panel.winfo_children()[0])
        self.basic = _form(_labelframe(self.win, "Basic"))
        self.advanced = _form(_labelframe(self.win, "Advanced"))
        self.inference = _form(_labelframe(self.win, "Inference"))

    # -- helpers ----------------------------------------------------------
    def set_mode(self, mode):
        cbx = self.gen["dataset_mode"]
        cbx.set(mode)
        cbx.event_generate("<<ComboboxSelected>>")
        self.win.update()

    @staticmethod
    def set_text(widget, value):
        import tkinter as tk
        widget.delete(0, tk.END)
        widget.insert(0, value)

    def checkbutton(self, text):
        for chk in _by_class(self.win, "Checkbutton"):
            if str(chk.cget("text")) == text:
                return chk
        raise AssertionError(f"no Checkbutton labelled {text!r}")

    def button(self, text):
        for btn in _by_class(self.win, "TButton"):
            if str(btn.cget("text")) == text:
                return btn
        raise AssertionError(f"no button labelled {text!r}")

    def tab_states(self):
        return [self.notebook.tab(i, "state") for i in range(3)]

    def tab_texts(self):
        return [self.notebook.tab(i, "text") for i in range(3)]

    def run(self):
        self.button("Run").invoke()


def _open(app):
    app.open_deep_spacr_window()
    return DeepWindow(app.root)


def _wait_for_worker(app, timeout=15.0):
    """Block until the background _worker thread reported its final message."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(app.gui_texts) >= 2:
            return app.gui_texts
        time.sleep(0.01)
    raise AssertionError(f"deep_spacr worker never finished: {app.gui_texts}")


# ---------------------------------------------------------------------------
# 1. Window construction
# ---------------------------------------------------------------------------

def test_window_builds_three_tabs_and_action_buttons(deep_app):
    dw = _open(deep_app)
    assert dw.win.title() == "Deep SPACR — Train"
    assert dw.tab_texts() == ["Generate training dataset", "Train", "Apply model"]
    # Run/Cancel live on the window, not inside the notebook.
    assert dw.button("Run").winfo_exists()
    assert dw.button("Cancel").winfo_exists()
    # Defaults land in the widgets.
    assert dw.gen["size (cropped PNG side)"].get() == "224"
    assert dw.gen["image_size (model input)"].get() == "224"
    assert dw.gen["file_type / png_type"].get() == "cell_png"
    assert dw.basic["epochs"].get() == "100"
    assert dw.basic["batch_size"].get() == "64"
    assert dw.advanced["gradient_accumulation_steps"].get() == "4"
    assert dw.inference["score_threshold"].get() == "0.5"
    # list-valued defaults are rendered with repr(), not stringified elementwise
    assert dw.gen["class_metadata (list-of-lists)"].get() == "[['c1'], ['c2']]"
    assert dw.gen["classes (list)"].get() == "['nc', 'pc']"
    assert dw.gen["annotated_classes (list)"].get() == "[1, 2]"


def test_cancel_button_destroys_window_without_running(deep_app, fake_deep_spacr):
    dw = _open(deep_app)
    dw.button("Cancel").invoke()
    assert dw.win.winfo_exists() == 0
    assert fake_deep_spacr["calls"] == []
    assert deep_app.gui_texts == []


def test_annotation_listbox_lists_png_list_columns(deep_app):
    import tkinter as tk

    dw = _open(deep_app)
    items = list(dw.listbox.get(0, tk.END))
    # png_path and prcfo are deliberately skipped; everything else is offered.
    assert items == ["test", "parasite", "annotate"]
    assert dw.listbox.cget("selectmode") == tk.EXTENDED


def test_listbox_is_empty_when_db_cannot_be_opened(tk_root, tmp_path):
    """A directory path makes sqlite3.connect raise -> the except branch runs."""
    unopenable = tmp_path / "not_a_db_dir"
    unopenable.mkdir()
    with pytest.raises(sqlite3.OperationalError):     # the injected failure
        sqlite3.connect(str(unopenable), timeout=10)
    app = _make_app(tk_root, unopenable)
    dw = _open(app)
    assert dw.listbox.size() == 0
    # The rest of the window is still fully built.
    assert dw.tab_texts()[0] == "Generate training dataset"


# ---------------------------------------------------------------------------
# 2. dataset_mode panel switching
# ---------------------------------------------------------------------------

def _managers(dw):
    return (dw.ann_panel.winfo_manager(),
            dw.meta_panel.winfo_manager(),
            dw.meas_panel.winfo_manager())


def test_default_mode_shows_only_the_metadata_panel(deep_app):
    """deep_spacr_defaults ships dataset_mode='metadata'."""
    dw = _open(deep_app)
    assert dw.gen["dataset_mode"].get() == "metadata"
    assert _managers(dw) == ("", "pack", "")


@pytest.mark.parametrize("mode,expected", [
    ("annotation", ("pack", "", "")),
    ("metadata", ("", "pack", "")),
    ("measurement", ("", "", "pack")),
    ("nonsense", ("", "", "")),
])
def test_toggle_gen_right_packs_exactly_one_panel(deep_app, mode, expected):
    dw = _open(deep_app)
    dw.set_mode(mode)
    assert _managers(dw) == expected


@pytest.mark.xfail(strict=True, reason=(
    "BUG: _set_disabled_state only iterates the LabelFrame's direct children, "
    "which are Frames with no -state option, so the inputs of a hidden "
    "dataset_mode panel are never disabled"))
def test_hidden_panel_inputs_are_disabled(deep_app):
    dw = _open(deep_app)
    dw.set_mode("annotation")
    assert str(dw.meas["measurement (csv: columns)"].cget("state")) == "disabled"


def test_insert_example_button_fills_parsable_json(deep_app):
    dw = _open(deep_app)
    assert dw.meta_entry.get() == ""
    dw.button("Insert example").invoke()
    rules = json.loads(dw.meta_entry.get())
    assert [r["name"] for r in rules] == ["test_1", "test_2", "parasite_1"]
    assert rules[2]["where"] == [{"column": "parasite", "op": "==", "value": 1}]


# ---------------------------------------------------------------------------
# 3. Header master toggles drive notebook tab state
# ---------------------------------------------------------------------------

def test_header_toggles_enable_and_disable_notebook_tabs(deep_app):
    dw = _open(deep_app)
    assert dw.tab_states() == ["normal", "normal", "normal"]

    dw.checkbutton("Generate training dataset").invoke()
    assert dw.tab_states() == ["disabled", "normal", "normal"]

    dw.checkbutton("Apply model to dataset").invoke()
    assert dw.tab_states() == ["disabled", "normal", "disabled"]

    dw.checkbutton("Generate training dataset").invoke()
    assert dw.tab_states() == ["normal", "normal", "disabled"]


# ---------------------------------------------------------------------------
# 4. Run: annotation mode
# ---------------------------------------------------------------------------

def test_run_annotation_mode_multi_selection_builds_class_column(
        deep_app, fake_deep_spacr):
    dw = _open(deep_app)
    dw.set_mode("annotation")
    dw.listbox.selection_set(0)
    dw.listbox.selection_set(2)

    # exercise the parser fallbacks + non-default widget values
    dw.gen["sample (rows, optional)"].set("25")
    dw.set_text(dw.gen["tables (csv)"], ",,")          # -> parts empty -> fallback
    dw.set_text(dw.gen["class_metadata (list-of-lists)"], "[[")  # literal_eval fails
    dw.set_text(dw.gen["custom_measurement (optional)"], "my_meas")
    dw.set_text(dw.gen["file_type / png_type"], "cyto_png")
    dw.set_text(dw.inference["model_path (optional override)"], "/models/m.pth")
    dw.set_text(dw.inference["dataset (apply on this path)"], "/data/apply_here")
    dw.set_text(dw.advanced["custom_model_path"], "/models/custom.pth")
    dw.advanced["n_jobs (DataLoader workers)"].set("2")
    dw.checkbutton("augment").invoke()

    dw.run()
    assert dw.win.winfo_exists() == 0, "Run must close the window"
    assert _wait_for_worker(deep_app) == ["Deep SPACR: preparing…", "Deep SPACR: done."]

    assert deep_app.ensure_calls == [(["test", "annotate"], "class_column", True)]
    (settings,) = fake_deep_spacr["calls"]
    assert settings["dataset_mode"] == "annotation"
    assert settings["use_db_columns"] is True
    assert settings["annotation_column"] == "class_column__built"
    assert settings["sample"] == 25
    assert settings["tables"] is None            # ",," -> no usable parts
    assert settings["class_metadata"] == [["c1"], ["c2"]]  # literal_eval fallback
    assert settings["custom_measurement"] == "my_meas"
    assert settings["file_type"] == "cyto_png" == settings["png_type"]
    assert settings["model_path"] == "/models/m.pth"
    assert settings["dataset"] == "/data/apply_here"
    assert settings["custom_model_path"] == "/models/custom.pth"
    assert settings["n_jobs"] == 2
    assert settings["augment"] is True
    assert settings["src"] == deep_app.src
    # annotation mode strips the other modes' keys
    for key in ("metadata_rules", "measurement", "threshold"):
        assert key not in settings
    # types are coerced away from the widgets' strings
    assert isinstance(settings["size"], int)
    assert isinstance(settings["learning_rate"], float)
    assert isinstance(settings["train"], bool)


def test_run_annotation_mode_single_selection_uses_that_column_name(
        deep_app, fake_deep_spacr):
    dw = _open(deep_app)
    dw.set_mode("annotation")
    dw.listbox.selection_set(1)          # 'parasite'
    dw.run()
    _wait_for_worker(deep_app)

    assert deep_app.ensure_calls == [(["parasite"], "parasite", True)]
    (settings,) = fake_deep_spacr["calls"]
    assert settings["annotation_column"] == "parasite__built"


def test_run_annotation_mode_without_selection_warns_and_aborts(
        deep_app, fake_deep_spacr, dialogs):
    dw = _open(deep_app)
    dw.set_mode("annotation")
    assert dw.listbox.curselection() == ()
    dw.run()

    assert dialogs == [("showwarning", "No DB columns selected",
                        "Select at least one annotation column or uncheck "
                        "the DB option.")]
    assert fake_deep_spacr["calls"] == []
    assert deep_app.ensure_calls == []
    assert deep_app.gui_texts == []
    assert dw.win.winfo_exists() == 1, "aborted run must leave the window open"


def test_run_annotation_mode_with_db_columns_off_uses_app_column(
        deep_app, fake_deep_spacr):
    dw = _open(deep_app)
    dw.set_mode("annotation")
    dw.checkbutton("Use selected DB columns as classes").invoke()   # -> False
    dw.set_text(dw.gen["tables (csv)"], "cell, nucleus")
    dw.set_text(dw.gen["file_metadata (csv)"], "plateID,rowID")
    dw.set_text(dw.inference["model_path (optional override)"], "")
    # blanked list-literal fields fall back to the defaults, they do not
    # become None
    dw.set_text(dw.gen["classes (list)"], "")
    dw.set_text(dw.gen["annotated_classes (list)"], "   ")
    dw.run()
    _wait_for_worker(deep_app)

    (settings,) = fake_deep_spacr["calls"]
    assert settings["classes"] == ["nc", "pc"]
    assert settings["annotated_classes"] == [1, 2]
    assert settings["use_db_columns"] is False
    assert settings["annotation_column"] == deep_app.annotation_column
    assert deep_app.ensure_calls == []
    assert settings["tables"] == ["cell", "nucleus"]
    assert settings["file_metadata"] == ["plateID", "rowID"]
    assert settings["custom_measurement"] is None
    assert settings["sample"] is None
    assert settings["model_path"] == ""       # untouched default, not overridden


def test_run_unknown_mode_skips_all_mode_specific_blocks(
        deep_app, fake_deep_spacr):
    dw = _open(deep_app)
    dw.set_mode("nonsense")
    dw.run()
    _wait_for_worker(deep_app)

    (settings,) = fake_deep_spacr["calls"]
    assert settings["dataset_mode"] == "nonsense"
    assert "metadata_rules" not in settings
    assert "measurement" not in settings
    assert deep_app.ensure_calls == []
    # annotation_column survives from the defaults dict
    assert settings["annotation_column"] == "annotate"


# ---------------------------------------------------------------------------
# 5. Run: metadata mode
# ---------------------------------------------------------------------------

def test_run_metadata_mode_parses_json_rules(deep_app, fake_deep_spacr):
    dw = _open(deep_app)
    dw.set_mode("metadata")
    dw.button("Insert example").invoke()
    dw.run()
    _wait_for_worker(deep_app)

    (settings,) = fake_deep_spacr["calls"]
    assert [r["name"] for r in settings["metadata_rules"]] == [
        "test_1", "test_2", "parasite_1"]
    for key in ("measurement", "threshold", "annotation_column",
                "db_annotation_columns", "use_db_columns"):
        assert key not in settings


def test_run_metadata_mode_falls_back_to_python_literal(deep_app, fake_deep_spacr):
    dw = _open(deep_app)
    dw.set_mode("metadata")
    # single quotes -> json.loads raises -> ast.literal_eval succeeds
    dw.set_text(dw.meta_entry, "[{'name': 'lit', 'where': []}]")
    dw.run()
    _wait_for_worker(deep_app)

    (settings,) = fake_deep_spacr["calls"]
    assert settings["metadata_rules"] == [{"name": "lit", "where": []}]


@pytest.mark.parametrize("raw", ["", "   ", "not json at all {{{", "[]"])
def test_run_metadata_mode_rejects_unusable_rules(
        deep_app, fake_deep_spacr, dialogs, raw):
    dw = _open(deep_app)
    dw.set_mode("metadata")
    dw.set_text(dw.meta_entry, raw)
    dw.run()

    assert [c[1] for c in dialogs] == ["Metadata rules"]
    assert fake_deep_spacr["calls"] == []
    assert dw.win.winfo_exists() == 1


# ---------------------------------------------------------------------------
# 6. Run: measurement mode
# ---------------------------------------------------------------------------

def test_run_measurement_mode_single_column_numeric_threshold(
        deep_app, fake_deep_spacr):
    dw = _open(deep_app)
    dw.set_mode("measurement")
    dw.set_text(dw.meas["measurement (csv: columns)"], "cell_area")
    dw.set_text(dw.meas["threshold (float or q1..q9)"], "0.42")
    dw.run()
    _wait_for_worker(deep_app)

    (settings,) = fake_deep_spacr["calls"]
    assert settings["measurement"] == "cell_area"      # single -> scalar
    assert settings["threshold"] == pytest.approx(0.42)
    assert isinstance(settings["threshold"], float)
    for key in ("metadata_rules", "annotation_column", "db_annotation_columns",
                "use_db_columns"):
        assert key not in settings


def test_run_measurement_mode_multi_column_quantile_threshold(
        deep_app, fake_deep_spacr):
    dw = _open(deep_app)
    dw.set_mode("measurement")
    dw.set_text(dw.meas["measurement (csv: columns)"], "cell_area, nucleus_area")
    dw.set_text(dw.meas["threshold (float or q1..q9)"], "q7")
    dw.run()
    _wait_for_worker(deep_app)

    (settings,) = fake_deep_spacr["calls"]
    assert settings["measurement"] == ["cell_area", "nucleus_area"]
    assert settings["threshold"] == "q7"               # non-numeric kept verbatim


@pytest.mark.parametrize("cols,threshold,title", [
    ("", "q8", "Measurement"),
    ("   ", "0.5", "Measurement"),
    ("cell_area", "", "Measurement"),
])
def test_run_measurement_mode_rejects_missing_inputs(
        deep_app, fake_deep_spacr, dialogs, cols, threshold, title):
    dw = _open(deep_app)
    dw.set_mode("measurement")
    dw.set_text(dw.meas["measurement (csv: columns)"], cols)
    dw.set_text(dw.meas["threshold (float or q1..q9)"], threshold)
    dw.run()

    assert [c[1] for c in dialogs] == [title]
    assert fake_deep_spacr["calls"] == []
    assert dw.win.winfo_exists() == 1


# ---------------------------------------------------------------------------
# 7. Worker error handling
# ---------------------------------------------------------------------------

def test_run_reports_worker_exception_through_gui_text(deep_app, fake_deep_spacr):
    fake_deep_spacr["exc"] = RuntimeError("no CUDA for you")
    dw = _open(deep_app)
    dw.set_mode("metadata")
    dw.button("Insert example").invoke()
    dw.run()
    texts = _wait_for_worker(deep_app)

    assert texts[0] == "Deep SPACR: preparing…"
    assert texts[1] == "Deep SPACR error: no CUDA for you"
    assert len(fake_deep_spacr["calls"]) == 1


# ---------------------------------------------------------------------------
# 8. Defaults-driven widget population
# ---------------------------------------------------------------------------

def _patch_defaults(monkeypatch, **overrides):
    import spacr.settings as S

    base = S.deep_spacr_defaults({})
    base.update(overrides)
    monkeypatch.setattr(S, "deep_spacr_defaults", lambda settings: dict(base))
    return base


def test_defaults_with_list_valued_fields_populate_entries(
        tk_root, png_db, monkeypatch):
    _patch_defaults(
        monkeypatch,
        sample=25,
        tables=["cell", "nucleus"],
        file_metadata=["plateID", "rowID"],
        custom_measurement="cm",
        measurement=["m1", "m2"],
        threshold=0.25,
        train_channels=["r", "g"],
        balance_to_smallest=False,
    )
    app = _make_app(tk_root, png_db, src="/data/proj")
    dw = _open(app)

    assert dw.gen["sample (rows, optional)"].get() == "25"
    assert dw.gen["tables (csv)"].get() == "cell,nucleus"
    assert dw.gen["file_metadata (csv)"].get() == "plateID,rowID"
    assert dw.gen["custom_measurement (optional)"].get() == "cm"
    assert dw.meas["measurement (csv: columns)"].get() == "m1,m2"
    assert dw.meas["threshold (float or q1..q9)"].get() == "0.25"
    assert dw.basic["train_channels"].get() == "['r', 'g']"
    # src is truthy -> it wins over the defaults' own 'src'
    assert dw.inference["dataset (apply on this path)"].get() == ""


def test_defaults_with_scalar_fields_and_empty_src(tk_root, png_db, monkeypatch):
    base = _patch_defaults(
        monkeypatch,
        file_metadata="plateID",
        measurement="m1",
        train_channels="not-a-list",
        train=False,
        test=False,
    )
    base.pop("dataset")          # force the defaults['src'] fallback at line 5762
    app = _make_app(tk_root, png_db, src="", annotation_column=None)
    dw = _open(app)

    assert dw.gen["sample (rows, optional)"].get() == ""
    assert dw.gen["tables (csv)"].get() == ""
    assert dw.gen["file_metadata (csv)"].get() == "plateID"
    assert dw.gen["custom_measurement (optional)"].get() == ""
    assert dw.meas["measurement (csv: columns)"].get() == "m1"
    assert dw.basic["train_channels"].get() == "['r','g','b']"
    # empty src -> defaults['src'] ('path') -> dataset entry
    assert dw.inference["dataset (apply on this path)"].get() == "path"
    # train and test both off -> the Train tab starts disabled
    assert dw.tab_states() == ["normal", "disabled", "normal"]


def test_torchvision_import_failure_falls_back_to_static_model_list(
        tk_root, png_db, monkeypatch):
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "torchvision" or name.startswith("torchvision."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    app = _make_app(tk_root, png_db)
    dw = _open(app)
    monkeypatch.undo()

    assert tuple(dw.basic["model_type"].cget("values")) == (
        "resnet18", "resnet34", "resnet50", "densenet121", "mobilenet_v2")


def test_model_list_comes_from_torchvision_when_importable(deep_app):
    pytest.importorskip("torchvision")
    dw = _open(deep_app)
    values = tuple(dw.basic["model_type"].cget("values"))
    assert "resnet50" in values and "densenet121" in values
    assert len(values) > 5
    # the default model_type is preselected even though it is not in the list
    assert dw.basic["model_type"].get() == "maxvit_t"


# ---------------------------------------------------------------------------
# 9. Integration: the Train button of a fully constructed AnnotateApp
# ---------------------------------------------------------------------------

def test_train_button_of_real_annotate_app_opens_the_window(tk_root, png_db):
    """End-to-end: a real AnnotateApp's 'Train' button opens this window."""
    import tkinter as tk

    from spacr.gui_elements import AnnotateApp

    top = tk.Toplevel(tk_root)
    app = None
    try:
        app = AnnotateApp(root=top, db_path=str(png_db),
                          src=str(png_db.parent.parent), image_size=200)
        tk_root.update_idletasks()
        app.dl_train_button.invoke()
        dw = DeepWindow(top)
        assert dw.tab_texts() == ["Generate training dataset", "Train",
                                  "Apply model"]
        # the real app's annotation column reaches the window's DB listbox
        assert "annotate" in list(dw.listbox.get(0, "end"))
        dw.button("Cancel").invoke()
        assert dw.win.winfo_exists() == 0
    finally:
        if app is not None:
            app.terminate = True
            try:
                app.update_queue.put(app.SENTINEL)
            except Exception:
                pass
            if app.db_update_thread.is_alive():
                app.db_update_thread.join(timeout=5)
            assert not app.db_update_thread.is_alive()
        try:
            top.destroy()
        except Exception:
            pass
