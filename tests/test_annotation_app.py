"""
Tests for the annotation app (spacr.gui_elements.AnnotateApp +
spacr.app_annotate).

Headless: we construct AnnotateApp against a Toplevel + synthetic sqlite
DB and verify its side effects on the DB, then always tear down the
background worker thread before returning so pytest can exit cleanly.
"""
from __future__ import annotations

import os
import sqlite3
import contextlib
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.gui  # needs a display


# ---------------------------------------------------------------------------
# Contextmanager helper that ALWAYS cleans up the background thread the
# AnnotateApp constructor starts. Without this, pytest hangs at the end
# of the module because the worker never exits.
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _make_annotate_app(tk_root, **kwargs):
    import tkinter as tk
    from spacr.gui_elements import AnnotateApp

    top = tk.Toplevel(tk_root)
    app = None
    try:
        app = AnnotateApp(root=top, **kwargs)
        tk_root.update_idletasks()
        yield app
    finally:
        # Tear down the worker thread reliably.
        if app is not None:
            app.terminate = True
            try:
                app.update_queue.put(app.SENTINEL)
            except Exception:
                pass
            if app.db_update_thread.is_alive():
                app.db_update_thread.join(timeout=5)
        try:
            top.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Minimal fixture: a src dir with measurements/measurements.db that carries
# a png_list table (that's what AnnotateApp reads).
# ---------------------------------------------------------------------------

@pytest.fixture
def annotate_src(tmp_path):
    """A src directory shaped like a spacr project so AnnotateApp can start."""
    (tmp_path / "measurements").mkdir()
    (tmp_path / "images").mkdir()
    db_path = tmp_path / "measurements" / "measurements.db"

    n = 5
    png_list = pd.DataFrame({
        "png_path": [str(tmp_path / "images" / f"cell_{i}.png") for i in range(n)],
        "prc": [f"plate1_A01_{i:03d}" for i in range(n)],
        "cell_id": [f"o{i}" for i in range(n)],
        "plateID": ["plate1"] * n,
        "rowID": ["A"] * n,
        "columnID": ["1"] * n,
        "fieldID": [f"{i:03d}" for i in range(n)],
    })
    con = sqlite3.connect(db_path)
    try:
        png_list.to_sql("png_list", con, index=False)
    finally:
        con.close()
    return {"src": str(tmp_path), "db_path": str(db_path), "n_png": n}


# ---------------------------------------------------------------------------
# 1. AnnotateApp constructs against a real sqlite DB.
# ---------------------------------------------------------------------------

def test_annotate_app_constructs(tk_root, annotate_src):
    with _make_annotate_app(
        tk_root,
        db_path=annotate_src["db_path"],
        src=annotate_src["src"],
        image_size=200,
    ) as app:
        assert app.db_path == annotate_src["db_path"]
        assert app.src == annotate_src["src"]
        assert app.image_size == (200, 200)
        # The background DB-writer thread should be alive while the app is up.
        assert app.db_update_thread.is_alive()


# ---------------------------------------------------------------------------
# 2. _ensure_annotation_column mutates the DB when the column is missing
# ---------------------------------------------------------------------------

def test_annotate_app_ensures_annotation_column_exists(tk_root, annotate_src):
    with _make_annotate_app(
        tk_root,
        db_path=annotate_src["db_path"],
        src=annotate_src["src"],
        annotation_column="annotate",
    ):
        pass  # constructor did the work
    # After construction, the annotate column should be present on png_list.
    with sqlite3.connect(annotate_src["db_path"]) as con:
        cols = [row[1] for row in con.execute("PRAGMA table_info(png_list)").fetchall()]
    assert "annotate" in cols, f"annotate column not added; got {cols}"


# ---------------------------------------------------------------------------
# 3. Custom annotation_column names round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("colname", ["annotate", "hit_class", "my_label"])
def test_annotate_app_accepts_custom_annotation_column(tk_root, tmp_path, colname):
    (tmp_path / "measurements").mkdir()
    db_path = tmp_path / "measurements" / "measurements.db"
    df = pd.DataFrame({
        "png_path": [str(tmp_path / "x.png")],
        "prc": ["p1_A01_001"], "cell_id": ["o1"],
        "plateID": ["p1"], "rowID": ["A"], "columnID": ["1"], "fieldID": ["001"],
    })
    with sqlite3.connect(db_path) as con:
        df.to_sql("png_list", con, index=False)

    with _make_annotate_app(
        tk_root, db_path=str(db_path), src=str(tmp_path), annotation_column=colname,
    ):
        pass

    with sqlite3.connect(db_path) as con:
        cols = [row[1] for row in con.execute("PRAGMA table_info(png_list)").fetchall()]
    assert colname in cols


# ---------------------------------------------------------------------------
# 4. AnnotateApp exposes the buttons it actually packs — regression check
#    for the missing-button bug we just fixed.
# ---------------------------------------------------------------------------

def test_annotate_app_has_all_packed_buttons(tk_root, annotate_src):
    """Every button referenced in AnnotateApp.__init__'s pack section must
    exist as an attribute. Regression test: skip_to_last_annotated_button
    was previously packed without ever being created, which crashed the
    constructor."""
    with _make_annotate_app(
        tk_root,
        db_path=annotate_src["db_path"],
        src=annotate_src["src"],
    ) as app:
        for name in (
            "next_button", "previous_button", "skip_to_last_annotated_button",
            "exit_button", "settings_button", "clear_button", "count_button",
            "dl_train_button",
        ):
            assert hasattr(app, name), f"AnnotateApp missing button {name}"


# ---------------------------------------------------------------------------
# 5. The convert_to_number helper (used by initiate_annotation_app)
# ---------------------------------------------------------------------------

def test_convert_to_number_int():
    from spacr.app_annotate import convert_to_number
    assert convert_to_number("42") == 42
    assert isinstance(convert_to_number("42"), int)


def test_convert_to_number_float():
    from spacr.app_annotate import convert_to_number
    v = convert_to_number("3.14")
    assert v == pytest.approx(3.14)
    assert isinstance(v, float)


def test_convert_to_number_rejects_garbage():
    from spacr.app_annotate import convert_to_number
    with pytest.raises(ValueError):
        convert_to_number("not_a_number")


# ---------------------------------------------------------------------------
# 6. Public entry points exist
# ---------------------------------------------------------------------------

def test_annotate_module_entry_points_exist():
    from spacr import app_annotate
    assert callable(getattr(app_annotate, "initiate_annotation_app", None))
    assert callable(getattr(app_annotate, "start_annotate_app", None))
