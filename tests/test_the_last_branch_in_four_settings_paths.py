"""Four more last branches, in code that turns a user's choices into settings.

Three of the four are "the user left this blank" -- the case a fixture written
by someone filling every field in never produces, and the case a real user
produces constantly.
"""
from __future__ import annotations

import os
import sqlite3

import pytest


# ---------------------------------------------------------------------------
# picture_settings.to_crop_settings — arc 536 -> 542, no object type chosen
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# picture_settings.to_crop_settings — arc 536 -> 542, no object type chosen
# ---------------------------------------------------------------------------

def _picture(**items):
    """A minimal stand-in for the picture settings object the function reads."""

    class _Picture:
        def __init__(self, values):
            self._values = values

        def get(self, key, default=None):
            return self._values.get(key, default)

        def items(self):
            return dict(self._values)

    return _Picture(items)


def test_streaming_from_a_database_without_an_object_type_sets_no_array():
    """The ``if object_type:`` branch not taken.

    The object type names the coordinate columns, and a blank one means the
    user has not said which object to cut yet. Writing ``object_array = ""``
    would hand the montage worker an empty name to look coordinate columns up
    by, which fails later and further away. Leaving the key ABSENT is what
    lets the downstream default apply.
    """
    from spacr.picture_settings import STREAM_FROM_DB, to_crop_settings

    out = to_crop_settings({"crop_source": STREAM_FROM_DB,
                            "object_type": "   "})

    assert "object_array" not in out


def test_streaming_from_a_database_with_an_object_type_names_the_array():
    """The taken side, lower-cased and stripped as the reader expects."""
    from spacr.picture_settings import STREAM_FROM_DB, to_crop_settings

    out = to_crop_settings({"crop_source": STREAM_FROM_DB,
                            "object_type": "  Cell "})

    assert out["object_array"] == "cell"


# ---------------------------------------------------------------------------
# regression_backends.backend_install_offer — arc 673 -> 677
# ---------------------------------------------------------------------------

def test_a_missing_package_for_a_wired_backend_is_offered_without_the_caveat(
        monkeypatch):
    """The ``if not spec['implemented']:`` branch not taken.

    The caveat warns that installing will not make the entry choosable. On a
    backend spaCR DOES route fits through, that sentence is false, and a false
    warning is worse than none -- the user declines an install that would have
    worked. pyfixest is wired and fits 'ols', so the offer must be a plain
    install offer with no caveat attached.

    ``package_installed`` is forced to False because whether pyfixest happens
    to be present in the test environment is not what is under test.
    """
    from spacr import regression_backends as rb

    monkeypatch.setattr(rb, "package_installed", lambda *_a, **_k: False)
    offer = rb.backend_install_offer("pyfixest", regression_type="ols")

    assert "pyfixest" in offer.message
    assert "not installed in" in offer.message
    assert "routes no fit through this backend yet" not in offer.message


def test_a_missing_package_for_an_unwired_backend_carries_the_caveat(monkeypatch):
    """The taken side: installing numpyro will not make the entry choosable.

    Saying so is the difference between a user installing a package that helps
    and one installing a package that changes nothing they can see.
    """
    from spacr import regression_backends as rb

    monkeypatch.setattr(rb, "package_installed", lambda *_a, **_k: False)
    spec = dict(rb.REGRESSION_BACKENDS["numpyro"])
    assert not spec["implemented"], "fixture assumes numpyro is still unwired"

    offer = rb.backend_install_offer("numpyro",
                                     regression_type=spec["types"][0])

    assert "routes no fit through this backend yet" in offer.message


# ---------------------------------------------------------------------------
# umap_annotations.write_umap_annotations — arc 61 -> 64, the column is there
# ---------------------------------------------------------------------------

def _png_list_db(path, *, column=None, rows=("a.png", "b.png")):
    """A database with a ``png_list`` table, optionally already annotated."""
    conn = sqlite3.connect(path)
    try:
        extra = f", {column} INTEGER" if column else ""
        conn.execute(f'CREATE TABLE png_list (png_path TEXT{extra})')
        conn.executemany('INSERT INTO png_list (png_path) VALUES (?)',
                         [(r,) for r in rows])
        conn.commit()
    finally:
        conn.close()


def test_a_second_annotation_pass_reuses_the_column_it_already_added(tmp_path):
    """The ``if column not in present:`` branch not taken.

    Re-scoring a UMAP is the ordinary way this is used -- the user annotates,
    looks, and annotates again. ALTER TABLE ADD COLUMN on a column that is
    already there raises OperationalError, so taking this branch a second time
    would make every re-annotation fail. The guard is what makes the operation
    repeatable, and it had only ever been tested on a fresh column.
    """
    from spacr.umap_annotations import write_umap_annotations

    db = tmp_path / "measurements.db"
    _png_list_db(str(db), column="umap_group")
    records = [{"db_path": str(db), "db_png_path": "a.png"},
               {"db_path": str(db), "db_png_path": "b.png"}]

    updated, skipped = write_umap_annotations(records, [1, 2], "umap_group")

    assert updated == 2 and skipped == 0
    conn = sqlite3.connect(str(db))
    try:
        got = dict(conn.execute(
            'SELECT png_path, umap_group FROM png_list').fetchall())
    finally:
        conn.close()
    assert got == {"a.png": 1, "b.png": 2}


def test_a_first_pass_adds_the_column(tmp_path):
    """The taken side, so the reuse above is visibly the second visit."""
    from spacr.umap_annotations import write_umap_annotations

    db = tmp_path / "measurements.db"
    _png_list_db(str(db))
    records = [{"db_path": str(db), "db_png_path": "a.png"}]

    updated, _skipped = write_umap_annotations(records, [7], "umap_group")

    assert updated == 1
    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute(
            'SELECT umap_group FROM png_list WHERE png_path = "a.png"'
        ).fetchone()[0] == 7
    finally:
        conn.close()


def test_a_database_with_no_png_list_skips_its_whole_group(tmp_path):
    """The ``if not present:`` guard above, which the two tests must not hit."""
    from spacr.umap_annotations import write_umap_annotations

    db = tmp_path / "other.db"
    sqlite3.connect(str(db)).close()
    records = [{"db_path": str(db), "db_png_path": "a.png"}]

    updated, skipped = write_umap_annotations(records, [1], "umap_group")

    assert (updated, skipped) == (0, 1)


# ---------------------------------------------------------------------------
# ports.port_problems — line 907, the per-port entry point
# ---------------------------------------------------------------------------

def test_a_port_pointing_at_nothing_reports_a_missing_input(tmp_path):
    """The whole body of ``port_problems``, which nothing had called.

    It is the per-port entry point for callers holding a Port without a module
    key -- documented as returning the same problems as the full readiness
    check. A documented public function with no test is a promise nobody has
    checked, and this one is two lines from the private helper it wraps, which
    is exactly how such a wrapper drifts.
    """
    from spacr.ports import Port, port_problems

    port = Port(kind="merged_arrays", role="merged", path="merged",
                pattern="*.npy", required=True, min_count=1)

    problems = port_problems(port, str(tmp_path), sample=0)

    assert problems
    assert any("merged" in str(getattr(p, "message", p)) or
               "merged" in str(p) for p in problems)


def test_a_satisfied_port_reports_nothing(tmp_path):
    """The empty tuple the docstring promises, so the test above is a contrast."""
    import numpy as np
    from spacr.ports import Port, port_problems

    merged = tmp_path / "merged"
    merged.mkdir()
    np.save(merged / "plate1_A01_F001.npy", np.zeros((4, 4), dtype=np.uint16))

    port = Port(kind="merged_arrays", role="merged", path="merged",
                pattern="*.npy", required=True, min_count=1)

    assert port_problems(port, str(tmp_path), sample=0) == ()
