"""The mask module writes `relationships` when it finishes.

`spacr.filters.ensure_relationships_table` builds the table on demand, so
the Gate Editor has always worked. What it could not do is be CURRENT: the
table describes which nucleus is in which cell, and that is decided by the
masks. A run that changes the masks and leaves the old table behind leaves
an answer about objects that no longer exist.
"""

from __future__ import annotations

import sqlite3

import pytest


def test_write_relationships_rebuilds_rather_than_tops_up():
    """The distinction the whole change rests on.

    `write_relationships` exists separately from
    `ensure_relationships_table` precisely so the mask step can say "rebuild
    this" without the caller having to know about a flag.
    """
    import inspect

    from spacr import filters

    source = inspect.getsource(filters.write_relationships)
    assert "rebuild=True" in source, (
        "write_relationships no longer forces a rebuild, so a second mask "
        "run would leave the first run's relationships in place")


def test_the_mask_pipeline_calls_it_once_per_source_folder():
    """Inside the per-folder loop, not after it.

    `db_path` is the loop variable. A call placed after the loop runs for
    the LAST folder only, so a multi-plate run would leave every other
    plate without the table -- and would look fine, because the one plate
    the developer checked would have it.
    """
    import inspect

    from spacr import core

    source = inspect.getsource(core.preprocess_generate_masks)
    assert "write_relationships" in source, (
        "the mask pipeline no longer writes the relationships table")

    lines = source.splitlines()
    call = next(i for i, line in enumerate(lines)
                if "write_relationships(db_path)" in line)
    loop = next(i for i, line in enumerate(lines)
                if "for source_folder in source_folders" in line)
    stamp = next(i for i, line in enumerate(lines)
                 if "ledger.stamp(db_path)" in line)
    assert loop < call, "the call is outside the per-folder loop"
    # Same indentation as the stamp beside it means the same block.
    indent = lambda i: len(lines[i]) - len(lines[i].lstrip())
    assert indent(call) >= indent(stamp), (
        "the call sits shallower than ledger.stamp, so it is not in the "
        "per-folder body")


def test_a_failure_to_write_does_not_fail_the_run():
    """Masking succeeded. The table is rebuilt on demand anyway.

    Failing here throws away hours of segmentation to protect a lookup
    that costs seconds.
    """
    import inspect

    from spacr import core

    source = inspect.getsource(core.preprocess_generate_masks)
    start = source.index("write_relationships(db_path)")
    window = source[max(0, start - 400):start + 400]
    assert "except Exception" in window, (
        "the relationships write is not wrapped; a failure would take the "
        "whole mask run with it")


def test_ensure_relationships_table_still_builds_on_demand(tmp_path):
    """The Gate Editor's fallback must survive this change."""
    from spacr.filters import ensure_relationships_table

    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE cell (plate TEXT, rowID TEXT, "
                     "columnID TEXT, fieldID TEXT, object_label INTEGER)")
        conn.execute("INSERT INTO cell VALUES ('p1','A','1','f1',1)")

    frame = ensure_relationships_table(str(db))
    assert frame is not None
