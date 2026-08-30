"""Three more modules that were one branch short, and what each branch guards.

Same shape as tests/test_the_last_branch_in_six_modules.py: full statement
coverage, one untaken arc, and in each case the untaken side is the one where
the function decides to leave something alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# baseline.apply — arc 208 -> 210, a frame without the column being shifted
# ---------------------------------------------------------------------------

def test_a_table_without_the_effect_column_is_returned_unshifted():
    """The ``if column in out.columns:`` branch not taken.

    ``apply`` is handed every panel a run produces, not only the coefficient
    table -- exports, summaries and diagnostics go through the same call. A
    frame that has no ``coefficient`` column is therefore an ordinary input,
    not an error, and the correct behaviour is to hand back a copy unchanged.
    Raising or inventing the column would break the panels that share the call.
    """
    from spacr.baseline import Baseline, apply

    baseline = Baseline(kind="mean", shift=0.25, n=12, sentence="against mean")
    frame = pd.DataFrame({"gene": ["a", "b"], "p_value": [0.01, 0.4]})

    out = apply(frame, baseline)

    assert list(out.columns) == ["gene", "p_value"]
    assert out["p_value"].tolist() == [0.01, 0.4]
    assert out is not frame                    # still a copy, as documented


def test_a_table_with_the_effect_column_is_shifted():
    """The taken side, so the test above is a contrast and not an accident."""
    from spacr.baseline import Baseline, apply

    baseline = Baseline(kind="mean", shift=0.25, n=12, sentence="against mean")
    frame = pd.DataFrame({"coefficient": [1.0, 0.25]})

    out = apply(frame, baseline)

    assert out["coefficient"].tolist() == [0.75, 0.0]
    assert frame["coefficient"].tolist() == [1.0, 0.25]   # input untouched


def test_a_baseline_that_does_not_move_returns_the_frame_itself():
    """The early return above both, which makes the copy above meaningful."""
    from spacr.baseline import Baseline, apply

    frame = pd.DataFrame({"coefficient": [1.0]})
    still = apply(frame, Baseline(kind="zero", shift=0.0, n=0, sentence="zero"))

    assert still is frame


# ---------------------------------------------------------------------------
# checkpoint.CheckpointStore.update — arc 285 -> 287, metadata without a status
# ---------------------------------------------------------------------------

def _store(tmp_path, **kwargs):
    from spacr.checkpoint import CheckpointStore

    return CheckpointStore(str(tmp_path / "run.json"), workflow="demo",
                           signature={"input": "plate1"}, boundary="field",
                           **kwargs)


def test_metadata_can_be_recorded_without_touching_the_status(tmp_path):
    """The ``if status is not None:`` branch not taken.

    Progress metadata is written far more often than status is: a run updates
    "which field am I on" continuously and changes status a handful of times.
    Writing a status of ``None`` on every metadata update would erase the real
    one, so the guard is what keeps a resumed run knowing what it was doing --
    and the common path through this function had never been tested.
    """
    store = _store(tmp_path)
    store.update(meta={"phase": "starting"}, status="partial")
    store.update(meta={"last_field": 7})                 # no status this time

    resumed = _store(tmp_path, resume=True)

    assert resumed.status == "partial"                   # NOT overwritten
    assert resumed.meta.get("last_field") == 7
    assert resumed.meta.get("phase") == "starting"       # merged, not replaced


def test_a_status_can_be_recorded_without_any_metadata(tmp_path):
    """The other guard in the same function, by the same argument."""
    store = _store(tmp_path)
    store.update(meta={"phase": "starting"})
    store.update(status="finished")

    resumed = _store(tmp_path, resume=True)

    assert resumed.status == "finished"
    assert resumed.meta.get("phase") == "starting"


def test_an_update_with_neither_still_writes_the_document(tmp_path):
    """Both guards false: flush() is unconditional and that is the contract."""
    store = _store(tmp_path)
    store.update()
    assert _store(tmp_path, resume=True).resumed


# ---------------------------------------------------------------------------
# gene_facts._segment_index — arc 412 -> 418, every segment slot present
# ---------------------------------------------------------------------------

def test_a_protein_using_every_segment_slot_is_read_to_the_end(monkeypatch):
    """The ``for n in range(...)`` loop completing instead of breaking.

    The loop stops at the first absent ``tm_<n>_start`` column, which is how
    it avoids 64 lookups for a protein with two helices. The loop RUNNING OUT
    -- a table that really does carry all 64 slots -- had never happened under
    test, and it is the case where the break is not what ends the loop. A
    multi-pass membrane protein is exactly the input that gets there.
    """
    from spacr import gene_facts
    from spacr import annotation

    n_slots = gene_facts._MAX_SEGMENTS
    row = {"gene_nr": ["TGGT1_231640"]}
    for n in range(1, n_slots + 1):
        row[f"tm_{n}_start"] = [10 * n]
        row[f"tm_{n}_end"] = [10 * n + 4]
        row[f"tm_{n}_length"] = [5]
    frame = pd.DataFrame(row)

    monkeypatch.setattr(annotation, "supplementary", lambda *a, **k: frame)
    gene_facts._segment_index.cache_clear()
    try:
        index = gene_facts._segment_index()
    finally:
        gene_facts._segment_index.cache_clear()

    segments = index.get("231640", ())
    assert len(segments) == n_slots
    # Sorted by start, which is what the return comprehension promises.
    assert [s.start for s in segments] == sorted(s.start for s in segments)


def test_the_scan_stops_at_the_first_absent_slot(monkeypatch):
    """The break, so the completion above is visibly the other outcome."""
    from spacr import gene_facts
    from spacr import annotation

    frame = pd.DataFrame({"gene_nr": ["TGGT1_231641"],
                          "tm_1_start": [10], "tm_1_end": [14], "tm_1_length": [5],
                          "tm_2_start": [30], "tm_2_end": [34], "tm_2_length": [5]})

    monkeypatch.setattr(annotation, "supplementary", lambda *a, **k: frame)
    gene_facts._segment_index.cache_clear()
    try:
        index = gene_facts._segment_index()
    finally:
        gene_facts._segment_index.cache_clear()

    assert len(index.get("231641", ())) == 2
